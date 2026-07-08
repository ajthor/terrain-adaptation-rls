"""Online baseline comparisons on one Phoenix scene."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from terrain_adaptation_rls.configuration import load_config
from terrain_adaptation_rls.data.load_data import load_scenes
from terrain_adaptation_rls.estimators.linear import linear_predict, solve_ridge_coefficients
from terrain_adaptation_rls.evaluation.diagnostic_plots import integrate_planar_deltas
from terrain_adaptation_rls.evaluation.fe_rls import (
    load_trained_function_encoder,
    scene_streaming_tensors,
)
from terrain_adaptation_rls.evaluation.metrics import (
    DEFAULT_LOGGED_K_STEP_HORIZONS,
    summarize_logged_k_step_metrics,
    summarize_prediction_metrics,
)
from terrain_adaptation_rls.methods.runtime import (
    FunctionEncoderBasisProvider,
    LinearBasisProvider,
    Observation,
    RuntimeInput,
    TorchCoefficientMethod,
)
from terrain_adaptation_rls.training.neuralfly import (
    load_neuralfly_style_basis,
    predict_neuralfly_style_batch,
)
from terrain_adaptation_rls.training.supervised import scene_supervised_batch
from terrain_adaptation_rls.models.maml import loss_fn as maml_loss_fn
from terrain_adaptation_rls.training.maml import adapt_model, load_trained_maml


METHOD_LABELS = {
    "fe_rls": "FE-RLS",
    "fe_prior_static": "FE prior-static",
    "fe_prior_rls": "FE prior-RLS",
    "fe_kalman": "FE-Kalman",
    "fe_sgd": "FE-SGD",
    "fe_window_ls": "FE-window LS",
    "offline_fe": "offline FE",
    "neuralfly_rls": "NeuralFly-style RLS",
    "neuralfly_prior_static": "NeuralFly prior-static",
    "neuralfly_prior_rls": "NeuralFly prior-RLS",
    "offline_neuralfly": "offline NeuralFly-style",
    "static_node": "static NODE",
    "maml_static": "MAML-static",
    "maml_online": "MAML-online",
    "linear_rls": "linear RLS",
    "offline_linear": "offline linear",
    "zero_delta": "zero delta",
}


@dataclass(frozen=True)
class OnlineBaselineArtifacts:
    """Summary of a completed online baseline comparison."""

    artifact_dir: Path
    summary: dict[str, object]


def run_online_baseline_comparison(
    *,
    fe_run_dir: str | Path,
    artifact_dir: str | Path,
    scene: str,
    neuralfly_run_dir: str | Path | None = None,
    node_run_dir: str | Path | None = None,
    maml_run_dir: str | Path | None = None,
    device: torch.device | str = "cpu",
    max_points: int = 512,
    start_index: int = 0,
    n_example_points: int | None = None,
    forgetting_factor: float = 0.95,
    initial_covariance: float = 1_000.0,
    measurement_noise: float = 1e-6,
    include_fe_variants: bool = True,
    kalman_process_noise: float = 0.0,
    fe_sgd_learning_rate: float = 1.0,
    fe_sgd_momentum: float = 0.0,
    fe_sgd_weight_decay: float = 0.0,
    fe_window_size: int = 100,
    fe_window_ridge: float = 1e-6,
    linear_include_bias: bool = True,
    include_prior_baselines: bool = False,
    prior_points_per_scene: int | None = None,
    prior_ridge: float = 1e-6,
    maml_inner_learning_rate: float | None = None,
    maml_inner_steps: int | None = None,
    include_recursive_k_step: bool = True,
    recursive_k_step_horizons: tuple[int, ...] = (1, 5, 10, 20, 50),
    recursive_k_step_max_rollouts: int = 64,
) -> OnlineBaselineArtifacts:
    """Compare trained and online update baselines on one held-out scene."""

    fe_run_dir = Path(fe_run_dir)
    neuralfly_run_path = None if neuralfly_run_dir is None else Path(neuralfly_run_dir)
    node_run_path = None if node_run_dir is None else Path(node_run_dir)
    maml_run_path = None if maml_run_dir is None else Path(maml_run_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fe_config = load_config(fe_run_dir / "resolved_config.json")
    if fe_config.platform is None:
        raise ValueError("FE training config must include platform")

    device = torch.device(device)
    inputs, targets = load_scenes([scene], fe_config.platform)[scene]
    xs, dt, target, time = scene_streaming_tensors(
        inputs=inputs,
        targets=targets,
        start_index=start_index,
        max_points=max_points,
        device=device,
    )

    n_examples = n_example_points or int(fe_config.training.get("n_example_points", 128))
    example_batch = scene_supervised_batch(
        inputs=inputs,
        targets=targets,
        n_example_points=n_examples,
        max_query_points=1,
        device=device,
        seed=fe_config.seed,
    )
    _, _, _, example_xs, example_dt, example_ys = example_batch

    predictions: dict[str, torch.Tensor] = {}
    coefficient_histories: dict[str, torch.Tensor] = {}
    recursive_predictors: dict[str, object] = {}
    resolved_maml_inner_lr: float | None = None
    resolved_maml_inner_steps: int | None = None

    fe_model = load_trained_function_encoder(fe_config, fe_run_dir, device=device)
    fe_provider = FunctionEncoderBasisProvider(fe_model)
    fe_coefficients, _ = fe_model.compute_coefficients((example_xs, example_dt), example_ys)
    predictions["offline_fe"] = fe_model((xs, dt), coefficients=fe_coefficients).detach().cpu()
    recursive_predictors["offline_fe"] = _FixedCoefficientPredictor(
        fe_provider,
        fe_coefficients.detach(),
    )

    prior_xs: torch.Tensor | None = None
    prior_dt: torch.Tensor | None = None
    prior_ys: torch.Tensor | None = None
    resolved_prior_points = (
        int(prior_points_per_scene)
        if prior_points_per_scene is not None
        else int(fe_config.training.get("n_example_points", 128))
    )
    if include_prior_baselines:
        prior_xs, prior_dt, prior_ys = _training_prior_examples(
            fe_config,
            device=device,
            points_per_scene=resolved_prior_points,
        )
        fe_prior_coefficients = _solve_prior_coefficients(
            fe_provider,
            xs=prior_xs,
            dt=prior_dt,
            ys=prior_ys,
            ridge=prior_ridge,
        )
        predictions["fe_prior_static"] = _predict_with_coefficients(
            fe_provider,
            xs=xs,
            dt=dt,
            coefficients=fe_prior_coefficients,
        ).detach().cpu()
        recursive_predictors["fe_prior_static"] = _FixedCoefficientPredictor(
            fe_provider,
            fe_prior_coefficients.detach(),
        )
        fe_prior_method = TorchCoefficientMethod(
            fe_provider,
            update_rule="rls",
            forgetting_factor=forgetting_factor,
            initial_covariance=initial_covariance,
            measurement_noise=measurement_noise,
            initial_coefficients=fe_prior_coefficients.detach(),
            device=device,
        )
        predictions["fe_prior_rls"], coefficient_histories["fe_prior_rls"] = (
            stream_runtime_method(
                fe_prior_method,
                xs=xs,
                dt=dt,
                target=target,
                time=time,
            )
        )
        recursive_predictors["fe_prior_rls"] = _HistoryCoefficientPredictor(
            fe_provider,
            coefficient_histories["fe_prior_rls"],
            initial_coefficients=fe_prior_coefficients.detach(),
            device=device,
        )

    fe_method = TorchCoefficientMethod(
        fe_provider,
        update_rule="rls",
        forgetting_factor=forgetting_factor,
        initial_covariance=initial_covariance,
        measurement_noise=measurement_noise,
        device=device,
    )
    predictions["fe_rls"], coefficient_histories["fe_rls"] = stream_runtime_method(
        fe_method,
        xs=xs,
        dt=dt,
        target=target,
        time=time,
    )
    recursive_predictors["fe_rls"] = _HistoryCoefficientPredictor(
        fe_provider,
        coefficient_histories["fe_rls"],
        device=device,
    )

    if include_fe_variants:
        fe_variant_methods = {
            "fe_kalman": TorchCoefficientMethod(
                fe_provider,
                update_rule="kalman",
                output_dim=6,
                initial_covariance=initial_covariance,
                measurement_noise=measurement_noise,
                process_noise=kalman_process_noise,
                device=device,
            ),
            "fe_sgd": TorchCoefficientMethod(
                fe_provider,
                update_rule="sgd",
                output_dim=6,
                learning_rate=fe_sgd_learning_rate,
                momentum=fe_sgd_momentum,
                weight_decay=fe_sgd_weight_decay,
                device=device,
            ),
            "fe_window_ls": TorchCoefficientMethod(
                fe_provider,
                update_rule="window_ls",
                output_dim=6,
                window_size=fe_window_size,
                ridge=fe_window_ridge,
                device=device,
            ),
        }
        for name, method in fe_variant_methods.items():
            predictions[name], coefficient_histories[name] = stream_runtime_method(
                method,
                xs=xs,
                dt=dt,
                target=target,
                time=time,
            )
            recursive_predictors[name] = _HistoryCoefficientPredictor(
                fe_provider,
                coefficient_histories[name],
                device=device,
            )

    if neuralfly_run_path is not None:
        neuralfly_config = load_config(neuralfly_run_path / "resolved_config.json")
        neuralfly_model = load_neuralfly_style_basis(
            neuralfly_config,
            neuralfly_run_path,
            device=device,
        )
        ridge = float(neuralfly_config.model.get("ridge", 1e-6))
        predictions["offline_neuralfly"] = predict_neuralfly_style_batch(
            neuralfly_model,
            (xs, dt, target, example_xs, example_dt, example_ys),
            ridge=ridge,
        ).detach().cpu()
        neuralfly_features = neuralfly_model(RuntimeInput(example_xs, example_dt))
        neuralfly_coefficients = solve_ridge_coefficients(
            neuralfly_features,
            example_ys,
            ridge=ridge,
        )
        recursive_predictors["offline_neuralfly"] = _FixedCoefficientPredictor(
            neuralfly_model,
            neuralfly_coefficients.detach(),
        )
        if include_prior_baselines:
            if prior_xs is None or prior_dt is None or prior_ys is None:
                prior_xs, prior_dt, prior_ys = _training_prior_examples(
                    fe_config,
                    device=device,
                    points_per_scene=resolved_prior_points,
                )
            neuralfly_prior_coefficients = _solve_prior_coefficients(
                neuralfly_model,
                xs=prior_xs,
                dt=prior_dt,
                ys=prior_ys,
                ridge=ridge,
            )
            predictions["neuralfly_prior_static"] = _predict_with_coefficients(
                neuralfly_model,
                xs=xs,
                dt=dt,
                coefficients=neuralfly_prior_coefficients,
            ).detach().cpu()
            recursive_predictors["neuralfly_prior_static"] = _FixedCoefficientPredictor(
                neuralfly_model,
                neuralfly_prior_coefficients.detach(),
            )
            neuralfly_prior_method = TorchCoefficientMethod(
                neuralfly_model,
                update_rule="rls",
                output_dim=6,
                forgetting_factor=forgetting_factor,
                initial_covariance=initial_covariance,
                measurement_noise=measurement_noise,
                initial_coefficients=neuralfly_prior_coefficients.detach(),
                device=device,
            )
            (
                predictions["neuralfly_prior_rls"],
                coefficient_histories["neuralfly_prior_rls"],
            ) = stream_runtime_method(
                neuralfly_prior_method,
                xs=xs,
                dt=dt,
                target=target,
                time=time,
            )
            recursive_predictors["neuralfly_prior_rls"] = _HistoryCoefficientPredictor(
                neuralfly_model,
                coefficient_histories["neuralfly_prior_rls"],
                initial_coefficients=neuralfly_prior_coefficients.detach(),
                device=device,
            )
        neuralfly_method = TorchCoefficientMethod(
            neuralfly_model,
            update_rule="rls",
            output_dim=6,
            forgetting_factor=forgetting_factor,
            initial_covariance=initial_covariance,
            measurement_noise=measurement_noise,
            device=device,
        )
        (
            predictions["neuralfly_rls"],
            coefficient_histories["neuralfly_rls"],
        ) = stream_runtime_method(
            neuralfly_method,
            xs=xs,
            dt=dt,
            target=target,
            time=time,
        )
        recursive_predictors["neuralfly_rls"] = _HistoryCoefficientPredictor(
            neuralfly_model,
            coefficient_histories["neuralfly_rls"],
            device=device,
        )

    if node_run_path is not None:
        node_config = load_config(node_run_path / "resolved_config.json")
        node_model = load_trained_neural_ode(node_config, node_run_path, device=device)
        predictions["static_node"] = node_model((xs, dt)).detach().cpu()
        recursive_predictors["static_node"] = _DirectModelPredictor(node_model)

    if maml_run_path is not None:
        maml_config = load_config(maml_run_path / "resolved_config.json")
        maml_model = load_trained_maml(maml_config, maml_run_path, device=device)
        maml_inner_lr = (
            float(maml_inner_learning_rate)
            if maml_inner_learning_rate is not None
            else float(
                maml_config.training.get(
                    "inner_learning_rate",
                    maml_config.training.get("inner_lr", 1e-2),
                )
            )
        )
        maml_steps = (
            int(maml_inner_steps)
            if maml_inner_steps is not None
            else int(maml_config.training.get("inner_steps", 1))
        )
        resolved_maml_inner_lr = maml_inner_lr
        resolved_maml_inner_steps = maml_steps
        predictions["maml_static"] = maml_model((xs, dt)).detach().cpu()
        recursive_predictors["maml_static"] = _DirectModelPredictor(maml_model)
        predictions["maml_online"] = stream_maml_online(
            maml_model,
            xs=xs,
            dt=dt,
            target=target,
            inner_lr=maml_inner_lr,
            inner_steps=maml_steps,
            device=device,
        )
        recursive_predictors["maml_online"] = _MAMLOnlinePredictor(
            maml_model,
            xs=xs,
            dt=dt,
            target=target,
            inner_lr=maml_inner_lr,
            inner_steps=maml_steps,
            device=device,
        )

    linear_provider = LinearBasisProvider(
        input_dim=9,
        output_dim=6,
        include_bias=linear_include_bias,
    )
    linear_example_features = linear_provider(RuntimeInput(example_xs, example_dt))
    linear_coefficients = solve_ridge_coefficients(
        linear_example_features,
        example_ys,
        ridge=1e-6,
    )
    linear_query_features = linear_provider(RuntimeInput(xs, dt))
    predictions["offline_linear"] = linear_predict(
        linear_query_features,
        linear_coefficients,
    ).detach().cpu()
    recursive_predictors["offline_linear"] = _FixedCoefficientPredictor(
        linear_provider,
        linear_coefficients.detach(),
    )

    linear_method = TorchCoefficientMethod(
        linear_provider,
        update_rule="rls",
        output_dim=6,
        forgetting_factor=forgetting_factor,
        initial_covariance=initial_covariance,
        measurement_noise=measurement_noise,
        device=device,
    )
    predictions["linear_rls"], coefficient_histories["linear_rls"] = stream_runtime_method(
        linear_method,
        xs=xs,
        dt=dt,
        target=target,
        time=time,
    )
    recursive_predictors["linear_rls"] = _HistoryCoefficientPredictor(
        linear_provider,
        coefficient_histories["linear_rls"],
        device=device,
    )

    target_cpu = target.detach().cpu()
    dt_cpu = dt.detach().cpu()
    time_cpu = time.detach().cpu()
    predictions["zero_delta"] = torch.zeros_like(target_cpu)
    recursive_predictors["zero_delta"] = _ZeroDeltaPredictor(device=device)
    recursive_k_step_metrics = (
        summarize_recursive_k_step_metrics(
            xs=xs,
            dt=dt,
            target=target,
            predictors=recursive_predictors,
            horizons=recursive_k_step_horizons,
            max_rollouts=recursive_k_step_max_rollouts,
        )
        if include_recursive_k_step
        else {}
    )

    summary = summarize_baseline_predictions(
        target=target_cpu,
        dt=dt_cpu,
        predictions=predictions,
        coefficient_histories=coefficient_histories,
        recursive_k_step_metrics=recursive_k_step_metrics,
        scene=scene,
        forgetting_factor=forgetting_factor,
        initial_covariance=initial_covariance,
        measurement_noise=measurement_noise,
        start_index=start_index,
        n_example_points=n_examples,
        include_fe_variants=include_fe_variants,
        kalman_process_noise=kalman_process_noise,
        fe_sgd_learning_rate=fe_sgd_learning_rate,
        fe_sgd_momentum=fe_sgd_momentum,
        fe_sgd_weight_decay=fe_sgd_weight_decay,
        fe_window_size=fe_window_size,
        fe_window_ridge=fe_window_ridge,
        linear_include_bias=linear_include_bias,
        include_prior_baselines=include_prior_baselines,
        prior_points_per_scene=resolved_prior_points,
        prior_ridge=prior_ridge,
        maml_inner_learning_rate=resolved_maml_inner_lr,
        maml_inner_steps=resolved_maml_inner_steps,
        include_recursive_k_step=include_recursive_k_step,
        recursive_k_step_horizons=recursive_k_step_horizons,
        recursive_k_step_max_rollouts=recursive_k_step_max_rollouts,
    )
    write_online_baseline_artifacts(
        artifact_dir,
        scene=scene,
        time=time_cpu,
        dt=dt_cpu,
        target=target_cpu,
        predictions=predictions,
        coefficient_histories=coefficient_histories,
        summary=summary,
    )
    return OnlineBaselineArtifacts(artifact_dir=artifact_dir, summary=summary)


def load_trained_neural_ode(
    config: object,
    train_run_dir: str | Path,
    *,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained static Neural ODE from a run directory."""

    from terrain_adaptation_rls.models.neural_ode import create_model

    n_basis = int(config.model.get("n_basis", 8))
    hidden_size = int(config.model.get("hidden_size", 128))
    model = create_model(device, n_basis=n_basis, hidden_size=hidden_size)
    state = torch.load(Path(train_run_dir) / "neural_ode_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _training_prior_examples(
    config: object,
    *,
    device: torch.device,
    points_per_scene: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample deterministic prior-solve examples from the training scenes."""

    platform = getattr(config, "platform", None)
    if platform is None:
        raise ValueError("Prior coefficient solve requires config.platform")
    train_scenes = [str(scene) for scene in config.data.get("train_scenes", ())]
    if not train_scenes:
        raise ValueError("Prior coefficient solve requires data.train_scenes")
    if points_per_scene <= 0:
        raise ValueError("prior_points_per_scene must be positive")

    loaded = load_scenes(train_scenes, platform)
    xs_chunks: list[torch.Tensor] = []
    dt_chunks: list[torch.Tensor] = []
    ys_chunks: list[torch.Tensor] = []
    base_seed = int(getattr(config, "seed", 0))
    for scene_index, scene in enumerate(train_scenes):
        inputs, targets = loaded[scene]
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"{scene} inputs and targets have different row counts")
        n_points = min(points_per_scene, int(inputs.shape[0]))
        generator = torch.Generator(device="cpu").manual_seed(
            base_seed + 104_729 * (scene_index + 1)
        )
        indices = torch.randperm(inputs.shape[0], generator=generator)[:n_points]
        selected_inputs = inputs[indices]
        selected_targets = targets[indices]
        xs = selected_inputs[:, 1:]
        dt = selected_targets[:, 0] - selected_inputs[:, 0]
        ys = selected_targets[:, 1:] - xs[:, :6]
        xs_chunks.append(xs)
        dt_chunks.append(dt)
        ys_chunks.append(ys)

    return (
        torch.cat(xs_chunks, dim=0).unsqueeze(0).to(device),
        torch.cat(dt_chunks, dim=0).unsqueeze(0).to(device),
        torch.cat(ys_chunks, dim=0).unsqueeze(0).to(device),
    )


@torch.no_grad()
def _solve_prior_coefficients(
    feature_provider: object,
    *,
    xs: torch.Tensor,
    dt: torch.Tensor,
    ys: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Solve one global coefficient vector for a learned basis."""

    features = feature_provider(RuntimeInput(xs, dt))
    return solve_ridge_coefficients(features, ys, ridge=ridge)


@torch.no_grad()
def _predict_with_coefficients(
    feature_provider: object,
    *,
    xs: torch.Tensor,
    dt: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Predict deltas for a coefficient-linear runtime model."""

    features = feature_provider(RuntimeInput(xs, dt))
    return linear_predict(features, coefficients)


@torch.no_grad()
def stream_runtime_method(
    method: TorchCoefficientMethod,
    *,
    xs: torch.Tensor,
    dt: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream one trajectory through a predict-before-update runtime method."""

    state = method.initial_state()
    predictions: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []

    for idx in range(xs.shape[1]):
        step_inputs = RuntimeInput(
            xs=xs[:, idx : idx + 1],
            dt=dt[:, idx : idx + 1],
        )
        step_target = target[:, idx : idx + 1]
        prediction = method.predict(state, step_inputs)
        state = method.update(
            state,
            Observation(
                inputs=step_inputs,
                target=step_target,
                time=float(time[idx].detach().cpu()),
            ),
        )
        predictions.append(prediction.squeeze(0).squeeze(0).detach().cpu())
        coefficients.append(state.coefficients.squeeze(0).detach().cpu())

    return torch.stack(predictions).unsqueeze(0), torch.stack(coefficients)


def stream_maml_online(
    model: torch.nn.Module,
    *,
    xs: torch.Tensor,
    dt: torch.Tensor,
    target: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Stream a MAML initialization with predict-before-update adaptation."""

    adapted_model = copy.deepcopy(model)
    adapted_model.to(device)
    predictions: list[torch.Tensor] = []
    for idx in range(xs.shape[1]):
        adapted_model.eval()
        with torch.no_grad():
            prediction = adapted_model((xs[:, idx : idx + 1], dt[:, idx : idx + 1]))
        predictions.append(prediction.squeeze(0).squeeze(0).detach().cpu())
        with torch.enable_grad():
            adapt_model(
                adapted_model,
                support_data=(
                    xs[:, idx : idx + 1],
                    dt[:, idx : idx + 1],
                    target[:, idx : idx + 1],
                ),
                loss_fn=maml_loss_fn,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                device=device,
                clone=False,
            )

    return torch.stack(predictions).unsqueeze(0)


@torch.no_grad()
def summarize_recursive_k_step_metrics(
    *,
    xs: torch.Tensor,
    dt: torch.Tensor,
    target: torch.Tensor,
    predictors: dict[str, object],
    horizons: tuple[int, ...],
    max_rollouts: int,
) -> dict[str, dict[str, float]]:
    """Summarize recursive open-loop rollout error for each method."""

    xs_single = xs.squeeze(0)
    dt_single = dt.squeeze(0)
    target_delta = target.squeeze(0)
    n_steps = int(target_delta.shape[0])
    valid_horizons = sorted({int(horizon) for horizon in horizons if 0 < int(horizon) <= n_steps})
    if not valid_horizons or max_rollouts <= 0:
        return {name: {} for name in predictors}

    max_horizon = max(valid_horizons)
    max_start = n_steps - max_horizon
    if max_start < 0:
        return {name: {} for name in predictors}

    starts = torch.linspace(
        0,
        max_start,
        steps=min(max_rollouts, max_start + 1),
        device=xs_single.device,
    ).round().to(torch.long).unique(sorted=True)
    summaries: dict[str, dict[str, float]] = {}
    for name, predictor in predictors.items():
        method_summary: dict[str, float] = {}
        horizon_errors: dict[int, list[float]] = {horizon: [] for horizon in valid_horizons}
        horizon_accumulated: dict[int, list[float]] = {
            horizon: [] for horizon in valid_horizons
        }
        horizon_rmses: dict[int, list[float]] = {horizon: [] for horizon in valid_horizons}
        horizon_integral_square: dict[int, list[float]] = {
            horizon: [] for horizon in valid_horizons
        }

        for start_tensor in starts:
            start = int(start_tensor.item())
            begin_rollout = getattr(predictor, "begin_rollout", None)
            if begin_rollout is not None:
                begin_rollout(start)
            state = xs_single[start, :6].clone()
            step_errors: list[float] = []
            step_dts: list[float] = []
            for offset in range(max_horizon):
                index = start + offset
                control = xs_single[index, 6:]
                delta = predictor.predict(
                    state=state,
                    control=control,
                    dt=dt_single[index],
                    start_index=start,
                )
                true_next = target_delta[index] + xs_single[index, :6]
                predicted_next = torch.cat((delta[:3], state[3:6] + delta[3:6]))
                step_errors.append(float(torch.linalg.norm(predicted_next - true_next)))
                step_dts.append(float(dt_single[index].detach().cpu()))
                state = _roll_legacy_body_state(state, delta)

                horizon = offset + 1
                if horizon in horizon_errors:
                    square_integral = sum(
                        error * error * step_dt
                        for error, step_dt in zip(step_errors, step_dts)
                    )
                    duration = max(sum(step_dts), 1e-12)
                    horizon_errors[horizon].append(step_errors[-1])
                    horizon_accumulated[horizon].append(sum(step_errors))
                    horizon_rmses[horizon].append((square_integral / duration) ** 0.5)
                    horizon_integral_square[horizon].append(square_integral)

        for horizon in valid_horizons:
            errors = horizon_errors[horizon]
            accumulated = horizon_accumulated[horizon]
            rmses = horizon_rmses[horizon]
            integral_square = horizon_integral_square[horizon]
            prefix = f"recursive_k{horizon}"
            method_summary[f"{prefix}_n_rollouts"] = float(len(errors))
            method_summary[f"{prefix}_final_step_error_mean"] = _mean(errors)
            method_summary[f"{prefix}_final_step_error_median"] = _median(errors)
            method_summary[f"{prefix}_final_step_error_p95"] = _quantile_list(errors, 0.95)
            method_summary[f"{prefix}_accumulated_error_mean"] = _mean(accumulated)
            method_summary[f"{prefix}_accumulated_error_median"] = _median(accumulated)
            method_summary[f"{prefix}_accumulated_error_p95"] = _quantile_list(
                accumulated,
                0.95,
            )
            method_summary[f"{prefix}_trajectory_rmse_mean"] = _mean(rmses)
            method_summary[f"{prefix}_trajectory_rmse_median"] = _median(rmses)
            method_summary[f"{prefix}_trajectory_rmse_p95"] = _quantile_list(rmses, 0.95)
            method_summary[f"{prefix}_integral_square_error_mean"] = _mean(
                integral_square
            )
            method_summary[f"{prefix}_integral_square_error_median"] = _median(
                integral_square
            )
            method_summary[f"{prefix}_integral_square_error_p95"] = _quantile_list(
                integral_square,
                0.95,
            )
        summaries[name] = method_summary
    return summaries


def summarize_baseline_predictions(
    *,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    coefficient_histories: dict[str, torch.Tensor],
    dt: torch.Tensor | None = None,
    recursive_k_step_metrics: dict[str, dict[str, float]] | None = None,
    scene: str,
    forgetting_factor: float,
    initial_covariance: float,
    measurement_noise: float,
    start_index: int,
    n_example_points: int,
    include_fe_variants: bool = True,
    kalman_process_noise: float = 0.0,
    fe_sgd_learning_rate: float = 1.0,
    fe_sgd_momentum: float = 0.0,
    fe_sgd_weight_decay: float = 0.0,
    fe_window_size: int = 100,
    fe_window_ridge: float = 1e-6,
    linear_include_bias: bool = True,
    include_prior_baselines: bool = False,
    prior_points_per_scene: int = 128,
    prior_ridge: float = 1e-6,
    maml_inner_learning_rate: float | None = None,
    maml_inner_steps: int | None = None,
    include_recursive_k_step: bool = True,
    recursive_k_step_horizons: tuple[int, ...] = (1, 5, 10, 20, 50),
    recursive_k_step_max_rollouts: int = 64,
) -> dict[str, object]:
    """Compute scalar metrics for all baseline predictions."""

    target_delta = target.squeeze(0)
    zero_prediction = predictions.get("zero_delta", torch.zeros_like(target))
    zero_prediction_delta = zero_prediction.squeeze(0)
    zero_error = torch.linalg.norm(zero_prediction_delta - target_delta, dim=-1)
    zero_metrics = summarize_prediction_metrics(
        target=target_delta,
        prediction=zero_prediction_delta,
        dt=dt,
    )

    method_summaries: dict[str, object] = {}
    for name, prediction in predictions.items():
        prediction_delta = prediction.squeeze(0)
        error = torch.linalg.norm(prediction_delta - target_delta, dim=-1)
        metrics = summarize_prediction_metrics(
            target=target_delta,
            prediction=prediction_delta,
            dt=dt,
            zero_metrics=zero_metrics,
        )
        metrics.update(
            summarize_logged_k_step_metrics(
                target=target_delta,
                prediction=prediction_delta,
                dt=dt,
                horizons=DEFAULT_LOGGED_K_STEP_HORIZONS,
            )
        )
        if recursive_k_step_metrics is not None:
            metrics.update(recursive_k_step_metrics.get(name, {}))
        method_summaries[name] = {
            "label": METHOD_LABELS.get(name, name),
            **metrics,
        }
        # Keep these aliases explicit for older notebooks and scripts.
        method_summaries[name]["mean_error"] = float(error.mean())
        method_summaries[name]["final_accumulated_error"] = float(error.sum())
        method_summaries[name]["mean_error_to_zero_delta_ratio"] = _safe_ratio(
            float(error.mean()),
            float(zero_error.mean()),
        )
        method_summaries[name]["mse"] = float(
            torch.nn.functional.mse_loss(prediction_delta, target_delta)
        )

    coefficient_summaries: dict[str, object] = {}
    for name, history in coefficient_histories.items():
        norms = torch.linalg.norm(history, dim=-1)
        coefficient_summaries[name] = {
            "label": METHOD_LABELS.get(name, name),
            "n_coeff": int(history.shape[-1]),
            "final_norm": float(norms[-1]),
            "mean_norm": float(norms.mean()),
            "max_norm": float(norms.max()),
            "final": [float(value) for value in history[-1]],
        }

    return {
        "scene": scene,
        "n_steps": int(target_delta.shape[0]),
        "start_index": start_index,
        "n_example_points": n_example_points,
        "methods": method_summaries,
        "coefficients": coefficient_summaries,
        "rls_parameters": {
            "forgetting_factor": forgetting_factor,
            "initial_covariance": initial_covariance,
            "measurement_noise": measurement_noise,
        },
        "fe_variant_parameters": {
            "include": include_fe_variants,
            "kalman_process_noise": kalman_process_noise,
            "sgd_learning_rate": fe_sgd_learning_rate,
            "sgd_momentum": fe_sgd_momentum,
            "sgd_weight_decay": fe_sgd_weight_decay,
            "window_size": fe_window_size,
            "window_ridge": fe_window_ridge,
        },
        "linear_baseline": {
            "include_bias": linear_include_bias,
        },
        "prior_baselines": {
            "include": include_prior_baselines,
            "points_per_scene": prior_points_per_scene,
            "ridge": prior_ridge,
            "source": "training_scenes",
        },
        "maml_parameters": {
            "inner_learning_rate": maml_inner_learning_rate,
            "inner_steps": maml_inner_steps,
        },
        "recursive_k_step": {
            "include": include_recursive_k_step,
            "horizons": list(recursive_k_step_horizons),
            "max_rollouts": recursive_k_step_max_rollouts,
            "rollout_update": "legacy_body_velocity_frame",
        },
    }


def write_online_baseline_artifacts(
    artifact_dir: Path,
    *,
    scene: str,
    time: torch.Tensor,
    dt: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    coefficient_histories: dict[str, torch.Tensor],
    summary: dict[str, object],
) -> None:
    """Write CSV, JSON, and plot artifacts for online baseline comparisons."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_streaming_predictions_csv(
        artifact_dir / "streaming_predictions.csv",
        time=time,
        target=target,
        predictions=predictions,
    )
    write_streaming_error_plot(
        artifact_dir / "streaming_error.png",
        scene=scene,
        time=time,
        target=target,
        predictions=predictions,
    )
    write_streaming_components_plot(
        artifact_dir / "streaming_components.png",
        scene=scene,
        time=time,
        target=target,
        predictions=predictions,
    )
    write_streaming_delta_scale_plot(
        artifact_dir / "streaming_delta_scale.png",
        scene=scene,
        time=time,
        target=target,
        predictions=predictions,
    )
    write_streaming_trajectory_plot(
        artifact_dir / "streaming_trajectory.png",
        scene=scene,
        target=target,
        predictions=predictions,
        include_offline=True,
    )
    write_streaming_trajectory_plot(
        artifact_dir / "streaming_trajectory_online.png",
        scene=scene,
        target=target,
        predictions=predictions,
        include_offline=False,
    )
    write_coefficient_norm_plot(
        artifact_dir / "coefficient_norms.png",
        scene=scene,
        time=time,
        coefficient_histories=coefficient_histories,
    )
    write_delta_time_summary(
        artifact_dir / "trajectory_summary.json",
        scene=scene,
        dt=dt,
        target=target,
        predictions=predictions,
    )
    write_logged_k_step_metrics_csv(
        artifact_dir / "logged_k_step_metrics.csv",
        summary=summary,
    )


def write_streaming_predictions_csv(
    path: Path,
    *,
    time: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    """Write target and prediction values for each streaming step."""

    target_delta = target.squeeze(0)
    prediction_delta = {name: value.squeeze(0) for name, value in predictions.items()}
    errors = {
        name: torch.linalg.norm(value - target_delta, dim=-1)
        for name, value in prediction_delta.items()
    }

    with path.open("w", newline="") as f:
        fieldnames = ["index", "time"]
        fieldnames += [f"target_{idx}" for idx in range(target_delta.shape[-1])]
        for name in predictions:
            fieldnames.append(f"{name}_error")
            fieldnames += [f"{name}_prediction_{idx}" for idx in range(target_delta.shape[-1])]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(target_delta.shape[0]):
            row = {
                "index": idx,
                "time": float(time[idx]),
            }
            row.update(
                {
                    f"target_{dim}": float(target_delta[idx, dim])
                    for dim in range(target_delta.shape[-1])
                }
            )
            for name, prediction in prediction_delta.items():
                row[f"{name}_error"] = float(errors[name][idx])
                row.update(
                    {
                        f"{name}_prediction_{dim}": float(prediction[idx, dim])
                        for dim in range(target_delta.shape[-1])
                    }
                )
            writer.writerow(row)


def write_streaming_error_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    """Write per-step and accumulated error for every compared method."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_delta = target.squeeze(0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for name, prediction in predictions.items():
        error = torch.linalg.norm(prediction.squeeze(0) - target_delta, dim=-1)
        kwargs = _plot_kwargs(name)
        axes[0].plot(time, error, label=METHOD_LABELS.get(name, name), **kwargs)
        axes[1].plot(time, torch.cumsum(error, dim=0), label=METHOD_LABELS.get(name, name), **kwargs)
    axes[0].set_ylabel("error norm")
    axes[0].legend(ncol=2)
    axes[1].set_xlabel("relative time [s]")
    axes[1].set_ylabel("accumulated error")
    axes[1].legend(ncol=2)
    fig.suptitle(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_streaming_components_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    """Write per-dimension traces for the main online baselines."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_delta = target.squeeze(0)
    selected = _selected_prediction_names(predictions)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for dim, ax in enumerate(axes.ravel()):
        ax.plot(time, target_delta[:, dim], label="target", linewidth=1.3, color="black")
        for name in selected:
            ax.plot(
                time,
                predictions[name].squeeze(0)[:, dim],
                label=METHOD_LABELS.get(name, name),
                **_plot_kwargs(name),
            )
        ax.set_ylabel(f"dim {dim}")
    axes.ravel()[0].legend(ncol=2)
    axes.ravel()[-1].set_xlabel("relative time [s]")
    axes.ravel()[-2].set_xlabel("relative time [s]")
    fig.suptitle(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_streaming_delta_scale_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    """Write delta norm and cumulative planar distance for compared methods."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_delta = target.squeeze(0)
    selected = _selected_prediction_names(predictions, include_offline=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time, torch.linalg.norm(target_delta, dim=-1), label="target", color="black")
    axes[1].plot(
        time,
        torch.cumsum(torch.linalg.norm(target_delta[:, :2], dim=-1), dim=0),
        label="target",
        color="black",
    )
    for name in selected:
        prediction = predictions[name].squeeze(0)
        axes[0].plot(
            time,
            torch.linalg.norm(prediction, dim=-1),
            label=METHOD_LABELS.get(name, name),
            **_plot_kwargs(name),
        )
        axes[1].plot(
            time,
            torch.cumsum(torch.linalg.norm(prediction[:, :2], dim=-1), dim=0),
            label=METHOD_LABELS.get(name, name),
            **_plot_kwargs(name),
        )
    axes[0].set_ylabel("delta norm")
    axes[0].legend(ncol=2)
    axes[1].set_xlabel("relative time [s]")
    axes[1].set_ylabel("cumulative planar delta")
    axes[1].legend(ncol=2)
    fig.suptitle(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_streaming_trajectory_plot(
    path: Path,
    *,
    scene: str,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    include_offline: bool,
) -> None:
    """Write integrated local trajectory comparison."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_pose = integrate_planar_deltas(target.squeeze(0))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(target_pose[:, 0], target_pose[:, 1], label="target", linewidth=1.6, color="black")
    for name in _selected_prediction_names(predictions, include_offline=include_offline):
        pose = integrate_planar_deltas(predictions[name].squeeze(0))
        ax.plot(
            pose[:, 0],
            pose[:, 1],
            label=METHOD_LABELS.get(name, name),
            **_plot_kwargs(name),
        )
    ax.scatter(target_pose[0, 0], target_pose[0, 1], s=20, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("integrated local x")
    ax.set_ylabel("integrated local y")
    title = scene if include_offline else f"{scene}: online methods"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_coefficient_norm_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    coefficient_histories: dict[str, torch.Tensor],
) -> None:
    """Write coefficient norm histories for online methods."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, history in coefficient_histories.items():
        ax.plot(
            time,
            torch.linalg.norm(history, dim=-1),
            label=METHOD_LABELS.get(name, name),
            **_plot_kwargs(name),
        )
    ax.set_xlabel("relative time [s]")
    ax.set_ylabel("coefficient norm")
    ax.set_title(scene)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_delta_time_summary(
    path: Path,
    *,
    scene: str,
    dt: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    """Write compact trajectory-scale metrics for every method."""

    target_delta = target.squeeze(0)
    payload: dict[str, object] = {
        "scene": scene,
        "n_steps": int(target_delta.shape[0]),
        "dt": {
            "mean": float(dt.mean()),
            "min": float(dt.min()),
            "max": float(dt.max()),
            "total": float(dt.sum()),
        },
        "target_path_length": _path_length(target_delta),
        "methods": {},
    }
    for name, prediction in predictions.items():
        prediction_delta = prediction.squeeze(0)
        payload["methods"][name] = {
            "label": METHOD_LABELS.get(name, name),
            "prediction_path_length": _path_length(prediction_delta),
            "prediction_to_target_path_length_ratio": _safe_ratio(
                _path_length(prediction_delta),
                _path_length(target_delta),
            ),
            "prediction_delta_norm_mean": float(
                torch.linalg.norm(prediction_delta, dim=-1).mean()
            ),
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_logged_k_step_metrics_csv(
    path: Path,
    *,
    summary: dict[str, object],
) -> None:
    """Write flattened logged-input k-step metrics for every method."""

    methods = summary.get("methods", {})
    rows: list[dict[str, object]] = []
    if isinstance(methods, dict):
        for method, method_summary in methods.items():
            if not isinstance(method_summary, dict):
                continue
            row = {
                "method": method,
                "label": method_summary.get("label", method),
            }
            for key, value in method_summary.items():
                if str(key).startswith("logged_k"):
                    row[str(key)] = value
            rows.append(row)

    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class _FixedCoefficientPredictor:
    def __init__(self, feature_provider: object, coefficients: torch.Tensor) -> None:
        self.feature_provider = feature_provider
        self.coefficients = coefficients

    def predict(
        self,
        *,
        state: torch.Tensor,
        control: torch.Tensor,
        dt: torch.Tensor,
        start_index: int,
    ) -> torch.Tensor:
        del start_index
        return _coefficient_predict(
            self.feature_provider,
            self.coefficients,
            state=state,
            control=control,
            dt=dt,
        )


class _HistoryCoefficientPredictor:
    def __init__(
        self,
        feature_provider: object,
        history: torch.Tensor,
        *,
        initial_coefficients: torch.Tensor | None = None,
        device: torch.device,
    ) -> None:
        self.feature_provider = feature_provider
        self.history = history
        self.initial_coefficients = initial_coefficients
        self.device = device

    def predict(
        self,
        *,
        state: torch.Tensor,
        control: torch.Tensor,
        dt: torch.Tensor,
        start_index: int,
    ) -> torch.Tensor:
        coefficients = self._coefficients_for_start(start_index)
        return _coefficient_predict(
            self.feature_provider,
            coefficients,
            state=state,
            control=control,
            dt=dt,
        )

    def _coefficients_for_start(self, start_index: int) -> torch.Tensor:
        if start_index <= 0:
            if self.initial_coefficients is not None:
                coefficients = self.initial_coefficients.to(self.device)
                if coefficients.ndim == 1:
                    coefficients = coefficients.unsqueeze(0)
                return coefficients
            return torch.zeros(
                1,
                self.history.shape[-1],
                dtype=self.history.dtype,
                device=self.device,
            )
        index = min(start_index - 1, self.history.shape[0] - 1)
        return self.history[index].to(self.device).unsqueeze(0)


class _DirectModelPredictor:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def predict(
        self,
        *,
        state: torch.Tensor,
        control: torch.Tensor,
        dt: torch.Tensor,
        start_index: int,
    ) -> torch.Tensor:
        del start_index
        xs_step, dt_step = _rollout_input_tensors(state=state, control=control, dt=dt)
        return self.model((xs_step, dt_step)).squeeze(0).squeeze(0)


class _MAMLOnlinePredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        xs: torch.Tensor,
        dt: torch.Tensor,
        target: torch.Tensor,
        inner_lr: float,
        inner_steps: int,
        device: torch.device,
    ) -> None:
        self.initial_model = copy.deepcopy(model)
        self.xs = xs
        self.dt = dt
        self.target = target
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        self.rollout_model: torch.nn.Module | None = None
        self.rollout_start: int | None = None

    def begin_rollout(self, start_index: int) -> None:
        self.rollout_model = copy.deepcopy(self.initial_model).to(self.device)
        with torch.enable_grad():
            for idx in range(start_index):
                adapt_model(
                    self.rollout_model,
                    support_data=(
                        self.xs[:, idx : idx + 1],
                        self.dt[:, idx : idx + 1],
                        self.target[:, idx : idx + 1],
                    ),
                    loss_fn=maml_loss_fn,
                    inner_lr=self.inner_lr,
                    inner_steps=self.inner_steps,
                    device=self.device,
                    clone=False,
                )
        self.rollout_model.eval()
        self.rollout_start = start_index

    def predict(
        self,
        *,
        state: torch.Tensor,
        control: torch.Tensor,
        dt: torch.Tensor,
        start_index: int,
    ) -> torch.Tensor:
        if self.rollout_model is None or self.rollout_start != start_index:
            self.begin_rollout(start_index)
        assert self.rollout_model is not None
        xs_step, dt_step = _rollout_input_tensors(state=state, control=control, dt=dt)
        return self.rollout_model((xs_step, dt_step)).squeeze(0).squeeze(0)


class _ZeroDeltaPredictor:
    def __init__(self, *, device: torch.device) -> None:
        self.device = device

    def predict(
        self,
        *,
        state: torch.Tensor,
        control: torch.Tensor,
        dt: torch.Tensor,
        start_index: int,
    ) -> torch.Tensor:
        del state, control, dt, start_index
        return torch.zeros(6, dtype=torch.float32, device=self.device)


def _coefficient_predict(
    feature_provider: object,
    coefficients: torch.Tensor,
    *,
    state: torch.Tensor,
    control: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    xs_step, dt_step = _rollout_input_tensors(state=state, control=control, dt=dt)
    features = feature_provider(RuntimeInput(xs_step, dt_step))
    coefficients = coefficients.to(device=features.device, dtype=features.dtype)
    return linear_predict(features, coefficients).squeeze(0).squeeze(0)


def _rollout_input_tensors(
    *,
    state: torch.Tensor,
    control: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs_step = torch.cat((state, control)).reshape(1, 1, -1)
    dt_step = dt.reshape(1, 1)
    return xs_step, dt_step


def _roll_legacy_body_state(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    next_velocity_in_current_frame = state[3:6] + delta[3:6]
    next_velocity_body_frame = _inertial_to_body_single(
        pose_delta=delta[:3],
        vector=next_velocity_in_current_frame,
    )
    return torch.cat((torch.zeros_like(delta[:3]), next_velocity_body_frame))


def _inertial_to_body_single(
    *,
    pose_delta: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    yaw = pose_delta[2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    return torch.stack(
        (
            cos_yaw * vector[0] + sin_yaw * vector[1],
            -sin_yaw * vector[0] + cos_yaw * vector[1],
            vector[2],
        )
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.median())


def _quantile_list(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, q))


def _selected_prediction_names(
    predictions: dict[str, torch.Tensor],
    *,
    include_offline: bool = False,
) -> list[str]:
    order = [
        "fe_rls",
        "fe_prior_rls",
        "fe_prior_static",
        "fe_kalman",
        "fe_window_ls",
        "neuralfly_rls",
        "neuralfly_prior_rls",
        "neuralfly_prior_static",
        "maml_static",
        "maml_online",
        "linear_rls",
        "fe_sgd",
        "static_node",
    ]
    if include_offline:
        order += ["offline_fe", "offline_neuralfly", "offline_linear"]
    return [name for name in order if name in predictions]


def _plot_kwargs(name: str) -> dict[str, object]:
    if name == "zero_delta":
        return {"linewidth": 1.0, "alpha": 0.45, "linestyle": "--"}
    if name.startswith("offline"):
        return {"linewidth": 1.0, "alpha": 0.7, "linestyle": ":"}
    if name == "static_node":
        return {"linewidth": 1.1, "alpha": 0.8, "linestyle": "-."}
    if name == "maml_online":
        return {"linewidth": 1.1, "alpha": 0.85, "linestyle": "--"}
    if name == "maml_static":
        return {"linewidth": 1.0, "alpha": 0.75, "linestyle": ":"}
    if "prior" in name:
        return {"linewidth": 1.2, "alpha": 0.9, "linestyle": "-."}
    if name in {"fe_sgd", "fe_window_ls"}:
        return {"linewidth": 1.0, "alpha": 0.85}
    return {"linewidth": 1.2}


def _path_length(deltas: torch.Tensor) -> float:
    poses = integrate_planar_deltas(deltas)
    return float(torch.linalg.norm(poses[1:, :2] - poses[:-1, :2], dim=-1).sum())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator
