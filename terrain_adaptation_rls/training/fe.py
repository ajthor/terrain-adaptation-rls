"""Direct Function Encoder training on Phoenix-shaped scene data."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from terrain_adaptation_rls.configuration import ExperimentConfig
from terrain_adaptation_rls.data.load_data import PhoenixDataset, load_scenes
from terrain_adaptation_rls.models.function_encoder import create_model
from terrain_adaptation_rls.training.optimization import (
    build_optimizer,
    configure_step_learning_rate,
)
from terrain_adaptation_rls.training.supervised import (
    scene_supervised_batch,
    scene_trajectory_batch,
    write_training_plots,
)
from terrain_adaptation_rls.training.weak_form import (
    solve_weak_coefficients,
    weak_system_from_basis,
)


def run_function_encoder_training(
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
    max_steps: int | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train a Function Encoder with batch-computed coefficients."""

    if config.platform is None:
        raise ValueError("Function Encoder training requires config.platform")

    train_scenes = [str(scene) for scene in config.data.get("train_scenes", ())]
    validation_scenes = [str(scene) for scene in config.data.get("validation_scenes", ())]
    if not train_scenes:
        raise ValueError("Function Encoder training requires data.train_scenes")

    from torch.utils.data import DataLoader

    torch.manual_seed(config.seed)
    device = torch.device(device)

    n_basis = int(config.model.get("n_basis", 8))
    hidden_size = int(config.model.get("hidden_size", 128))
    model = create_model(device, n_basis=n_basis, hidden_size=hidden_size)

    training = config.training
    learning_rate = float(training.get("learning_rate", 1e-3))
    optimizer_name = str(training.get("optimizer", "adam"))
    weight_decay = float(training.get("weight_decay", 0.0))
    lr_schedule = str(training.get("lr_schedule", training.get("scheduler", "none")))
    warmup_steps = int(training.get("warmup_steps", 0))
    final_lr_fraction = float(training.get("final_lr_fraction", 0.1))
    configured_steps = int(training.get("steps", 1))
    steps = configured_steps if max_steps is None else min(configured_steps, max_steps)
    batch_size = int(training.get("batch_size", 2))
    n_points = int(training.get("n_points", 128))
    n_example_points = int(training.get("n_example_points", 32))
    eval_interval = int(training.get("eval_interval", max(1, steps)))
    log_interval = int(training.get("log_interval", 0))
    gradient_clip_norm = float(training.get("gradient_clip_norm", 1.0))
    loss_config = dict(training.get("loss", config.extras.get("loss", {})) or {})
    loss_type = str(loss_config.get("type", "pointwise"))
    weak_enabled = loss_type in {"weak", "weak_form", "hybrid_weak", "hybrid"}
    weak_weight = float(loss_config.get("weak_weight", 1.0))
    weak_n_tests = int(loss_config.get("weak_test_functions", 8))
    weak_ridge = float(loss_config.get("weak_ridge", loss_config.get("ridge", 1e-4)))
    weak_context_points = int(loss_config.get("weak_context_points", n_example_points))
    weak_query_points = int(loss_config.get("weak_query_points", n_points))
    weak_start_step = int(loss_config.get("weak_start_step", 1))
    weak_ramp_steps = int(loss_config.get("weak_ramp_steps", 0))
    weak_only = loss_type in {"weak", "weak_form"}
    if weak_enabled:
        if weak_context_points <= 1 or weak_query_points <= 1:
            raise ValueError("weak context/query windows must contain at least two points")
        if weak_n_tests <= 0:
            raise ValueError("weak_test_functions must be positive")
        if weak_start_step <= 0:
            raise ValueError("weak_start_step must be positive")
        if weak_ramp_steps < 0:
            raise ValueError("weak_ramp_steps must be non-negative")
    ode_regularization = dict(
        training.get(
            "ode_regularization",
            config.extras.get("ode_regularization", {}),
        )
        or {}
    )
    kinetic_weight = float(ode_regularization.get("kinetic_weight", 0.0))
    jacobian_weight = float(ode_regularization.get("jacobian_weight", 0.0))
    ode_reg_start_step = int(ode_regularization.get("start_step", 1))
    ode_reg_ramp_steps = int(ode_regularization.get("ramp_steps", 0))
    ode_reg_max_points = int(ode_regularization.get("max_points", 128))
    if kinetic_weight < 0.0 or jacobian_weight < 0.0:
        raise ValueError("ODE regularization weights must be non-negative")
    if ode_reg_start_step <= 0:
        raise ValueError("ODE regularization start_step must be positive")
    if ode_reg_ramp_steps < 0:
        raise ValueError("ODE regularization ramp_steps must be non-negative")
    if ode_reg_max_points <= 0:
        raise ValueError("ODE regularization max_points must be positive")

    evaluation = config.evaluation
    max_eval_points = int(evaluation.get("max_eval_points", 512))
    trajectory_query_start_index = int(
        evaluation.get("trajectory_query_start_index", n_example_points)
    )
    trajectory_example_policy = str(evaluation.get("trajectory_example_policy", "random_scene"))

    train_data = load_scenes(train_scenes, config.platform)
    train_inputs = [train_data[scene][0] for scene in train_scenes]
    train_targets = [train_data[scene][1] for scene in train_scenes]
    train_dataset = PhoenixDataset(
        train_inputs,
        train_targets,
        n_example_points=n_example_points,
        n_points=n_points,
    )
    train_iter = iter(DataLoader(train_dataset, batch_size=batch_size))

    validation_data = load_scenes(validation_scenes, config.platform) if validation_scenes else {}
    validation_batches = {
        scene: scene_supervised_batch(
            inputs=inputs,
            targets=targets,
            n_example_points=n_example_points,
            max_query_points=max_eval_points,
            device=device,
            seed=config.seed,
        )
        for scene, (inputs, targets) in validation_data.items()
    }
    validation_artifact_batches = {
        scene: scene_trajectory_batch(
            inputs=inputs,
            targets=targets,
            n_example_points=n_example_points,
            max_query_points=max_eval_points,
            device=device,
            query_start_index=trajectory_query_start_index,
            example_policy=trajectory_example_policy,
            seed=config.seed,
        )
        for scene, (inputs, targets) in validation_data.items()
    }

    optimizer = build_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    train_losses: list[float] = []
    pointwise_train_losses: list[float] = []
    weak_train_losses: list[float] = []
    ode_kinetic_losses: list[float] = []
    ode_jacobian_losses: list[float] = []
    learning_rates: list[float] = []
    validation_losses: list[dict[str, float | int | str]] = []
    latest_validation_losses: dict[str, float] = {}

    for step in range(1, steps + 1):
        batch = tuple(tensor.to(device) for tensor in next(train_iter))
        model.train()
        current_lr = configure_step_learning_rate(
            optimizer,
            step=step,
            total_steps=steps,
            base_learning_rate=learning_rate,
            schedule=lr_schedule,
            warmup_steps=warmup_steps,
            final_lr_fraction=final_lr_fraction,
        )
        optimizer.zero_grad(set_to_none=True)
        pointwise_loss = function_encoder_loss(model, batch)
        weak_loss = None
        current_weak_weight = 0.0
        if weak_enabled and step >= weak_start_step:
            current_weak_weight = _scheduled_weak_weight(
                weak_weight,
                step=step,
                start_step=weak_start_step,
                ramp_steps=weak_ramp_steps,
            )
            weak_batch = sample_weak_trajectory_batch(
                train_inputs,
                train_targets,
                batch_size=batch_size,
                context_points=weak_context_points,
                query_points=weak_query_points,
                device=device,
            )
            weak_loss = weak_function_encoder_loss(
                model,
                weak_batch,
                n_tests=weak_n_tests,
                ridge=weak_ridge,
            )
            loss = weak_loss if weak_only else pointwise_loss + current_weak_weight * weak_loss
        else:
            loss = pointwise_loss
        ode_kinetic_loss = None
        ode_jacobian_loss = None
        current_ode_reg_scale = 0.0
        if (
            step >= ode_reg_start_step
            and (kinetic_weight > 0.0 or jacobian_weight > 0.0)
        ):
            current_ode_reg_scale = _scheduled_weak_weight(
                1.0,
                step=step,
                start_step=ode_reg_start_step,
                ramp_steps=ode_reg_ramp_steps,
            )
            ode_kinetic_loss, ode_jacobian_loss = ode_vector_field_regularization(
                model,
                xs=batch[0],
                max_points=ode_reg_max_points,
                include_jacobian=jacobian_weight > 0.0,
            )
            loss = loss + current_ode_reg_scale * (
                kinetic_weight * ode_kinetic_loss
                + jacobian_weight * ode_jacobian_loss
            )
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        train_loss = float(loss.detach().cpu())
        train_losses.append(train_loss)
        pointwise_train_losses.append(float(pointwise_loss.detach().cpu()))
        if weak_loss is not None:
            weak_train_losses.append(float(weak_loss.detach().cpu()))
        if ode_kinetic_loss is not None:
            ode_kinetic_losses.append(float(ode_kinetic_loss.detach().cpu()))
        if ode_jacobian_loss is not None:
            ode_jacobian_losses.append(float(ode_jacobian_loss.detach().cpu()))
        learning_rates.append(current_lr)

        if validation_batches and (step == steps or step % eval_interval == 0):
            model.eval()
            with torch.no_grad():
                for scene, validation_batch in validation_batches.items():
                    validation_loss = float(
                        function_encoder_loss(model, validation_batch).detach().cpu()
                    )
                    latest_validation_losses[scene] = validation_loss
                    validation_losses.append(
                        {
                            "step": step,
                            "scene": scene,
                            "loss": validation_loss,
                        }
                    )

        if log_interval > 0 and (step == 1 or step == steps or step % log_interval == 0):
            message = f"step {step}/{steps} train_loss={train_loss:.6g}"
            if weak_loss is not None:
                message += (
                    f" pointwise_loss={pointwise_train_losses[-1]:.6g}"
                    f" weak_loss={weak_train_losses[-1]:.6g}"
                    f" weak_weight={current_weak_weight:.6g}"
                )
            if ode_kinetic_loss is not None:
                message += (
                    f" kinetic_loss={ode_kinetic_losses[-1]:.6g}"
                    f" jacobian_loss={ode_jacobian_losses[-1]:.6g}"
                    f" ode_reg_scale={current_ode_reg_scale:.6g}"
                )
            if latest_validation_losses:
                validation_text = ", ".join(
                    f"{scene}={loss_value:.6g}"
                    for scene, loss_value in sorted(latest_validation_losses.items())
                )
                message += f" validation_loss[{validation_text}]"
            print(message, flush=True)

    metrics: dict[str, object] = {
        "family": "function_encoder",
        "device": str(device),
        "platform": config.platform,
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "steps": steps,
        "batch_size": batch_size,
        "n_points": n_points,
        "n_example_points": n_example_points,
        "n_basis": n_basis,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "lr_schedule": lr_schedule,
        "warmup_steps": warmup_steps,
        "final_lr_fraction": final_lr_fraction,
        "eval_interval": eval_interval,
        "log_interval": log_interval,
        "gradient_clip_norm": gradient_clip_norm,
        "loss_type": loss_type,
        "loss_config": loss_config,
        "trajectory_query_start_index": trajectory_query_start_index,
        "trajectory_example_policy": trajectory_example_policy,
        "train_losses": train_losses,
        "pointwise_train_losses": pointwise_train_losses,
        "weak_train_losses": weak_train_losses,
        "ode_regularization": ode_regularization,
        "ode_kinetic_losses": ode_kinetic_losses,
        "ode_jacobian_losses": ode_jacobian_losses,
        "learning_rates": learning_rates,
        "validation_losses": validation_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_validation_loss": validation_losses[-1]["loss"] if validation_losses else None,
    }

    if artifact_dir is not None:
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), artifact_path / "function_encoder_model.pth")
        write_training_plots(artifact_path, metrics)
        if validation_artifact_batches:
            scene, validation_batch = next(iter(validation_artifact_batches.items()))
            write_function_encoder_prediction_artifacts(
                artifact_path,
                model,
                validation_batch,
                scene=scene,
            )

    return metrics


def function_encoder_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Compute FE supervised loss using coefficients from example points."""

    prediction = predict_function_encoder_batch(model, batch)
    target = batch[2]
    return torch.nn.functional.mse_loss(prediction, target)


def weak_function_encoder_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    *,
    n_tests: int = 8,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """Compute weak-form FE loss from context/query trajectory windows."""

    context_xs, context_dt, query_xs, query_dt = batch
    context_basis = evaluate_raw_basis_functions(model, context_xs)
    query_basis = evaluate_raw_basis_functions(model, query_xs)

    context_weak_basis, context_weak_target = weak_system_from_basis(
        context_xs[..., :6],
        context_basis,
        context_dt,
        n_tests=n_tests,
    )
    coefficients = solve_weak_coefficients(
        context_weak_basis,
        context_weak_target,
        ridge=ridge,
    )

    query_weak_basis, query_weak_target = weak_system_from_basis(
        query_xs[..., :6],
        query_basis,
        query_dt,
        n_tests=n_tests,
    )
    weak_prediction = torch.einsum("bmdk,bk->bmd", query_weak_basis, coefficients)
    return torch.nn.functional.mse_loss(weak_prediction, query_weak_target)


def ode_vector_field_regularization(
    model: torch.nn.Module,
    *,
    xs: torch.Tensor,
    max_points: int = 128,
    include_jacobian: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regularize FE-ODE basis vector fields before RK4 integration.

    This follows the spirit of neural ODE kinetic/Jacobian regularization: keep
    vector fields small and locally smooth so the integrated dynamics are easier
    to solve and less prone to irregular trajectories.
    """

    regularization_xs = _subsample_regularization_points(xs, max_points=max_points)
    regularization_xs = regularization_xs.detach().clone().requires_grad_(include_jacobian)
    raw_basis = evaluate_raw_basis_functions(model, regularization_xs)
    kinetic_loss = raw_basis.square().mean()
    if not include_jacobian:
        return kinetic_loss, raw_basis.new_zeros(())

    probe = torch.randn_like(raw_basis)
    projection = (raw_basis * probe).sum()
    (gradient,) = torch.autograd.grad(
        projection,
        regularization_xs,
        create_graph=True,
        retain_graph=True,
    )
    jacobian_loss = gradient.square().mean()
    return kinetic_loss, jacobian_loss


def _subsample_regularization_points(xs: torch.Tensor, *, max_points: int) -> torch.Tensor:
    flat_xs = xs.reshape(-1, xs.shape[-1])
    if flat_xs.shape[0] <= max_points:
        return flat_xs
    indices = torch.randperm(flat_xs.shape[0], device=flat_xs.device)[:max_points]
    return flat_xs[indices]


def evaluate_raw_basis_functions(
    model: torch.nn.Module,
    xs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate FE basis vector fields before RK4 integration.

    The public FE basis returns integrated delta features because each basis is
    a ``NeuralODE``. Weak-form training needs the continuous vector fields
    ``g_i(x)`` instead.
    """

    try:
        basis_modules = model.basis_functions.basis_functions
    except AttributeError as exc:
        raise TypeError("model does not expose FunctionEncoder basis modules") from exc

    time = torch.zeros(xs.shape[:-1], dtype=xs.dtype, device=xs.device)
    raw_basis = [basis.ode_func(time, xs) for basis in basis_modules]
    return torch.stack(raw_basis, dim=-1)


def sample_weak_trajectory_batch(
    inputs: list[torch.Tensor],
    targets: list[torch.Tensor],
    *,
    batch_size: int,
    context_points: int,
    query_points: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample contiguous context/query windows for weak-form FE training."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have matching scene counts")

    context_xs: list[torch.Tensor] = []
    context_dt: list[torch.Tensor] = []
    query_xs: list[torch.Tensor] = []
    query_dt: list[torch.Tensor] = []
    for _ in range(batch_size):
        scene_index = torch.randint(0, len(inputs), (1,)).item()
        scene_inputs = inputs[scene_index]
        scene_targets = targets[scene_index]
        context_start = _sample_window_start(scene_inputs.shape[0], context_points)
        query_start = _sample_window_start(scene_inputs.shape[0], query_points)

        c_inputs = scene_inputs[context_start : context_start + context_points]
        c_targets = scene_targets[context_start : context_start + context_points]
        q_inputs = scene_inputs[query_start : query_start + query_points]
        q_targets = scene_targets[query_start : query_start + query_points]

        context_xs.append(c_inputs[:, 1:])
        context_dt.append(c_targets[:, 0] - c_inputs[:, 0])
        query_xs.append(q_inputs[:, 1:])
        query_dt.append(q_targets[:, 0] - q_inputs[:, 0])

    device = torch.device(device)
    return (
        torch.stack(context_xs).to(device),
        torch.stack(context_dt).to(device),
        torch.stack(query_xs).to(device),
        torch.stack(query_dt).to(device),
    )


def _sample_window_start(n_rows: int, window: int) -> int:
    if n_rows < window:
        raise ValueError(f"scene has {n_rows} rows, fewer than weak window {window}")
    return torch.randint(0, n_rows - window + 1, (1,)).item()


def _scheduled_weak_weight(
    weak_weight: float,
    *,
    step: int,
    start_step: int,
    ramp_steps: int,
) -> float:
    if ramp_steps <= 0:
        return weak_weight
    ramp_position = min(max(step - start_step + 1, 0), ramp_steps)
    return weak_weight * ramp_position / ramp_steps


def predict_function_encoder_batch(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Predict query deltas after solving coefficients from example points."""

    xs, dt, _, example_xs, example_dt, example_ys = batch
    coefficients, _ = model.compute_coefficients((example_xs, example_dt), example_ys)
    return model((xs, dt), coefficients=coefficients)


@torch.no_grad()
def write_function_encoder_prediction_artifacts(
    artifact_dir: Path,
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    *,
    scene: str,
) -> None:
    """Write FE validation predictions and plots with stable artifact names."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    xs, dt, target, *_ = batch
    prediction = predict_function_encoder_batch(model, batch)
    error = torch.linalg.norm(prediction - target, dim=-1)
    time = torch.cumsum(dt.squeeze(0), dim=0)

    csv_path = artifact_dir / "validation_predictions.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["scene", "index", "time", "error"]
        fieldnames += [f"target_{idx}" for idx in range(target.shape[-1])]
        fieldnames += [f"prediction_{idx}" for idx in range(prediction.shape[-1])]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        target_cpu = target.squeeze(0).detach().cpu()
        prediction_cpu = prediction.squeeze(0).detach().cpu()
        error_cpu = error.squeeze(0).detach().cpu()
        time_cpu = time.detach().cpu()
        for idx in range(target_cpu.shape[0]):
            row = {
                "scene": scene,
                "index": idx,
                "time": float(time_cpu[idx]),
                "error": float(error_cpu[idx]),
            }
            row.update(
                {
                    f"target_{dim}": float(target_cpu[idx, dim])
                    for dim in range(target_cpu.shape[-1])
                }
            )
            row.update(
                {
                    f"prediction_{dim}": float(prediction_cpu[idx, dim])
                    for dim in range(prediction_cpu.shape[-1])
                }
            )
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time.detach().cpu(), error.squeeze(0).detach().cpu())
    ax.set_xlabel("relative time [s]")
    ax.set_ylabel("prediction error norm")
    ax.set_title(scene)
    fig.tight_layout()
    fig.savefig(artifact_dir / "validation_error.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(9, 7), sharex=True)
    axes_flat = axes.ravel()
    target_cpu = target.squeeze(0).detach().cpu()
    prediction_cpu = prediction.squeeze(0).detach().cpu()
    time_cpu = time.detach().cpu()
    for dim, ax in enumerate(axes_flat):
        ax.plot(time_cpu, target_cpu[:, dim], label="target", linewidth=1.2)
        ax.plot(time_cpu, prediction_cpu[:, dim], label="prediction", linewidth=1.0)
        ax.set_ylabel(f"dim {dim}")
    axes_flat[0].legend()
    axes_flat[-1].set_xlabel("relative time [s]")
    axes_flat[-2].set_xlabel("relative time [s]")
    fig.tight_layout()
    fig.savefig(artifact_dir / "validation_components.png", dpi=160)
    plt.close(fig)

    from terrain_adaptation_rls.evaluation.diagnostic_plots import write_fe_diagnostics

    write_fe_diagnostics(
        artifact_dir,
        model=model,
        batch=batch,
        prediction=prediction,
        scene=scene,
    )
