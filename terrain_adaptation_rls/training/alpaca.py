"""ALPaCA-style Bayesian last-layer training on Phoenix-shaped scene data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from terrain_adaptation_rls.configuration import ExperimentConfig
from terrain_adaptation_rls.data.load_data import PhoenixDataset, load_scenes
from terrain_adaptation_rls.evaluation.diagnostic_plots import (
    write_delta_scale_plot,
    write_error_histogram,
    write_trajectory_snapshot,
    write_trajectory_summary,
)
from terrain_adaptation_rls.methods.runtime import ALPaCABasisProvider, RuntimeInput
from terrain_adaptation_rls.training.optimization import (
    build_optimizer,
    configure_step_learning_rate,
)
from terrain_adaptation_rls.training.supervised import (
    scene_supervised_batch,
    scene_trajectory_batch,
    write_training_plots,
)


@dataclass(frozen=True)
class ALPaCAPosterior:
    """Posterior over shared low-dimensional coefficients."""

    mean: torch.Tensor
    covariance: torch.Tensor
    precision: torch.Tensor
    rhs: torch.Tensor


def run_alpaca_training(
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
    max_steps: int | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train an ALPaCA feature map and coefficient prior."""

    if config.platform is None:
        raise ValueError("ALPaCA training requires config.platform")

    train_scenes = [str(scene) for scene in config.data.get("train_scenes", ())]
    validation_scenes = [str(scene) for scene in config.data.get("validation_scenes", ())]
    if not train_scenes:
        raise ValueError("ALPaCA training requires data.train_scenes")

    from torch.utils.data import DataLoader

    torch.manual_seed(config.seed)
    device = torch.device(device)

    n_basis = int(config.model.get("n_basis", 8))
    hidden_size = int(config.model.get("hidden_size", 128))
    n_hidden_layers = int(config.model.get("n_hidden_layers", 2))
    initial_prior_variance = float(config.model.get("initial_prior_variance", 1.0))
    initial_noise_variance = float(config.model.get("initial_noise_variance", 1e-4))
    model = ALPaCABasisProvider(
        input_dim=9,
        output_dim=6,
        n_basis=n_basis,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers,
        initial_prior_variance=initial_prior_variance,
        initial_noise_variance=initial_noise_variance,
    ).to(device)

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
        loss = alpaca_loss(model, batch)
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        train_loss = float(loss.detach().cpu())
        train_losses.append(train_loss)
        learning_rates.append(current_lr)

        if validation_batches and (step == steps or step % eval_interval == 0):
            model.eval()
            with torch.no_grad():
                for scene, validation_batch in validation_batches.items():
                    validation_loss = float(alpaca_loss(model, validation_batch).detach().cpu())
                    latest_validation_losses[scene] = validation_loss
                    validation_losses.append(
                        {"step": step, "scene": scene, "loss": validation_loss}
                    )

        if log_interval > 0 and (step == 1 or step == steps or step % log_interval == 0):
            message = f"step {step}/{steps} train_loss={train_loss:.6g}"
            if latest_validation_losses:
                validation_text = ", ".join(
                    f"{scene}={loss_value:.6g}"
                    for scene, loss_value in sorted(latest_validation_losses.items())
                )
                message += f" validation_loss[{validation_text}]"
            print(message, flush=True)

    metrics: dict[str, object] = {
        "family": "alpaca",
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
        "n_hidden_layers": n_hidden_layers,
        "initial_prior_variance": initial_prior_variance,
        "initial_noise_variance": initial_noise_variance,
        "learned_prior_variance_mean": float(model.prior_variance().detach().mean().cpu()),
        "learned_noise_variance": float(model.noise_variance().detach().cpu()),
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "lr_schedule": lr_schedule,
        "warmup_steps": warmup_steps,
        "final_lr_fraction": final_lr_fraction,
        "eval_interval": eval_interval,
        "log_interval": log_interval,
        "gradient_clip_norm": gradient_clip_norm,
        "trajectory_query_start_index": trajectory_query_start_index,
        "trajectory_example_policy": trajectory_example_policy,
        "train_losses": train_losses,
        "learning_rates": learning_rates,
        "validation_losses": validation_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_validation_loss": validation_losses[-1]["loss"] if validation_losses else None,
    }

    if artifact_dir is not None:
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), artifact_path / "alpaca_model.pth")
        write_training_plots(artifact_path, metrics)
        if validation_artifact_batches:
            scene, validation_batch = next(iter(validation_artifact_batches.items()))
            write_alpaca_prediction_artifacts(
                artifact_path,
                model,
                validation_batch,
                scene=scene,
            )

    return metrics


def alpaca_loss(
    model: ALPaCABasisProvider,
    batch: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Episodic ALPaCA negative log predictive likelihood."""

    xs, dt, target, example_xs, example_dt, example_ys = batch
    example_features = model(RuntimeInput(example_xs, example_dt))
    posterior = alpaca_posterior_from_context(model, example_features, example_ys)
    query_features = model(RuntimeInput(xs, dt))
    query_design, query_target = flatten_coefficient_regression(query_features, target)
    prediction = torch.einsum("bnk,bk->bn", query_design, posterior.mean)
    predictive_variance = predictive_variance_from_posterior(
        model,
        query_design,
        posterior.covariance,
    )
    error = query_target - prediction
    return 0.5 * (torch.log(predictive_variance) + error.square() / predictive_variance).mean()


def predict_alpaca_batch(
    model: ALPaCABasisProvider,
    batch: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Predict query deltas from the ALPaCA posterior computed on examples."""

    xs, dt, _, example_xs, example_dt, example_ys = batch
    example_features = model(RuntimeInput(example_xs, example_dt))
    posterior = alpaca_posterior_from_context(model, example_features, example_ys)
    query_features = model(RuntimeInput(xs, dt))
    return predict_alpaca_features(query_features, posterior.mean)


def alpaca_posterior_from_context(
    model: ALPaCABasisProvider,
    features: torch.Tensor,
    target: torch.Tensor,
) -> ALPaCAPosterior:
    """Compute a batched diagonal-prior Bayesian linear posterior."""

    design, flat_target = flatten_coefficient_regression(features, target)
    batch_size = design.shape[0]
    prior_precision = model.prior_variance().reciprocal()
    noise_precision = model.noise_variance().reciprocal()
    prior_precision_matrix = torch.diag(prior_precision).unsqueeze(0).expand(
        batch_size,
        model.n_coeff,
        model.n_coeff,
    )
    precision = prior_precision_matrix + noise_precision * torch.einsum(
        "bnk,bnl->bkl",
        design,
        design,
    )
    rhs = prior_precision.unsqueeze(0) * model.prior_mean.unsqueeze(0)
    rhs = rhs + noise_precision * torch.einsum("bnk,bn->bk", design, flat_target)
    covariance = torch.linalg.inv(precision)
    mean = torch.linalg.solve(precision, rhs.unsqueeze(-1)).squeeze(-1)
    return ALPaCAPosterior(
        mean=mean,
        covariance=covariance,
        precision=precision,
        rhs=rhs,
    )


def alpaca_prior_posterior(model: ALPaCABasisProvider, *, batch_size: int = 1) -> ALPaCAPosterior:
    """Return the learned ALPaCA prior in posterior-state form."""

    prior_precision = model.prior_variance().reciprocal()
    precision = torch.diag(prior_precision).unsqueeze(0).expand(
        batch_size,
        model.n_coeff,
        model.n_coeff,
    ).clone()
    covariance = torch.diag(model.prior_variance()).unsqueeze(0).expand(
        batch_size,
        model.n_coeff,
        model.n_coeff,
    ).clone()
    mean = model.prior_mean.unsqueeze(0).expand(batch_size, model.n_coeff).clone()
    rhs = prior_precision.unsqueeze(0) * mean
    return ALPaCAPosterior(
        mean=mean,
        covariance=covariance,
        precision=precision,
        rhs=rhs,
    )


def alpaca_zero_posterior(
    model: ALPaCABasisProvider,
    *,
    batch_size: int = 1,
    initial_covariance: float = 100.0,
) -> ALPaCAPosterior:
    """Return a zero-mean posterior state for cold-start ALPaCA adaptation."""

    variance = torch.full(
        (model.n_coeff,),
        float(initial_covariance),
        device=model.prior_mean.device,
        dtype=model.prior_mean.dtype,
    )
    precision_diag = variance.reciprocal()
    precision = torch.diag(precision_diag).unsqueeze(0).expand(
        batch_size,
        model.n_coeff,
        model.n_coeff,
    ).clone()
    covariance = torch.diag(variance).unsqueeze(0).expand(
        batch_size,
        model.n_coeff,
        model.n_coeff,
    ).clone()
    mean = torch.zeros(
        batch_size,
        model.n_coeff,
        device=model.prior_mean.device,
        dtype=model.prior_mean.dtype,
    )
    rhs = torch.zeros_like(mean)
    return ALPaCAPosterior(
        mean=mean,
        covariance=covariance,
        precision=precision,
        rhs=rhs,
    )


def alpaca_update_posterior(
    model: ALPaCABasisProvider,
    posterior: ALPaCAPosterior,
    features: torch.Tensor,
    target: torch.Tensor,
) -> ALPaCAPosterior:
    """Apply a Bayesian linear-regression update to an existing posterior."""

    design, flat_target = flatten_coefficient_regression(features, target)
    noise_precision = model.noise_variance().reciprocal()
    precision = posterior.precision + noise_precision * torch.einsum(
        "bnk,bnl->bkl",
        design,
        design,
    )
    rhs = posterior.rhs + noise_precision * torch.einsum("bnk,bn->bk", design, flat_target)
    covariance = torch.linalg.inv(precision)
    mean = torch.linalg.solve(precision, rhs.unsqueeze(-1)).squeeze(-1)
    return ALPaCAPosterior(
        mean=mean,
        covariance=covariance,
        precision=precision,
        rhs=rhs,
    )


def predict_alpaca_features(features: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Predict structured deltas from ALPaCA features and coefficient mean."""

    if coefficients.ndim == 1:
        coefficients = coefficients.unsqueeze(0)
    view_shape = (coefficients.shape[0],) + (1,) * (features.ndim - 3) + (
        1,
        coefficients.shape[-1],
    )
    return torch.sum(features * coefficients.reshape(view_shape), dim=-1)


def flatten_coefficient_regression(
    features: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten time/output dimensions into ALPaCA regression rows."""

    if features.shape[:-1] != target.shape:
        raise ValueError(
            "features and target dimensions must match except coefficient axis: "
            f"{tuple(features.shape[:-1])} != {tuple(target.shape)}"
        )
    return features.reshape(features.shape[0], -1, features.shape[-1]), target.reshape(
        target.shape[0],
        -1,
    )


def predictive_variance_from_posterior(
    model: ALPaCABasisProvider,
    design: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    """Return ALPaCA predictive variance for flattened design rows."""

    epistemic = torch.einsum("bnk,bkl,bnl->bn", design, covariance, design)
    return (model.noise_variance() + epistemic).clamp_min(1e-10)


def load_alpaca_model(
    config: ExperimentConfig,
    train_run_dir: str | Path,
    *,
    device: torch.device | str,
) -> ALPaCABasisProvider:
    """Load a trained ALPaCA feature map from an artifact directory."""

    device = torch.device(device)
    n_basis = int(config.model.get("n_basis", 8))
    hidden_size = int(config.model.get("hidden_size", 128))
    n_hidden_layers = int(config.model.get("n_hidden_layers", 2))
    initial_prior_variance = float(config.model.get("initial_prior_variance", 1.0))
    initial_noise_variance = float(config.model.get("initial_noise_variance", 1e-4))
    model = ALPaCABasisProvider(
        input_dim=9,
        output_dim=6,
        n_basis=n_basis,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers,
        initial_prior_variance=initial_prior_variance,
        initial_noise_variance=initial_noise_variance,
    ).to(device)
    state = torch.load(Path(train_run_dir) / "alpaca_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def write_alpaca_prediction_artifacts(
    artifact_dir: Path,
    model: ALPaCABasisProvider,
    batch: tuple[torch.Tensor, ...],
    *,
    scene: str,
) -> None:
    """Write validation predictions and plots with familiar artifact names."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    xs, dt, target, *_ = batch
    prediction = predict_alpaca_batch(model, batch)
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

    write_trajectory_summary(
        artifact_dir / "trajectory_summary.json",
        target=target,
        prediction=prediction,
        dt=dt,
        scene=scene,
    )
    write_delta_scale_plot(
        artifact_dir / "validation_delta_scale.png",
        target=target,
        prediction=prediction,
        dt=dt,
        scene=scene,
    )
    write_error_histogram(artifact_dir / "validation_error_histogram.png", error)
    write_trajectory_snapshot(
        artifact_dir / "validation_trajectory_snapshot.png",
        target=target,
        prediction=prediction,
        scene=scene,
    )
