"""NeuralFly-style learned-basis training on Phoenix-shaped scene data."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from terrain_adaptation_rls.configuration import ExperimentConfig
from terrain_adaptation_rls.data.load_data import PhoenixDataset, load_scenes
from terrain_adaptation_rls.estimators.linear import linear_predict, solve_ridge_coefficients
from terrain_adaptation_rls.evaluation.diagnostic_plots import (
    write_delta_scale_plot,
    write_error_histogram,
    write_trajectory_snapshot,
    write_trajectory_summary,
)
from terrain_adaptation_rls.methods.runtime import NeuralFlyStyleBasisProvider, RuntimeInput
from terrain_adaptation_rls.training.supervised import (
    scene_supervised_batch,
    scene_trajectory_batch,
    write_training_plots,
)


def run_neuralfly_style_training(
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
    max_steps: int | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train a NeuralFly-style basis with batch-computed coefficients."""

    if config.platform is None:
        raise ValueError("NeuralFly-style training requires config.platform")

    train_scenes = [str(scene) for scene in config.data.get("train_scenes", ())]
    validation_scenes = [str(scene) for scene in config.data.get("validation_scenes", ())]
    if not train_scenes:
        raise ValueError("NeuralFly-style training requires data.train_scenes")

    from torch.utils.data import DataLoader

    torch.manual_seed(config.seed)
    device = torch.device(device)

    n_basis = int(config.model.get("n_basis", 8))
    hidden_size = int(config.model.get("hidden_size", 128))
    n_hidden_layers = int(config.model.get("n_hidden_layers", 2))
    ridge = float(config.model.get("ridge", 1e-6))
    model = NeuralFlyStyleBasisProvider(
        input_dim=9,
        output_dim=6,
        n_basis=n_basis,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers,
    ).to(device)

    training = config.training
    learning_rate = float(training.get("learning_rate", 1e-3))
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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses: list[float] = []
    validation_losses: list[dict[str, float | int | str]] = []
    latest_validation_losses: dict[str, float] = {}

    for step in range(1, steps + 1):
        batch = tuple(tensor.to(device) for tensor in next(train_iter))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = neuralfly_style_loss(model, batch, ridge=ridge)
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        train_loss = float(loss.detach().cpu())
        train_losses.append(train_loss)

        if validation_batches and (step == steps or step % eval_interval == 0):
            model.eval()
            with torch.no_grad():
                for scene, validation_batch in validation_batches.items():
                    validation_loss = float(
                        neuralfly_style_loss(model, validation_batch, ridge=ridge)
                        .detach()
                        .cpu()
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
            if latest_validation_losses:
                validation_text = ", ".join(
                    f"{scene}={loss_value:.6g}"
                    for scene, loss_value in sorted(latest_validation_losses.items())
                )
                message += f" validation_loss[{validation_text}]"
            print(message, flush=True)

    metrics: dict[str, object] = {
        "family": "neuralfly_style",
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
        "ridge": ridge,
        "learning_rate": learning_rate,
        "eval_interval": eval_interval,
        "log_interval": log_interval,
        "gradient_clip_norm": gradient_clip_norm,
        "trajectory_query_start_index": trajectory_query_start_index,
        "trajectory_example_policy": trajectory_example_policy,
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_validation_loss": validation_losses[-1]["loss"] if validation_losses else None,
    }

    if artifact_dir is not None:
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), artifact_path / "neuralfly_style_basis.pth")
        write_training_plots(artifact_path, metrics)
        if validation_artifact_batches:
            scene, validation_batch = next(iter(validation_artifact_batches.items()))
            write_neuralfly_prediction_artifacts(
                artifact_path,
                model,
                validation_batch,
                scene=scene,
                ridge=ridge,
            )

    return metrics


def neuralfly_style_loss(
    model: NeuralFlyStyleBasisProvider,
    batch: tuple[torch.Tensor, ...],
    *,
    ridge: float,
) -> torch.Tensor:
    """Compute learned-basis supervised loss using example-point coefficients."""

    prediction = predict_neuralfly_style_batch(model, batch, ridge=ridge)
    return torch.nn.functional.mse_loss(prediction, batch[2])


def predict_neuralfly_style_batch(
    model: NeuralFlyStyleBasisProvider,
    batch: tuple[torch.Tensor, ...],
    *,
    ridge: float,
) -> torch.Tensor:
    """Predict query deltas after solving low-dimensional coefficients."""

    xs, dt, _, example_xs, example_dt, example_ys = batch
    example_features = model(RuntimeInput(example_xs, example_dt))
    coefficients = solve_ridge_coefficients(example_features, example_ys, ridge=ridge)
    query_features = model(RuntimeInput(xs, dt))
    return linear_predict(query_features, coefficients)


def load_neuralfly_style_basis(
    config: ExperimentConfig,
    train_run_dir: str | Path,
    *,
    device: torch.device | str,
) -> NeuralFlyStyleBasisProvider:
    """Load a trained NeuralFly-style basis from an artifact directory."""

    device = torch.device(device)
    n_basis = int(config.model.get("n_basis", 8))
    hidden_size = int(config.model.get("hidden_size", 128))
    n_hidden_layers = int(config.model.get("n_hidden_layers", 2))
    model = NeuralFlyStyleBasisProvider(
        input_dim=9,
        output_dim=6,
        n_basis=n_basis,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers,
    ).to(device)
    state = torch.load(Path(train_run_dir) / "neuralfly_style_basis.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def write_neuralfly_prediction_artifacts(
    artifact_dir: Path,
    model: NeuralFlyStyleBasisProvider,
    batch: tuple[torch.Tensor, ...],
    *,
    scene: str,
    ridge: float,
) -> None:
    """Write validation predictions and plots with stable artifact names."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    xs, dt, target, *_ = batch
    prediction = predict_neuralfly_style_batch(model, batch, ridge=ridge)
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
