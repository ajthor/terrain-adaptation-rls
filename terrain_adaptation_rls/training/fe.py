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
        loss = function_encoder_loss(model, batch)
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
