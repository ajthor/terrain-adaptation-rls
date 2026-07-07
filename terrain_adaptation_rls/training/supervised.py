"""Small supervised training utilities for FE/NODE smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Callable

import torch

from terrain_adaptation_rls.configuration import ExperimentConfig


LossFn = Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.device], torch.Tensor]


@dataclass(frozen=True)
class BuiltModel:
    """A model and loss function resolved from an experiment config."""

    model: torch.nn.Module
    loss_fn: LossFn
    family: str


def build_model_from_config(
    config: ExperimentConfig,
    *,
    device: torch.device | str,
) -> BuiltModel:
    """Build a trainable model from the config envelope."""

    family = str(config.model.get("family", "neural_ode"))
    n_basis = int(config.model.get("n_basis", 8))
    hidden_size = int(config.model.get("hidden_size", 128))

    if family in {"neural_ode", "node"}:
        from terrain_adaptation_rls.models.neural_ode import create_model, loss_fn

        return BuiltModel(
            model=create_model(device, n_basis=n_basis, hidden_size=hidden_size),
            loss_fn=loss_fn,
            family="neural_ode",
        )

    if family in {"function_encoder", "fe"}:
        from terrain_adaptation_rls.models.function_encoder import create_model, loss_fn

        return BuiltModel(
            model=create_model(device, n_basis=n_basis, hidden_size=hidden_size),
            loss_fn=loss_fn,
            family="function_encoder",
        )

    if family in {"maml", "maml_neural_ode"}:
        raise NotImplementedError(
            "MAML meta-training still needs to be ported from the missing legacy "
            "meta_learning dependency."
        )

    raise ValueError(f"Unknown model family '{family}'")


def run_synthetic_supervised_training(
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
    max_steps: int | None = None,
) -> dict[str, object]:
    """Run a tiny fixed-batch training loop for constructor/loss smoke tests."""

    torch.manual_seed(config.seed)
    device = torch.device(device)
    built = build_model_from_config(config, device=device)
    learning_rate = float(config.training.get("learning_rate", 1e-3))
    configured_steps = int(config.training.get("steps", 1))
    steps = configured_steps if max_steps is None else min(configured_steps, max_steps)
    batch_size = int(config.training.get("batch_size", 2))
    n_points = int(config.training.get("n_points", 8))
    n_example_points = int(config.training.get("n_example_points", 8))

    batch = synthetic_batch(
        batch_size=batch_size,
        n_points=n_points,
        n_example_points=n_example_points,
        device=device,
    )
    optimizer = torch.optim.Adam(built.model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for _ in range(steps):
        built.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = built.loss_fn(built.model, batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(built.model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return {
        "family": built.family,
        "device": str(device),
        "steps": steps,
        "batch_size": batch_size,
        "n_points": n_points,
        "n_example_points": n_example_points,
        "losses": losses,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
    }


def run_configured_supervised_training(
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
    max_steps: int | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train a configured supervised model on real scene data.

    Function Encoder training now has an explicit implementation in
    ``training.fe``. This compatibility entrypoint delegates FE configs there
    and keeps the older generic path for NODE/debug use.
    """

    family = str(config.model.get("family", "neural_ode"))
    if family in {"function_encoder", "fe"}:
        from terrain_adaptation_rls.training.fe import run_function_encoder_training

        return run_function_encoder_training(
            config,
            device=device,
            max_steps=max_steps,
            artifact_dir=artifact_dir,
        )
    if family in {"neuralfly_style", "neuralfly"}:
        from terrain_adaptation_rls.training.neuralfly import run_neuralfly_style_training

        return run_neuralfly_style_training(
            config,
            device=device,
            max_steps=max_steps,
            artifact_dir=artifact_dir,
        )

    if config.platform is None:
        raise ValueError("Real-data training requires config.platform")

    train_scenes = [str(scene) for scene in config.data.get("train_scenes", ())]
    validation_scenes = [str(scene) for scene in config.data.get("validation_scenes", ())]
    if not train_scenes:
        raise ValueError("Real-data training requires data.train_scenes")

    from torch.utils.data import DataLoader

    from terrain_adaptation_rls.data.load_data import PhoenixDataset, load_scenes

    torch.manual_seed(config.seed)
    device = torch.device(device)
    built = build_model_from_config(config, device=device)

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
    max_eval_points = int(config.evaluation.get("max_eval_points", 512))
    trajectory_query_start_index = int(
        config.evaluation.get("trajectory_query_start_index", n_example_points)
    )
    trajectory_example_policy = str(
        config.evaluation.get("trajectory_example_policy", "random_scene")
    )

    train_data = load_scenes(train_scenes, config.platform)
    train_inputs = [train_data[scene][0] for scene in train_scenes]
    train_targets = [train_data[scene][1] for scene in train_scenes]
    train_dataset = PhoenixDataset(
        train_inputs,
        train_targets,
        n_example_points=n_example_points,
        n_points=n_points,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    train_iter = iter(train_loader)

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

    optimizer = torch.optim.Adam(built.model.parameters(), lr=learning_rate)
    train_losses: list[float] = []
    validation_losses: list[dict[str, float | int | str]] = []
    latest_validation_losses: dict[str, float] = {}

    for step in range(1, steps + 1):
        batch = tuple(tensor.to(device) for tensor in next(train_iter))
        built.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = built.loss_fn(built.model, batch, device)
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(built.model.parameters(), gradient_clip_norm)
        optimizer.step()
        train_loss = float(loss.detach().cpu())
        train_losses.append(train_loss)

        if validation_batches and (step == steps or step % eval_interval == 0):
            built.model.eval()
            with torch.no_grad():
                for scene, validation_batch in validation_batches.items():
                    validation_loss = built.loss_fn(built.model, validation_batch, device)
                    validation_loss_value = float(validation_loss.detach().cpu())
                    latest_validation_losses[scene] = validation_loss_value
                    validation_losses.append(
                        {
                            "step": step,
                            "scene": scene,
                            "loss": validation_loss_value,
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
        "family": built.family,
        "device": str(device),
        "platform": config.platform,
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "steps": steps,
        "batch_size": batch_size,
        "n_points": n_points,
        "n_example_points": n_example_points,
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
        torch.save(built.model.state_dict(), artifact_path / f"{built.family}_model.pth")
        write_training_plots(artifact_path, metrics)
        if validation_artifact_batches:
            scene, validation_batch = next(iter(validation_artifact_batches.items()))
            write_prediction_artifacts(
                artifact_path,
                built.model,
                built.family,
                validation_batch,
                scene=scene,
            )

    return metrics


def synthetic_batch(
    *,
    batch_size: int,
    n_points: int,
    n_example_points: int,
    device: torch.device | str,
    input_dim: int = 8,
    output_dim: int = 6,
) -> tuple[torch.Tensor, ...]:
    """Create one fixed Phoenix-shaped supervised batch."""

    generator = torch.Generator(device="cpu").manual_seed(17)
    xs = torch.randn(batch_size, n_points, input_dim, generator=generator, device="cpu")
    dt = 0.05 + 0.1 * torch.rand(batch_size, n_points, generator=generator, device="cpu")
    ys = torch.randn(batch_size, n_points, output_dim, generator=generator, device="cpu")
    example_xs = torch.randn(batch_size, n_example_points, input_dim, generator=generator, device="cpu")
    example_dt = 0.05 + 0.1 * torch.rand(
        batch_size,
        n_example_points,
        generator=generator,
        device="cpu",
    )
    example_ys = torch.randn(
        batch_size,
        n_example_points,
        output_dim,
        generator=generator,
        device="cpu",
    )

    device = torch.device(device)
    return tuple(
        tensor.to(device)
        for tensor in (xs, dt, ys, example_xs, example_dt, example_ys)
    )


def scene_supervised_batch(
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_example_points: int,
    max_query_points: int,
    device: torch.device | str,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Build one deterministic supervised batch from a processed scene."""

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same row count")
    if inputs.shape[0] <= n_example_points:
        raise ValueError("scene does not contain enough points for examples and queries")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(inputs.shape[0], generator=generator)
    example_indices = indices[:n_example_points]
    query_indices = indices[n_example_points : n_example_points + max_query_points]
    if query_indices.numel() == 0:
        raise ValueError("scene does not contain query points after examples")

    return scene_batch_from_indices(
        inputs=inputs,
        targets=targets,
        example_indices=example_indices,
        query_indices=query_indices,
        device=device,
    )


def scene_sequence_batch(
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_example_points: int,
    max_query_points: int,
    device: torch.device | str,
    start_index: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Build one contiguous supervised batch from a processed scene."""

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same row count")
    if n_example_points <= 0:
        raise ValueError("n_example_points must be positive")
    if max_query_points <= 0:
        raise ValueError("max_query_points must be positive")

    query_start = start_index + n_example_points
    query_stop = min(query_start + max_query_points, inputs.shape[0])
    if query_start >= inputs.shape[0] or query_stop <= query_start:
        raise ValueError("scene does not contain enough points for a contiguous query segment")

    example_indices = torch.arange(start_index, query_start)
    query_indices = torch.arange(query_start, query_stop)
    return scene_batch_from_indices(
        inputs=inputs,
        targets=targets,
        example_indices=example_indices,
        query_indices=query_indices,
        device=device,
    )


def scene_trajectory_batch(
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_example_points: int,
    max_query_points: int,
    device: torch.device | str,
    query_start_index: int = 0,
    example_policy: str = "random_scene",
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Build a contiguous query batch with explicit FE conditioning semantics."""

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same row count")
    if n_example_points <= 0:
        raise ValueError("n_example_points must be positive")
    if max_query_points <= 0:
        raise ValueError("max_query_points must be positive")
    if query_start_index < 0:
        raise ValueError("query_start_index must be non-negative")

    query_stop = min(query_start_index + max_query_points, inputs.shape[0])
    if query_start_index >= inputs.shape[0] or query_stop <= query_start_index:
        raise ValueError("scene does not contain enough points for a contiguous query segment")

    query_indices = torch.arange(query_start_index, query_stop)
    if example_policy == "random_scene":
        available = torch.ones(inputs.shape[0], dtype=torch.bool)
        available[query_indices] = False
        available_indices = torch.arange(inputs.shape[0])[available]
        if available_indices.numel() < n_example_points:
            raise ValueError("scene does not contain enough non-query points for random examples")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(available_indices.numel(), generator=generator)
        example_indices = available_indices[permutation[:n_example_points]]
    elif example_policy == "preceding":
        if query_start_index < n_example_points:
            raise ValueError("preceding examples require query_start_index >= n_example_points")
        example_indices = torch.arange(query_start_index - n_example_points, query_start_index)
    else:
        raise ValueError(
            "unknown trajectory example policy "
            f"'{example_policy}'; expected 'random_scene' or 'preceding'"
        )

    return scene_batch_from_indices(
        inputs=inputs,
        targets=targets,
        example_indices=example_indices,
        query_indices=query_indices,
        device=device,
    )


def scene_batch_from_indices(
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    example_indices: torch.Tensor,
    query_indices: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.Tensor, ...]:
    """Build a supervised batch from explicit example and query indices."""

    xs_all = inputs[:, 1:]
    dt_all = targets[:, 0] - inputs[:, 0]
    ys_all = targets[:, 1:] - xs_all[:, :6]

    batch = (
        xs_all[query_indices].unsqueeze(0),
        dt_all[query_indices].unsqueeze(0),
        ys_all[query_indices].unsqueeze(0),
        xs_all[example_indices].unsqueeze(0),
        dt_all[example_indices].unsqueeze(0),
        ys_all[example_indices].unsqueeze(0),
    )
    device = torch.device(device)
    return tuple(tensor.to(device) for tensor in batch)


def predict_supervised_batch(
    model: torch.nn.Module,
    family: str,
    batch: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Predict deltas for one supervised FE/NODE batch."""

    xs, dt, _, example_xs, example_dt, example_ys = batch
    if family == "function_encoder":
        coefficients, _ = model.compute_coefficients((example_xs, example_dt), example_ys)
        return model((xs, dt), coefficients=coefficients)
    return model((xs, dt))


def write_training_plots(artifact_dir: Path, metrics: dict[str, object]) -> None:
    """Write training and validation loss plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_losses = [float(loss) for loss in metrics["train_losses"]]
    validation_losses = list(metrics["validation_losses"])

    fig, ax = plt.subplots(figsize=(7, 4))
    if train_losses:
        ax.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    if validation_losses:
        ax.scatter(
            [int(item["step"]) for item in validation_losses],
            [float(item["loss"]) for item in validation_losses],
            label="validation",
            s=24,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "training_curve.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def write_prediction_artifacts(
    artifact_dir: Path,
    model: torch.nn.Module,
    family: str,
    batch: tuple[torch.Tensor, ...],
    *,
    scene: str,
) -> None:
    """Write validation prediction CSV and plots for one scene."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    xs, dt, target, *_ = batch
    prediction = predict_supervised_batch(model, family, batch)
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

    from terrain_adaptation_rls.evaluation.diagnostic_plots import write_supervised_diagnostics

    write_supervised_diagnostics(
        artifact_dir,
        model=model,
        family=family,
        batch=batch,
        prediction=prediction,
        scene=scene,
    )
