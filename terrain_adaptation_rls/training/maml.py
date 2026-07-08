"""Direct first-order MAML training for Phoenix-shaped scene data."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import torch

from terrain_adaptation_rls.configuration import ExperimentConfig
from terrain_adaptation_rls.data.load_data import PhoenixDataset, load_scenes
from terrain_adaptation_rls.models.maml import create_model, loss_fn as maml_loss_fn
from terrain_adaptation_rls.training.optimization import (
    build_optimizer,
    configure_step_learning_rate,
)
from terrain_adaptation_rls.training.supervised import (
    scene_supervised_batch,
    scene_trajectory_batch,
    write_prediction_artifacts,
    write_training_plots,
)


MAMLLossFn = Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.device], torch.Tensor]


def run_maml_training(
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
    max_steps: int | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train a Neural ODE initialization with first-order MAML updates."""

    if config.platform is None:
        raise ValueError("MAML training requires config.platform")

    train_scenes = [str(scene) for scene in config.data.get("train_scenes", ())]
    validation_scenes = [str(scene) for scene in config.data.get("validation_scenes", ())]
    if not train_scenes:
        raise ValueError("MAML training requires data.train_scenes")

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
    inner_lr = float(training.get("inner_learning_rate", training.get("inner_lr", 1e-2)))
    inner_steps = int(training.get("inner_steps", 1))
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
        current_lr = configure_step_learning_rate(
            optimizer,
            step=step,
            total_steps=steps,
            base_learning_rate=learning_rate,
            schedule=lr_schedule,
            warmup_steps=warmup_steps,
            final_lr_fraction=final_lr_fraction,
        )
        train_loss = meta_update_step(
            model=model,
            batch=batch,
            loss_fn=maml_loss_fn,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            meta_optimizer=optimizer,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )
        train_losses.append(train_loss)
        learning_rates.append(current_lr)

        if validation_batches and (step == steps or step % eval_interval == 0):
            model.eval()
            for scene, validation_batch in validation_batches.items():
                validation_loss = maml_query_loss(
                    model,
                    validation_batch,
                    loss_fn=maml_loss_fn,
                    inner_lr=inner_lr,
                    inner_steps=inner_steps,
                    device=device,
                )
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
        "family": "maml_neural_ode",
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
        "inner_learning_rate": inner_lr,
        "inner_steps": inner_steps,
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
        torch.save(model.state_dict(), artifact_path / "maml_model.pth")
        write_training_plots(artifact_path, metrics)
        if validation_artifact_batches:
            scene, validation_batch = next(iter(validation_artifact_batches.items()))
            adapted_model = adapt_model(
                model,
                support_data=_support_data(validation_batch),
                loss_fn=maml_loss_fn,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                device=device,
                clone=True,
            )
            write_prediction_artifacts(
                artifact_path,
                adapted_model,
                "neural_ode",
                validation_batch,
                scene=scene,
            )

    return metrics


def meta_update_step(
    *,
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    loss_fn: MAMLLossFn,
    inner_lr: float,
    inner_steps: int,
    meta_optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float = 0.0,
) -> float:
    """Apply one first-order MAML update and return query loss."""

    query_data = _query_data(batch)
    support_data = _support_data(batch)
    adapted_model = adapt_model(
        model,
        support_data=support_data,
        loss_fn=loss_fn,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        device=device,
        clone=True,
    )

    meta_optimizer.zero_grad(set_to_none=True)
    adapted_model.train()
    query_loss = loss_fn(adapted_model, query_data, device)
    adapted_parameters = tuple(adapted_model.parameters())
    grads = torch.autograd.grad(
        query_loss,
        adapted_parameters,
        allow_unused=True,
    )
    for parameter, grad in zip(model.parameters(), grads):
        parameter.grad = None if grad is None else grad.detach().clone()
    if gradient_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    meta_optimizer.step()
    return float(query_loss.detach().cpu())


def adapt_model(
    model: torch.nn.Module,
    *,
    support_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    loss_fn: MAMLLossFn,
    inner_lr: float,
    inner_steps: int,
    device: torch.device,
    clone: bool = True,
) -> torch.nn.Module:
    """Adapt a model on support data with a few SGD steps."""

    adapted_model = copy.deepcopy(model) if clone else model
    adapted_model.train()
    optimizer = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)
    for _ in range(max(0, inner_steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(adapted_model, support_data, device)
        loss.backward()
        optimizer.step()
    return adapted_model


def maml_query_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    *,
    loss_fn: MAMLLossFn,
    inner_lr: float,
    inner_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate query loss after adapting on the support half of a batch."""

    adapted_model = adapt_model(
        model,
        support_data=_support_data(batch),
        loss_fn=loss_fn,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        device=device,
        clone=True,
    )
    adapted_model.eval()
    return loss_fn(adapted_model, _query_data(batch), device)


def load_trained_maml(
    config: ExperimentConfig,
    train_run_dir: str | Path,
    *,
    device: torch.device | str,
) -> torch.nn.Module:
    """Load a trained MAML initialization from an artifact directory."""

    device = torch.device(device)
    n_basis = int(config.model.get("n_basis", 8))
    hidden_size = int(config.model.get("hidden_size", 128))
    model = create_model(device, n_basis=n_basis, hidden_size=hidden_size)
    state = torch.load(Path(train_run_dir) / "maml_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _query_data(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, dt, ys, *_ = batch
    return xs, dt, ys


def _support_data(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, _, example_xs, example_dt, example_ys = batch
    return example_xs, example_dt, example_ys
