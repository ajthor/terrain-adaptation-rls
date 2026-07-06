"""Small supervised training utilities for FE/NODE smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
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
