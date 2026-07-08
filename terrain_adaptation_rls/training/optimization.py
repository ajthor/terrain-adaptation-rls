"""Small optimizer and learning-rate helpers for direct training loops."""

from __future__ import annotations

import math

import torch


def build_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Build the configured optimizer with conservative defaults."""

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"unknown optimizer: {optimizer_name}")


def learning_rate_scale(
    *,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_steps: int = 0,
    final_lr_fraction: float = 0.1,
) -> float:
    """Return a multiplicative learning-rate scale for one training step."""

    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step <= warmup_steps:
        return max(step / warmup_steps, 1e-6)
    if schedule == "none":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"unknown learning-rate schedule: {schedule}")

    decay_steps = max(total_steps - max(warmup_steps, 0), 1)
    decay_step = min(max(step - max(warmup_steps, 0), 0), decay_steps)
    progress = decay_step / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr_fraction + (1.0 - final_lr_fraction) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Set the learning rate on every optimizer parameter group."""

    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def configure_step_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    base_learning_rate: float,
    schedule: str,
    warmup_steps: int,
    final_lr_fraction: float,
) -> float:
    """Apply and return the learning rate for the current step."""

    scale = learning_rate_scale(
        step=step,
        total_steps=total_steps,
        schedule=schedule,
        warmup_steps=warmup_steps,
        final_lr_fraction=final_lr_fraction,
    )
    learning_rate = base_learning_rate * scale
    set_optimizer_lr(optimizer, learning_rate)
    return learning_rate
