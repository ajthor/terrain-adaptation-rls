"""Small neural network building blocks used by model constructors."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

import torch


class MLP(torch.nn.Module):
    """Fully connected network that preserves leading tensor dimensions."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        *,
        activation: torch.nn.Module | Callable[[], torch.nn.Module] = torch.nn.ReLU,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output dimensions")

        layers: list[torch.nn.Module] = []
        for input_dim, output_dim in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append(torch.nn.Linear(input_dim, output_dim))
            layers.append(_make_activation(activation))
        layers.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _make_activation(
    activation: torch.nn.Module | Callable[[], torch.nn.Module],
) -> torch.nn.Module:
    if isinstance(activation, torch.nn.Module):
        return copy.deepcopy(activation)
    return activation()
