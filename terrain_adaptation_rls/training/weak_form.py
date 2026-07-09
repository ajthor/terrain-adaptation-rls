"""Weak-form utilities for linear-in-coefficients dynamics models."""

from __future__ import annotations

import torch


def sine_test_functions_from_dt(
    dt: torch.Tensor,
    *,
    n_tests: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build endpoint-vanishing sine test functions on batched time grids.

    Args:
        dt: Per-sample timestep with shape ``[batch, time]``.
        n_tests: Number of sine test functions.

    Returns:
        ``(phi, phi_prime)`` with shape ``[batch, n_tests, time]``.
    """

    if dt.ndim != 2:
        raise ValueError(f"dt must have shape [batch, time], got {tuple(dt.shape)}")
    if n_tests <= 0:
        raise ValueError("n_tests must be positive")

    # Use the left endpoint of each sample interval as the quadrature location.
    t = torch.cumsum(dt, dim=-1) - dt
    horizon = t[..., -1:].clamp_min(torch.finfo(dt.dtype).eps)
    tau = t / horizon

    ks = torch.arange(1, n_tests + 1, device=dt.device, dtype=dt.dtype).view(1, -1, 1)
    tau = tau.unsqueeze(1)
    horizon = horizon.unsqueeze(1)

    phi = torch.sin(torch.pi * ks * tau)
    phi_prime = (torch.pi * ks / horizon) * torch.cos(torch.pi * ks * tau)
    return phi, phi_prime


def weak_system_from_basis(
    state: torch.Tensor,
    basis: torch.Tensor,
    dt: torch.Tensor,
    *,
    n_tests: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the weak linear system ``B ~= A c``.

    Args:
        state: State trajectory with shape ``[batch, time, state_dim]``.
        basis: Vector-field basis values with shape
            ``[batch, time, state_dim, n_basis]``.
        dt: Per-sample timestep with shape ``[batch, time]``.
        n_tests: Number of weak test functions.

    Returns:
        ``(A, B)`` with shapes ``[batch, n_tests, state_dim, n_basis]`` and
        ``[batch, n_tests, state_dim]``.
    """

    if state.ndim != 3:
        raise ValueError(f"state must have shape [batch, time, state_dim], got {state.shape}")
    if basis.ndim != 4:
        raise ValueError(
            "basis must have shape [batch, time, state_dim, n_basis], "
            f"got {basis.shape}"
        )
    if state.shape != basis.shape[:-1]:
        raise ValueError(f"state/basis dimensions do not match: {state.shape} vs {basis.shape}")
    if dt.shape != state.shape[:2]:
        raise ValueError(f"dt shape {dt.shape} does not match state batch/time {state.shape[:2]}")

    phi, phi_prime = sine_test_functions_from_dt(dt, n_tests=n_tests)
    weak_target = -torch.einsum("bmt,btd,bt->bmd", phi_prime, state, dt)
    weak_basis = torch.einsum("bmt,btdk,bt->bmdk", phi, basis, dt)
    return weak_basis, weak_target


def solve_weak_coefficients(
    weak_basis: torch.Tensor,
    weak_target: torch.Tensor,
    *,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """Solve batched weak-form ridge least squares coefficients."""

    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    if weak_basis.shape[:-1] != weak_target.shape:
        raise ValueError(
            "weak basis and target must share batch/test/state dimensions: "
            f"{weak_basis.shape[:-1]} != {weak_target.shape}"
        )

    batch_size = weak_basis.shape[0]
    n_coeff = weak_basis.shape[-1]
    design = weak_basis.reshape(batch_size, -1, n_coeff)
    response = weak_target.reshape(batch_size, -1)

    gram = torch.matmul(design.transpose(-1, -2), design)
    rhs = torch.matmul(design.transpose(-1, -2), response.unsqueeze(-1)).squeeze(-1)
    eye = torch.eye(n_coeff, dtype=weak_basis.dtype, device=weak_basis.device)
    return torch.linalg.solve(gram + ridge * eye.unsqueeze(0), rhs)
