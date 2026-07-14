import torch


def rk4_delta_step(func, x, dt, **ode_kwargs):
    """Integrate augmented dynamics and return only the state delta.

    ``x`` is expected to contain six state dimensions followed by two controls.
    The controls are held fixed while RK4 advances the state dimensions through
    the substeps. The returned tensor has shape ``[..., 6]`` and should be
    added to the current state by rollout code when a next state is needed.
    """

    t = torch.zeros_like(dt, device=dt.device)
    u = x[..., 6:8]

    k1 = func(t, x, **ode_kwargs)

    _k1 = torch.cat([x[..., :6] + (dt / 2).unsqueeze(-1) * k1, u], dim=-1)
    k2 = func(t + dt / 2, _k1, **ode_kwargs)

    _k2 = torch.cat([x[..., :6] + (dt / 2).unsqueeze(-1) * k2, u], dim=-1)
    k3 = func(t + dt / 2, _k2, **ode_kwargs)

    _k3 = torch.cat([x[..., :6] + dt.unsqueeze(-1) * k3, u], dim=-1)
    k4 = func(t + dt, _k3, **ode_kwargs)

    return (dt / 6).unsqueeze(-1) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_state_step(func, x, dt, **ode_kwargs):
    """Integrate augmented dynamics and return the next augmented state.

    This is the explicit next-state companion to ``rk4_delta_step``. It is
    useful for rollout code, but should not be used as the FE/NODE/MAML model
    integrator while their supervised target remains ``next_state - state``.
    """

    delta = rk4_delta_step(func, x, dt, **ode_kwargs)
    return torch.cat([x[..., :6] + delta, x[..., 6:8]], dim=-1)


rk4_step = rk4_delta_step


def make_augmented_rk4_delta_step(augmentation_dim: int):
    """Build an RK4 delta integrator with zero-initialized hidden state dims.

    The external model input remains ``[..., 8]``: six physical state dimensions
    followed by two controls. Internally, RK4 advances ``6 + augmentation_dim``
    state dimensions while holding controls fixed. Only the physical six-state
    delta is returned, preserving the FE/NODE runtime contract.
    """

    if augmentation_dim < 0:
        raise ValueError("augmentation_dim must be non-negative")
    if augmentation_dim == 0:
        return rk4_delta_step

    augmented_state_dim = 6 + augmentation_dim

    def augmented_rk4_delta_step(func, x, dt, **ode_kwargs):
        t = torch.zeros_like(dt, device=dt.device)
        u = x[..., 6:8]
        zeros = torch.zeros(
            (*x.shape[:-1], augmentation_dim),
            dtype=x.dtype,
            device=x.device,
        )
        z0 = torch.cat([x[..., :6], zeros], dim=-1)
        z = torch.cat([z0, u], dim=-1)

        k1 = func(t, z, **ode_kwargs)

        z1 = torch.cat([z0 + (dt / 2).unsqueeze(-1) * k1, u], dim=-1)
        k2 = func(t + dt / 2, z1, **ode_kwargs)

        z2 = torch.cat([z0 + (dt / 2).unsqueeze(-1) * k2, u], dim=-1)
        k3 = func(t + dt / 2, z2, **ode_kwargs)

        z3 = torch.cat([z0 + dt.unsqueeze(-1) * k3, u], dim=-1)
        k4 = func(t + dt, z3, **ode_kwargs)

        delta = (dt / 6).unsqueeze(-1) * (k1 + 2 * k2 + 2 * k3 + k4)
        return delta[..., :6]

    augmented_rk4_delta_step.augmentation_dim = augmentation_dim
    augmented_rk4_delta_step.state_dim = augmented_state_dim
    return augmented_rk4_delta_step
