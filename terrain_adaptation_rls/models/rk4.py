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
