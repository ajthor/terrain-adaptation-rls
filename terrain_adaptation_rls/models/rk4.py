import torch


def rk4_step(func, x, dt, **ode_kwargs):
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
