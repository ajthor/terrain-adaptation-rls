import torch
from math import sqrt
from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc
from .rk4 import rk4_step


def create_model(device, n_basis=8, hidden_size=128):
    """
    Create a MAML model instance.
    device: torch device string
    n_basis: controls hidden layer size
    """
    model = NeuralODE(
        ode_func=ODEFunc(
            model=MLP(
                layer_sizes=[9, int(hidden_size * sqrt(n_basis)), int(hidden_size * sqrt(n_basis)), 6],
                activation=torch.nn.ReLU(),
            )
        ),
        integrator=rk4_step,
    ).to(device)
    return model


def save_model(model, path):
    """Save a MAML model to a file."""
    torch.save(model.state_dict(), path)


def load_model(device, path, n_basis=8, hidden_size=128):
    """Load a MAML model from a file."""
    model = create_model(device, n_basis, hidden_size)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def loss_fn(model, batch):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    xs, dt, ys, *_ = batch  # ignore example_xs, example_dt, example_ys if present
    xs = xs.to(device)
    dt = dt.to(device)
    ys = ys.to(device)
    pred = model((xs, dt))
    loss = torch.nn.functional.mse_loss(pred, ys)
    return loss
