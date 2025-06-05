import torch
from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc
from function_encoder.function_encoder import FunctionEncoder, BasisFunctions
from .rk4 import rk4_step


def create_model(device, n_basis=8):
    """
    Create a FunctionEncoder model instance.
    device: torch device string
    n_basis: number of basis functions
    """

    def basis_factory():
        return NeuralODE(
            ode_func=ODEFunc(
                model=MLP(layer_sizes=[9, 128, 128, 6], activation=torch.nn.ReLU())
            ),
            integrator=rk4_step,
        )

    basis_functions = BasisFunctions(*[basis_factory() for _ in range(n_basis)])
    model = FunctionEncoder(basis_functions).to(device)
    return model


def save_model(model, path):
    """Save a FunctionEncoder model to a file."""
    torch.save(model.state_dict(), path)


def load_model(device, path, n_basis=8):
    """Load a FunctionEncoder model from a file."""
    model = create_model(device, n_basis)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def loss_fn(model, batch, device):
    xs, dt, ys, example_xs, example_dt, example_ys = batch
    xs = xs.to(device)
    dt = dt.to(device)
    ys = ys.to(device)
    example_xs = example_xs.to(device)
    example_dt = example_dt.to(device)
    example_ys = example_ys.to(device)
    coefficients, _ = model.compute_coefficients((example_xs, example_dt), example_ys)
    pred = model((xs, dt), coefficients=coefficients)
    loss = torch.nn.functional.mse_loss(pred, ys)
    return loss
