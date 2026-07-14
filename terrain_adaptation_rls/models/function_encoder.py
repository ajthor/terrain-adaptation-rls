import torch
from function_encoder.model.neural_ode import NeuralODE, ODEFunc
from function_encoder.function_encoder import FunctionEncoder, BasisFunctions
from .networks import MLP
from .rk4 import make_augmented_rk4_delta_step, rk4_step


def create_model(device, n_basis=8, hidden_size=128, augmentation_dim=0):
    """
    Create a FunctionEncoder model instance.
    device: torch device string
    n_basis: number of basis functions
    """

    augmentation_dim = int(augmentation_dim)
    if augmentation_dim < 0:
        raise ValueError("augmentation_dim must be non-negative")
    input_dim = 9 + augmentation_dim
    output_dim = 6 + augmentation_dim
    integrator = (
        rk4_step
        if augmentation_dim == 0
        else make_augmented_rk4_delta_step(augmentation_dim)
    )

    def basis_factory():
        return NeuralODE(
            ode_func=ODEFunc(
                model=MLP(
                    layer_sizes=[input_dim, hidden_size, hidden_size, output_dim],
                    activation=torch.nn.ReLU(),
                )
            ),
            integrator=integrator,
        )

    basis_functions = BasisFunctions(*[basis_factory() for _ in range(n_basis)])
    model = FunctionEncoder(basis_functions).to(device)
    model.augmentation_dim = augmentation_dim
    return model


def save_model(model, path):
    """Save a FunctionEncoder model to a file."""
    torch.save(model.state_dict(), path)


def load_model(device, path, n_basis=8, hidden_size=128, augmentation_dim=0):
    """Load a FunctionEncoder model from a file."""
    model = create_model(device, n_basis, hidden_size, augmentation_dim=augmentation_dim)
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

def rls_loss_fn(model, batch, coeffs, device):
    xs, dt, ys, _, _, _ = batch
    xs = xs.to(device)
    dt = dt.to(device)
    ys = ys.to(device)
    pred = model((xs, dt), coefficients=coeffs)
    loss = torch.nn.functional.mse_loss(pred, ys)
    return loss
