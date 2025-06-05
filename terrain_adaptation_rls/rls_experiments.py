from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step

from function_encoder.coefficients import recursive_least_squares_update
from data.load_data import load_all_scenes
from torch.utils.data import IterableDataset
from typing import List

import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)


class kStepTestDataset(IterableDataset):
    def __init__(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        n_example_points: int,
    ):
        self.inputs = inputs
        self.targets = targets
        self.n_example_points = n_example_points

    def __iter__(self):
        while True:
            # Traj length
            traj_length = 500

            # Sample random points from the data without replacement
            indices = torch.randperm(self.inputs[0].shape[0])
            example_indices = indices[: self.n_example_points]

            init_pt = torch.randint(self.inputs[0].shape[0]-1050, (1,)).item() 

            _xs = self.inputs[0][:, 1:]
            _dt = self.targets[0][:, 0] - self.inputs[0][:, 0]
            _ys = self.targets[0][:, 1:] - _xs[:, :6]
            example_xs = _xs[example_indices]
            example_dt = _dt[example_indices]
            example_ys = _ys[example_indices]

            x = self.inputs[0][init_pt:init_pt+traj_length, 1:7]
            dt = _dt[init_pt:init_pt+traj_length]
            u = self.inputs[0][init_pt:init_pt+traj_length, 7:9]
            y = self.targets[0][init_pt:init_pt+traj_length, 1:] 

            yield x, dt, u, y, example_xs, example_dt, example_ys


def load_model(model_type, device, n_basis=8, path=None):
    match model_type:
        case "neural_ode":
            from models.neural_ode import load_model, loss_fn
            model = load_model(device=device, path=path, n_basis=n_basis).to(device)
        case "function_encoder":
            from models.function_encoder import load_model, loss_fn
            model = load_model(device=device, path=path, n_basis=n_basis).to(device)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
    return model, loss_fn

# Load dataset

scene_data = load_all_scenes()
training_scenes = [0, 2, 3, 4, 6, 7]
interpolation_scenes = [5]
extrapolation_scenes = [1]
all_scenes = interpolation_scenes + training_scenes + extrapolation_scenes

# Preload all inputs/targets
scene_inputs, scene_targets = {}, {}
for idx in all_scenes:
    key = f"scene{idx}"
    scene_inputs[idx], scene_targets[idx] = scene_data[key]

# Choose the evaluation scene
scene = all_scenes[0]  # Only one scene for now

# Create a dataset for testing prediction errors. 
scene_input, scene_target = scene_inputs[scene], scene_targets[scene]
dataset = kStepTestDataset([scene_input], [scene_target], n_example_points=1000)
dataloader = DataLoader(dataset,batch_size=1)
dataloader_iter = iter(dataloader)

# Create model

# Load the model.
n_basis = 8 
seed = 0
model_path = f"logs/function_encoder/seed={seed}/function_encoder_model.pth"
model, loss_fn = load_model("function_encoder", device, n_basis, model_path)


# Evaluate model

import matplotlib.pyplot as plt


model.eval()
with torch.no_grad():

    # Initialize the coefficients, matching an assumed batch size of 1
    coefficients = torch.zeros(1, n_basis, device=device)
    P = torch.eye(n_basis, device=device).unsqueeze(0)

    losses_baseline = []
    losses_rls = []
    coefficient_baseline_norms = []
    coefficient_rls_norms = []
    parameter_estimate_norms = []

    # Fetch new observation
    batch = next(dataloader_iter)
    x, dt, u, y, example_xs, example_dt, example_ys = batch
    print(x.shape, dt.shape, u.shape, y.shape, example_xs.shape, example_dt.shape, example_ys.shape)
    x = x.to(device)
    dt = dt.to(device)
    u = u.to(device)
    y = y.to(device)
    # example_xs = example_xs.to(device)
    # example_dt = example_dt.to(device)
    # example_ys = example_ys.to(device)

    # Compute baseline coefficients
    coefficients_baseline, _ = model.compute_coefficients((torch.cat((x, u), dim=-1), dt), y)

    with tqdm.trange(dt.shape[1]) as tqdm_bar:
        for step in tqdm_bar:
            # slice relevant quantities at the current step
            dt_step = dt[:, step].unsqueeze(1)#.unsqueeze(0)
            y_step = y[:, step].unsqueeze(1)
            x_step = x[:, step].unsqueeze(1)  # x is constant for the trajectory
            u_step = u[:, step].unsqueeze(1)  # u is constant for the trajectory

            # Compute the basis functions 
            # [batch_size, n_points, n_features, n_basis]
            g = model.basis_functions((torch.cat((x_step,u_step), dim=-1), dt_step))

            L = torch.linalg.cholesky(P)
            coefficients, P = recursive_least_squares_update(
                method='qr',
                g=g,
                y=y_step,
                P=L,
                coefficients=coefficients,
                forgetting_factor=0.95,
            )

            # Generate a new batch of data for evaluation
            # n_points = 1000
            # _y0 = torch.empty(1, n_points, 2, device=device).uniform_(*dataset.y0_range)
            # _dt = torch.empty(1, n_points, device=device).uniform_(*dataset.dt_range)
            # _y1 = rk4_step(van_der_pol, _y0, _dt, mu=mu)

            # n_example_points = 100
            # y0_example = _y0[:, :n_example_points, :]
            # dt_example = _dt[:, :n_example_points]
            # y1_example = _y1[:, :n_example_points, :]
            # y0 = _y0[:, n_example_points:, :]
            # dt = _dt[:, n_example_points:]
            # y1 = _y1[:, n_example_points:, :]

            # Compute the baseline error
            pred_baseline = model((torch.cat((x_step, u_step), dim=-1), dt_step), coefficients=coefficients_baseline)
            # coefficients_baseline, _ = model.compute_coefficients(
            #     (y0_example, dt_example), y1_example
            # )
            # pred_baseline = model((y0, dt), coefficients=coefficients_baseline)
            loss_baseline = torch.nn.functional.mse_loss(pred_baseline, y_step)

            # # Compute the recursive least squares prediction error
            pred = model((torch.cat((x_step, u_step), dim=-1), dt_step), coefficients=coefficients)
            # pred = model((y0, dt), coefficients=coefficients)
            loss_rls = torch.nn.functional.mse_loss(pred, y_step)

            losses_baseline.append(loss_baseline.item())
            losses_rls.append(loss_rls.item())

            # coefficient_baseline_norms.append(
            #     coefficients_baseline.norm(dim=-1).mean().item()
            # )
            coefficient_rls_norms.append(coefficients.norm(dim=-1).mean().item())

            parameter_estimate_norms.append((coefficients - coefficients_baseline).norm(dim=-1).mean().item())

            tqdm_bar.set_postfix(
                {
                    "loss_baseline": f"{loss_baseline.item():.2e}",
                    "loss_rls": f"{loss_rls.item():.2e}",
                }
            )

    # Plot the losses
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].plot(losses_baseline, label="Baseline")
    ax[0].plot(losses_rls, label="Recursive Least Squares")

    ax[1].plot(coefficient_baseline_norms, label="Baseline Coefficients Norm")
    ax[1].plot(coefficient_rls_norms, label="RLS Coefficients Norm")

    ax[2].plot(parameter_estimate_norms, label="Parameter Estimate Norm (RLS - Baseline)")
    
    plt.show()
