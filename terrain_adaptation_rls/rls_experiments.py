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

import matplotlib.pyplot as plt

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
            traj_length = 1500

            # Sample random points from the data without replacement
            indices = torch.randperm(self.inputs[0].shape[0])
            example_indices = indices[: self.n_example_points]

            init_pt = torch.randint(self.inputs[0].shape[0]-2050, (1,)).item() 

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
scene = 0 # Only one scene for now

# Create a dataset for testing prediction errors. 
batch_size = 100
scene_input, scene_target = scene_inputs[scene], scene_targets[scene]
dataset = kStepTestDataset([scene_input], [scene_target], n_example_points=1000)
dataloader = DataLoader(dataset,batch_size=batch_size)
dataloader_iter = iter(dataloader)

# Load the model.
n_basis = 8 
seed = 0
model_path = f"logs/function_encoder/seed={seed}/function_encoder_model.pth"
model, loss_fn = load_model("function_encoder", device, n_basis, model_path)


# Evaluate model


model.eval()
with torch.no_grad():

    # Initialize the coefficients, matching an assumed batch size of 1
    coefficients = torch.zeros(batch_size, n_basis, device=device)
    P = torch.eye(n_basis, device=device).repeat(batch_size,1,1)#.unsqueeze(0)

    baseline_err_mean = []
    baseline_err_std = []
    rls_err_mean = []
    rls_err_std = []
    parameter_estimate_norms = []
    parameter_estimate_stds = []

    # Fetch new observation
    batch = next(dataloader_iter)
    x, dt, u, y, _, _, _ = batch
    x = x.to(device)
    dt = dt.to(device)
    u = u.to(device)
    y = y.to(device)

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


            # Compute the baseline error
            pred_baseline = model((torch.cat((x_step, u_step), dim=-1), dt_step), coefficients=coefficients_baseline)
            loss_baseline = torch.nn.functional.mse_loss(pred_baseline, y_step)

            # Compute the recursive least squares prediction error
            pred = model((torch.cat((x_step, u_step), dim=-1), dt_step), coefficients=coefficients)
            loss_rls = torch.nn.functional.mse_loss(pred, y_step)

            baseline_err_mean.append(torch.norm(y_step - pred_baseline, dim=-1).mean().item())
            baseline_err_std.append(torch.norm(y_step - pred_baseline, dim=-1).std().item())
            rls_err_mean.append(torch.norm(y_step - pred, dim=-1).mean().item())
            rls_err_std.append(torch.norm(y_step - pred, dim=-1).std().item())

            parameter_estimate_norms.append((coefficients - coefficients_baseline).norm(dim=-1).mean().item())
            parameter_estimate_stds.append((coefficients - coefficients_baseline).norm(dim=-1).std().item())

            tqdm_bar.set_postfix(
                {
                    "loss_baseline": f"{loss_baseline.item():.2e}",
                    "loss_rls": f"{loss_rls.item():.2e}",
                }
            )

# Use STIX fonts (LaTeX-style) and apply them consistently
plt.rcParams.update({
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# Plot the losses
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(baseline_err_mean, label="Baseline")
ax[0].fill_between(
    range(len(baseline_err_mean)),
    [x - y for x, y in zip(baseline_err_mean, baseline_err_std)],
    [x + y for x, y in zip(baseline_err_mean, baseline_err_std)],
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
)
ax[0].plot(rls_err_mean, label="RLS")
ax[0].fill_between(
    range(len(rls_err_mean)),
    [x - y for x, y in zip(rls_err_mean, rls_err_std)],
    [x + y for x, y in zip(rls_err_mean, rls_err_std)],
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
)
ax[0].set_xlabel("Lookahead Steps")
ax[0].set_ylabel("Single Step Mean Prediction Error")

ax[1].plot(parameter_estimate_norms)
ax[1].fill_between(
    range(len(parameter_estimate_stds)),
    [x - y for x, y in zip(parameter_estimate_norms, parameter_estimate_stds)],
    [x + y for x, y in zip(parameter_estimate_norms, parameter_estimate_stds)],
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
)
ax[1].set_xlabel("Lookahead Steps")
ax[1].set_ylabel("Norm of Coeff. Error (RLS - Baseline)")

ax[0].legend()

plt.tight_layout()
# plt.savefig(f"offline_rls_scene={scene}_forgetting=0_95.png", bbox_inches="tight", dpi=300)
# plt.close()
plt.show()
