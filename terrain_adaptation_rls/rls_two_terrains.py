"""
FE initialized with coefficients from scene A data. 
Deploy the model on scene A. 
Change the dynamics to scene B.
Update the coefficients using RLS and sequential data from B. 
"""

import torch
from torch.utils.data import DataLoader

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

torch.manual_seed(0)


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
            traj_length = 200

            # Sample random points from the data without replacement
            indices = torch.randperm(self.inputs[0].shape[0])
            example_indices = indices[: self.n_example_points]

            init_pt = torch.randint(self.inputs[0].shape[0]-250, (1,)).item() 

            _xs = self.inputs[0][:, 1:]
            _dt = self.targets[0][:, 0] - self.inputs[0][:, 0]
            _ys = self.targets[0][:, 1:] - _xs[:, :6]
            example_xs = _xs[example_indices]
            example_dt = _dt[example_indices]
            example_ys = _ys[example_indices]

            x = self.inputs[0][init_pt:init_pt+traj_length, 1:7]
            dt = _dt[init_pt:init_pt+traj_length]
            u = self.inputs[0][init_pt:init_pt+traj_length, 7:9]
            y = self.targets[0][init_pt:init_pt+traj_length, 1:] - x

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
scene = 5 # Only one scene for now
scene2 = 1

# Create a dataset for testing prediction errors. 
batch_size = 200
scene_input, scene_target = scene_inputs[scene], scene_targets[scene]
dataset = kStepTestDataset([scene_input], [scene_target], n_example_points=1000)
dataloader = DataLoader(dataset,batch_size=batch_size)
dataloader_iter = iter(dataloader)

scene_input2, scene_target2 = scene_inputs[scene2], scene_targets[scene2]
dataset2 = kStepTestDataset([scene_input2], [scene_target2], n_example_points=1000)
dataloader2 = DataLoader(dataset2,batch_size=batch_size)
dataloader_iter2 = iter(dataloader2)


# Load the model.
n_basis = 8 
seed = 0
model_path = f"logs/function_encoder/seed={seed}/function_encoder_model.pth"
model, loss_fn = load_model("function_encoder", device, n_basis, model_path)
node_model_path = f"logs/neural_ode/seed={seed}/neural_ode_model.pth"
node_model, node_loss_fn = load_model("neural_ode", device, n_basis, node_model_path)


# Evaluate model

node_model.eval()
model.eval()
with torch.no_grad():

    # Allocate memory for error plotting.
    baseline_err_med = []
    baseline_err_10th = []
    baseline_err_90th = []

    rls_err_med = []
    rls_err_10th = []
    rls_err_90th = []

    node_err_med = []
    node_err_10th = []
    node_err_90th = []

    parameter_estimate_norms = []
    parameter_estimate_stds = []

    # Fetch new observation
    batch = next(dataloader_iter)
    x, dt, u, y, _, _, _ = batch
    x = x.to(device)
    dt = dt.to(device)
    u = u.to(device)
    y = y.to(device)

    batch2 = next(dataloader_iter2)
    x2, dt2, u2, y2, _, _, _ = batch2
    x2 = x2.to(device)
    dt2 = dt2.to(device)
    u2 = u2.to(device)
    y2 = y2.to(device)

    # Initialize the coefficients using the first terrain.
    coefficients, _ = model.compute_coefficients((torch.cat((x, u), dim=-1), dt), y)
    P = torch.eye(n_basis, device=device).repeat(batch_size,1,1)#.unsqueeze(0)

    # Compute baseline coefficients
    coefficients_baseline, _ = model.compute_coefficients((torch.cat((x, u), dim=-1), dt), y)

    coefficients_baseline2, _ = model.compute_coefficients((torch.cat((x2, u2), dim=-1), dt2), y2)

    with tqdm.trange(dt.shape[1]) as tqdm_bar:
        for step in tqdm_bar:
            if step < 50:
                # slice relevant quantities at the current step
                dt_step = dt[:, step].unsqueeze(1)#.unsqueeze(0)
                y_step = y[:, step].unsqueeze(1)
                x_step = x[:, step].unsqueeze(1)  # x is constant for the trajectory
                u_step = u[:, step].unsqueeze(1)  # u is constant for the trajectory
            else:
                dt_step = dt2[:, step].unsqueeze(1)#.unsqueeze(0)
                y_step = y2[:, step].unsqueeze(1)
                x_step = x2[:, step].unsqueeze(1)  # x is constant for the trajectory
                u_step = u2[:, step].unsqueeze(1)  # u is constant for the trajectory


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
            if step < 50:
                pred_baseline = model((torch.cat((x_step, u_step), dim=-1), dt_step), coefficients=coefficients_baseline)        
            else:
                pred_baseline = model((torch.cat((x_step, u_step), dim=-1), dt_step), coefficients=coefficients_baseline2)        
            loss_baseline = torch.nn.functional.mse_loss(pred_baseline, y_step)

            # Compute the recursive least squares prediction error
            pred = model((torch.cat((x_step, u_step), dim=-1), dt_step), coefficients=coefficients)
            loss_rls = torch.nn.functional.mse_loss(pred, y_step)

            # Compute the neural ODE error
            node_pred = node_model((torch.cat((x_step, u_step), dim=-1), dt_step))
            loss_node = torch.nn.functional.mse_loss(node_pred, y_step)

            baseline_err_med.append(torch.norm(y_step - pred_baseline, dim=-1).median().item())
            baseline_err_10th.append(torch.norm(y_step - pred_baseline, dim=-1).quantile(0.10).item())
            baseline_err_90th.append(torch.norm(y_step - pred_baseline, dim=-1).quantile(0.90).item())

            rls_err_med.append(torch.norm(y_step - pred, dim=-1).median().item())
            rls_err_10th.append(torch.norm(y_step - pred, dim=-1).quantile(0.10).item())
            rls_err_90th.append(torch.norm(y_step - pred, dim=-1).quantile(0.90).item())

            node_err_med.append(torch.norm(y_step - node_pred, dim=-1).median().item())
            node_err_10th.append(torch.norm(y_step - node_pred, dim=-1).quantile(0.10).item())
            node_err_90th.append(torch.norm(y_step - node_pred, dim=-1).quantile(0.90).item())

            parameter_estimate_norms.append((coefficients - coefficients_baseline).norm(dim=-1).mean().item())
            parameter_estimate_stds.append((coefficients - coefficients_baseline).norm(dim=-1).std().item())
            

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
fig1 = plt.figure(figsize=(3.5, 2.5))

plt.plot(node_err_med, label="NODE", color="#D62728")
plt.fill_between(
    range(len(node_err_med)),
    node_err_10th,
    node_err_90th,
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
    color="#D62728"
)

plt.plot(baseline_err_med, label="FE", color="#1F77B4")
plt.fill_between(
    range(len(baseline_err_med)),
    baseline_err_10th,
    baseline_err_90th,
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
    color="#1F77B4"
)

plt.plot(rls_err_med, label="FE-RLS", color="#2ca02c")
plt.fill_between(
    range(len(rls_err_med)),
    rls_err_10th,
    rls_err_90th,
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
    color="#2ca02c"
)

plt.yscale("log")
plt.xlabel("Time Steps")
plt.ylabel("Single Step Mean Prediction Error")

fig1.legend(
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=3,
    frameon=False,
)

plt.tight_layout()
plt.savefig(f"plots/rls_two_terrains/scene={scene}_to_scene={scene2}", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()


fig2, ax2 = plt.subplots(figsize=(3.5,2.5))

ax2.plot(parameter_estimate_norms)
ax2.fill_between(
    range(len(parameter_estimate_stds)),
    [x - y for x, y in zip(parameter_estimate_norms, parameter_estimate_stds)],
    [x + y for x, y in zip(parameter_estimate_norms, parameter_estimate_stds)],
    alpha=0.2,
    edgecolor='none',
    linewidth=0.0,
)
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Normed Coefficient Error")
fig2.tight_layout()
plt.savefig(f"plots/rls_two_terrains/scene={scene}_to_scene={scene2}_coeffs", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()
