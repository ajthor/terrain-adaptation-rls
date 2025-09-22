"""
This script evaluates the k-step prediction error of different models on a specific scene.
For each model type, it loads a model trained with a specific seed. Then, it choose a
bunch of random initial states from the scene dataset. For each initial state, it
propagates the state forward in time using the model and the real control inputs. Then
it calculates the norm of the error between the predicted and actual next states. 
After repeating for all models, it calculates the accumulated error over all k steps.
Then it plots the accumulated error for each model type, showing the mean, median,
standard deviation, and first/third quartiles of the errors across all rollouts. """

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.load_data import load_scenes
from train_utils import inertial_to_body
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from typing import List

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

            x0 = self.inputs[0][init_pt, 1:7]
            dt = _dt[init_pt:init_pt+100]
            u = self.inputs[0][init_pt:init_pt+100, 7:9]
            y = self.targets[0][init_pt:init_pt+100, 1:] 

            yield x0, dt, u, y, example_xs, example_dt, example_ys


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


# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Config
n_basis = 8
hidden_size = 128
num_samples = 200
torch.manual_seed(30)
seeds = [0,1,2,3,4]
model_types = ["neural_ode", "function_encoder"]  

platform = 'warty'
scene = f"scene1"
scene_data = load_scenes([scene], platform)

# Preload all inputs/targets
scene_inputs, scene_targets = {}, {}
scene_inputs[scene], scene_targets[scene] = scene_data[scene]

# Evaluate
all_results = {model_type: {seed: [] for seed in range(10)} for model_type in model_types}

# Create a dataset for testing prediction errors. 
scene_input, scene_target = scene_inputs[scene], scene_targets[scene]
dataset = kStepTestDataset([scene_input], [scene_target], n_example_points=1000)
dataloader = DataLoader(dataset,batch_size=100)
dataloader_iter = iter(dataloader)


with torch.no_grad():
    # Repeat for every random seed
    for seed in seeds:
        # Repeat evaluation for each model type
        for model_type in model_types:
            # Load the model. 
            model_path = f"logs/{platform}/{model_type}/seed={seed}/hidden_size={hidden_size}/n_basis={n_basis}/{model_type}_model.pth"
            if not os.path.exists(model_path):
                print(f"Missing: {model_path}")
                exit()
            model, loss_fn = load_model(model_type, device, n_basis, model_path)

            # Get the next batch from the dataloader.
            batch = next(dataloader_iter)
            x0, dt, u, y, example_xs, example_dt, example_ys = batch

            x0 = x0.to(device)
            dt = dt.to(device)
            u = u.to(device)
            y = y.to(device)
            example_xs = example_xs.to(device)
            example_dt = example_dt.to(device)
            example_ys = example_ys.to(device)

            if model_type == "function_encoder":
                # Compute the basis coefficients
                coefficients, _ = model.compute_coefficients((example_xs, example_dt), example_ys)
            
            # Make a copy of the current state for processing.
            _x = x0.clone()
            err_list = []

            # Loop over the k steps and predict the next state.
            for k in range(dt.shape[1]):

                # Predict the next state and save the prediction. 
                if model_type == "function_encoder":
                    del_x = model((torch.cat((_x, u[:,k,:]), dim=1).unsqueeze(1), dt[:,k].unsqueeze(1)), coefficients=coefficients)
                elif model_type == "neural_ode":
                    del_x = model((torch.cat((_x, u[:,k,:]), dim=1).unsqueeze(1), dt[:,k].unsqueeze(1)))
                del_x = del_x.squeeze(1)

                # Get the next velocity in the initial body frame.
                next_vel_Bi = _x[:,3:6] + del_x[:,3:6]

                # Transform the velocity back to the body frame.
                next_vel_B = inertial_to_body(
                    bIMat=del_x[:,:3],
                    xIMat=next_vel_Bi,
                    device=device
                )

                # Prepare the new current state. 
                _x = torch.cat((torch.zeros((100,3), device=device), next_vel_B), dim=1)

                # Predict the change in pose/heading and the NEXT velocity. 
                pred = torch.cat((del_x[:,:3], next_vel_Bi), dim=1)
                err_list.append(torch.norm(y[:,k,:] - pred, dim=-1))

            error = torch.stack(err_list, dim=1)
            all_results[model_type][seed] = error.cpu().numpy()



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

# Plotting
fig = plt.figure(figsize=(3.5, 2.5))
colors = {"neural_ode": "#D62728", "function_encoder": "#1F77B4"}
markers = {"neural_ode": "o", "function_encoder": "^"}
names = {"neural_ode": "Neural ODE", "function_encoder": "Function Encoder"}

for model_type in model_types:

    # Collect the errors for all seeds
    all_errors = np.array([all_results[model_type][seed] for seed in seeds])
    all_errors = all_errors.reshape(-1, all_errors.shape[2])

    # Plot the accumulated errors from all seeds
    accum = np.cumsum(all_errors, axis=1)
    # Plot the mean and median
    plt.plot(np.median(accum, axis=0), label=names[model_type], color=colors[model_type])
    # Plot the first and third quartiles as a shaded area with dashed borders
    q1 = np.percentile(accum, 10, axis=0)
    q3 = np.percentile(accum, 90, axis=0)
    plt.fill_between(
        np.arange(accum.shape[1]),
        q1,
        q3,
        alpha=0.2,
        color=colors[model_type], 
        edgecolor="none",
        linewidth=0.0,
    )

# plt.yscale("log")
plt.ylabel("Accumulated Rollout MSE")
plt.xlabel("Lookahead Steps")
plt.ylim(0, 26)
plt.xlim(0, 100)

fig.legend(
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=2,
    frameon=False,
)

plt.tight_layout()
save_path = f'plots/{platform}'
os.makedirs(save_path, exist_ok=True)
plt.savefig(f"{save_path}/{scene}_error_k_step.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()
