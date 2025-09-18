import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.load_data import TestDataset
from train_utils import test_eval


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
platform = 'warty'
seeds = [0,1,2,3,4]
model_types = ["neural_ode", "function_encoder"]

if platform == 'warty':
    training_scenes = ['scene0', 'scene2', 'scene3', 'scene4', 'scene6', 'scene7']
    interpolation_scenes = ['scene5']
    extrapolation_scenes = ['scene1']
    all_scenes = training_scenes + interpolation_scenes + extrapolation_scenes
if platform == 'jackal_0770':
    training_scenes = ['grass', 'gym_floor', 'ice', 'mulch', 'pavement', 'turf']
    all_scenes = training_scenes


# Evaluate
all_results = {model_type: {scene: [] for scene in all_scenes} for model_type in model_types}

for seed in seeds:
    torch.manual_seed(seed)
    
    for scene in all_scenes:
        # Load pre-split data.
        load_path = f"terrain_adaptation_rls/data_split/{platform}/seed_{seed}/{scene}"
        test_input_df = pd.read_csv(f"{load_path}/test_input.csv", header=None) 
        test_target_df = pd.read_csv(f"{load_path}/test_target.csv", header=None) 
        test_input = torch.tensor(test_input_df.values).float()
        test_target = torch.tensor(test_target_df.values).float()

        dataset = TestDataset([test_input], [test_target], n_example_points=100)

        for model_type in model_types:
            model_path = f"logs/{platform}/{model_type}/seed={seed}/hidden_size={hidden_size}/n_basis={n_basis}/{model_type}_model.pth"
            if not os.path.exists(model_path):
                print(f"Skipping missing: {model_path}")
                continue
            model, loss_fn = load_model(model_type, device, n_basis, model_path)

            error = sum(test_eval(model, loss_fn, batch, device) for batch in dataset) / len(dataset)

            all_results[model_type][scene].append(error)

# Plotting
if platform == 'warty':
    order = ['scene0', 'scene6', 'scene7', 'scene2', 'scene5', 'scene3', 'scene4', 'scene1']
    scene_labels_ordered = [f"Scene {i[-1]}" for i in order]
    x_vals = [0, 0.25, 0.5, 0.75, 0.812, 0.875, 0.939, 1]
elif platform == 'jackal_0770':
    order = ['pavement', 'gym_floor', 'turf', 'grass', 'mulch',  'ice']
    scene_labels_ordered = ['Pavement', 'Gym Floor', 'Turf', 'Grass', 'Mulch',  'Ice']
    x_vals = [0, 0.5, 1.0, 1.5, 2.0, 2.5]

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

fig = plt.figure(figsize=(3.5, 2.5))
colors = {"neural_ode": "#D62728", "function_encoder": "#1F77B4"}
markers = {"neural_ode": "o", "function_encoder": "^"}
names = {"neural_ode": "Neural ODE", "function_encoder": "Function Encoder"}

for model_type in model_types:
    all_errors = np.array([[all_results[model_type][scene][i] for scene in order]
                           for i in range(len(seeds)) if len(all_results[model_type][order[0]]) > i])
    # all_errors shape: (num_runs, num_scenes)
    if all_errors.shape[0] == 0:
        continue

    med = np.median(all_errors, axis=0)
    min_ = np.min(all_errors, axis=0)
    max_ = np.max(all_errors, axis=0)

    # Plot the median
    plt.plot(x_vals, med, label=names[model_type], marker=markers[model_type], color=colors[model_type])
    # Plot the min/max
    plt.fill_between(
        x_vals, 
        min_, 
        max_, 
        alpha=0.2, 
        color=colors[model_type],
        edgecolor="none",
        linewidth=0.0,
    )

plt.yscale("log")
plt.ylabel("Mean Absolute Error")
plt.xticks(ticks=x_vals, labels=scene_labels_ordered)
plt.xticks(rotation=45, ha="right")

fig.legend(
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=2,
    frameon=False,
)

plt.tight_layout()
save_path = f'plots/{platform}'
os.makedirs(save_path, exist_ok=True)
plt.savefig(f"{save_path}/scene_error_over_seeds.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()
