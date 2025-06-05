from itertools import chain

import torch
import numpy as np
import matplotlib.pyplot as plt
from data.load_data import load_all_scenes, PhoenixDataset, TestDataset
from train_utils import test_eval


# Model loading utilities
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


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

n_basis = 8
model_types = [
    ("neural_ode", "logs/neural_ode/seed=42/neural_ode_model.pth"),
    ("function_encoder", "logs/function_encoder/seed=42/function_encoder_model.pth"),
]
seed = 42
torch.manual_seed(seed)
scene_data = load_all_scenes()
training_scenes = [0, 2, 3, 4, 6, 7]
interpolation_scenes = [5]
extrapolation_scenes = [1]

# Split into train (80%) and test (20%)
train_inputs = []
train_targets = []
unseen_test_inputs = []
unseen_test_targets = []
for scene_index in training_scenes:
    scene_str = f"scene{scene_index}"
    scene_input, scene_target = scene_data[scene_str]
    total_points = scene_input.shape[0]
    split_idx = int(0.8 * total_points)

    indices = torch.randperm(total_points)

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_inputs.append(scene_input[train_indices])
    train_targets.append(scene_target[train_indices])
    unseen_test_inputs.append(scene_input[test_indices])
    unseen_test_targets.append(scene_target[test_indices])

# grab the interpolation scenes
interpolation_inputs, interpolation_targets = [], []
for scene_index in interpolation_scenes:
    scene_str = f"scene{scene_index}"
    scene_input, scene_target = scene_data[scene_str]
    interpolation_inputs.append(scene_input)
    interpolation_targets.append(scene_target)

# grab the extrapolation scenes
extrapolation_inputs, extrapolation_targets = [], []
for scene_index in extrapolation_scenes:
    scene_str = f"scene{scene_index}"
    scene_input, scene_target = scene_data[scene_str]
    extrapolation_inputs.append(scene_input)
    extrapolation_targets.append(scene_target)


scene_labels = training_scenes + interpolation_scenes + extrapolation_scenes
results = {}
for model_type, model_path in model_types:
    model, loss_fn = load_model(model_type, device, n_basis, model_path)
    scene_errors = {}
    for dataset_index, inputs, targets in zip(training_scenes + interpolation_scenes + extrapolation_scenes,
                                              chain(train_inputs, interpolation_inputs, extrapolation_inputs),
                                              chain(train_targets, interpolation_targets, extrapolation_targets)):
        dataset = TestDataset([inputs], [targets], n_example_points=100)
        error = 0
        for batch in dataset:
            error += test_eval(model, loss_fn, batch, device)
        error /= len(dataset)
        scene_errors[dataset_index] = error
    results[model_type] = scene_errors

# Plotting
x = np.arange(len(scene_labels))
# Desired order for x-axis
order = [0, 6, 7, 2, 5, 3, 4, 1]
scene_labels_ordered = [f"Scene {i}" for i in order]

x_vals = [0, 0.25, 0.5, 0.75, 0.812, 0.875, 0.939, 1]

fig = plt.figure(figsize=(3.5, 2.5))


plt.plot(
    x_vals,
    [results["neural_ode"][i] for i in order],
    marker="o",
    label="NODE",
    color="#D62728",
)
plt.plot(
    x_vals,
    [results["function_encoder"][i] for i in order],
    marker="^",
    label="FE-NODE",
    color="#1F77B4",
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

plt.savefig(f"scene_mean_errors_seed={seed}.png", bbox_inches="tight", dpi=300)
plt.close()
