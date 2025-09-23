import csv
import os

import torch

from matplotlib import pyplot as plt
import numpy as np

from data.load_data import PhoenixDataset, TestDataset, LocalizedDataset

import tqdm

from train_utils import train_step, test_eval
import argparse


# Load a CSV file and return the data as a tensor.
def load_csv(filepath):
    data = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)


# Parse command line arguments.
args = argparse.ArgumentParser()
args.add_argument("--seed", type=int, default=42)
args.add_argument("--model", type=str, default="neural_ode")
args.add_argument("--n_basis", type=int, default=8)
args.add_argument("--gradsteps", type=int, default=10000)
args.add_argument("--logdir", type=str, default="logs")
args.add_argument("--device", type=str, default="cuda")
args.add_argument("--window", type=int, default=100)
args = args.parse_args()

# Create a personal logdir.
args.logdir = f"{args.logdir}/trained_on_bluebonnet_with_localized_examples_filtered_5_window={args.window}_bs=32_basis={args.n_basis}_residuals/{args.model}/seed={args.seed}"

# Seed the model.
torch.manual_seed(args.seed)

# Choose the scenes for training and testing.
training_scenes = ['grass','mulch','ice1','ice4','ice5']
test_scenes = ['gravel', 'ice2']

# Load the scenes. 
load_path = f"terrain_adaptation_rls/data_processed"

# Define lists to hold data for all scenes.
train_inputs = []
train_targets = []
test_inputs = []
test_targets = []

# Load the training scenes into a list. 
for scene in training_scenes:
    # Pull data for each scene. 
    train_input = torch.tensor(load_csv(f"{load_path}/bluebonnet_{scene}/input.csv")).float()
    train_target = torch.tensor(load_csv(f"{load_path}/bluebonnet_{scene}/target.csv")).float()

    # Save the train and test data for training scenes.
    train_inputs.append(train_input)
    train_targets.append(train_target)

# Iterate over the test scenes.
for scene in test_scenes:
    # Pull data for each scene. 
    test_input = torch.tensor(load_csv(f"{load_path}/bluebonnet_{scene}/input.csv")).float()
    test_target = torch.tensor(load_csv(f"{load_path}/bluebonnet_{scene}/target.csv")).float()

    # Save the train and test data for training scenes.
    test_inputs.append(test_input)
    test_targets.append(test_target)

# create an iterable dataset for training
train_dataset = LocalizedDataset(
    train_inputs, train_targets, n_example_points=100, n_points=796, window=args.window
)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
train_dataloader_iter = iter(train_dataloader)

# create datasets for eval
test_dataset = TestDataset(
    test_inputs, test_targets, n_example_points=100
)


# Define model
match args.model:
    case "neural_ode":
        from models.neural_ode import create_model, save_model, loss_fn

        model = create_model(args.device, n_basis=args.n_basis).to(args.device)

    case "function_encoder":
        from models.function_encoder import create_model, save_model, loss_fn
        
        model = create_model(args.device, n_basis=args.n_basis).to(args.device)

    case _:
        raise ValueError(f"Unknown model type: {args.model_type}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
training_data = {
    "training_loss": [],
    "test_loss": [],
}
with tqdm.trange(args.gradsteps) as tqdm_bar:
    for grad_step in tqdm_bar:
        # Get the next batch of data.
        batch = next(train_dataloader_iter)
        train_loss = train_step(model, optimizer, loss_fn, batch, args.device)
        training_data["training_loss"].append(train_loss)

        # Evaluate on all eval datasets
        test_loss = 0
        for batch in test_dataset:
            test_loss += test_eval(model, loss_fn, batch, args.device)
        test_loss /= len(test_dataset)
        training_data["test_loss"].append(test_loss)
        tqdm_bar.set_postfix_str(f"loss: {test_loss:.2e}")


# Save the model and data
os.makedirs(args.logdir, exist_ok=True)
torch.save(training_data, f"{args.logdir}/training_data.pth")
save_model(model, f"{args.logdir}/{args.model}_model.pth")

# create a training curve plot just to visualize
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
axs[0].plot(training_data["training_loss"])
axs[0].set_title("Training Loss")
axs[1].plot(training_data["test_loss"])
axs[1].set_title("Test Loss")
plt.tight_layout()
plt.savefig(f"{args.logdir}/training_curve.png")
