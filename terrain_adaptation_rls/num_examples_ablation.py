import os

import torch

from tqdm import trange

from data.load_data import load_all_scenes, OnlineTestDataset

from train_utils import test_eval
import argparse


args = argparse.ArgumentParser()
args.add_argument("--seed", type=int, default=42)
args.add_argument("--model", type=str, default="function_encoder")
args.add_argument("--n_basis", type=int, default=8)
args.add_argument("--device", type=str, default="cuda")
args.add_argument("--load_path", type=str)
args.add_argument("--n_example", type=str)
args = args.parse_args()

assert args.load_path is not None, "Please provide a path to the model to load."
assert os.path.exists(args.load_path), f"Path {args.load_path} does not exist."

# Load all scene data as a dictionary
scene_data = load_all_scenes()

# Split into train (80%) and test (20%)
inputs = []
targets = []
for scene_input, scene_target in scene_data.values():
    inputs.append(scene_input)
    targets.append(scene_target)


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
# Load model
model.load_state_dict(torch.load(args.load_path, map_location=args.device))

# test loop
losses = {}
if args.model == "neural_ode":
    n_examples = [1] # unused anyway
else:
    n_examples = [5, 10, 20, 50, 100]
for n_ex in n_examples:
    # create a dataset to iterate over in order to mimic an online setting
    online_dataset = OnlineTestDataset(inputs, targets, n_example_points=n_ex)

    # start the test loop
    loss = 0.0
    for timestep in trange(len(online_dataset)):
        # Get the next batch of data.
        batch = online_dataset[timestep]
        loss += test_eval(model, loss_fn, batch, args.device)
    loss /= len(online_dataset)
    losses[n_ex] = loss

# now save this data to the same dir as the model
save_dir = os.path.dirname(args.load_path)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "online_losses.pt")
torch.save(losses, save_path)