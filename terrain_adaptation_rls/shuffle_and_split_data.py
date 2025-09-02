import csv
import os
import torch
from data.load_data import load_all_sim_scenes, load_hardware_scenes

# Write a torch tensor to a CSV file.
def write_csv(path, tensor_data):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in tensor_data.tolist():
            writer.writerow(row)

# Choose hardware or simulation.
platform = 'bluebonnet'

# Load all scene data as a dictionary
if platform == 'warthog_sim':
    scene_data = load_all_sim_scenes()
    scenes = [0, 1, 2, 3, 4, 5, 6, 7]
else:
    scenes = ['grass', 'gravel', 'gym_floor', 'gym_floor_simple', 'ice1', 'ice2', 'ice4', 'ice5', 'mulch']
    scene_data = load_hardware_scenes(scenes, platform)

# Choose random seeds.
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]

# Create a directory to save the split data.
for seed in seeds:

    # Set the random seed. 
    torch.manual_seed(seed)

    # Iterate over all scenes and split into train and test.
    for scene_index in scenes:

        # Extract the inputs and targets. 
        if platform == 'warthog_sim':
            scene_str = f"scene{scene_index}"
        else:
            scene_str = scene_index

        scene_input, scene_target = scene_data[scene_str]

        # Generate random indices to split the data.
        total_points = scene_input.shape[0]
        indices = torch.randperm(total_points)

        # Get the split indices for 80% training and 20% testing.
        split_idx = int(0.8 * total_points)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        # Split the data into train and test sets.
        train_input = scene_input[train_indices]
        train_target = scene_target[train_indices]
        test_input = scene_input[test_indices]
        test_target = scene_target[test_indices]

        # Save the data to CSV files
        save_path = f"terrain_adaptation_rls/data_split/{platform}/seed_{seed}/{scene_str}"
        os.makedirs(save_path, exist_ok=True)
        write_csv(f"{save_path}/train_input.csv", train_input)
        write_csv(f"{save_path}/train_target.csv", train_target)
        write_csv(f"{save_path}/test_input.csv", test_input)
        write_csv(f"{save_path}/test_target.csv", test_target)