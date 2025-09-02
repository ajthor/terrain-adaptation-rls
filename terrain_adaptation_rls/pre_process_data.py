import csv
import os
from data.load_data import load_all_scenes, load_all_bluebonnet_scenes

# Write a torch tensor to a CSV file.
def write_csv(path, tensor_data):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in tensor_data.tolist():
            writer.writerow(row)

# Choose hardware or simulation.
platform = 'sim'

# Load all scene data as a dictionary
if platform == 'sim':
    scene_data = load_all_scenes()
    scenes = [0, 1, 2, 3, 4, 5, 6, 7]
elif platform == 'bluebonnet':
    scene_data = load_all_bluebonnet_scenes()
    scenes = ['ice1', 'ice2', 'ice4', 'ice5']

# Iterate over all scenes and save to CSVs.
for scene_index in scenes:

    # Extract the inputs and targets. 
    if platform == 'sim':
        scene_str = f"scene{scene_index}"
    elif platform == 'bluebonnet':
        scene_str = f"{scene_index}"

    scene_input, scene_target = scene_data[scene_str]

    # Save the data to CSV files
    if platform == 'sim':
        save_path = f"terrain_adaptation_rls/data_processed/scene{scene_index}"
    elif platform == 'bluebonnet':
        save_path = f"terrain_adaptation_rls/data_processed/bluebonnet_{scene_index}"
    os.makedirs(save_path, exist_ok=True)
    write_csv(f"{save_path}/input.csv", scene_input)
    write_csv(f"{save_path}/target.csv", scene_target)