import os
import pandas as pd

"""Processes pre-split data to remove any data point 
in which the robot's velcoity exceeded physical limits
(i.e., glitches in the odometry data)."""

base_dir = "terrain_adaptation_rls/data_split"
seeds = [d for d in os.listdir(base_dir) if d.startswith("seed_")]

for seed in seeds:
    input_dir = os.path.join(base_dir, seed, "bluebonnet_ice4")
    output_dir = os.path.join(base_dir, seed, "bluebonnet_ice4_filtered")
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "test"]:
        # Paths for input/target files
        input_csv = os.path.join(input_dir, f"{split}_input.csv")
        target_csv = os.path.join(input_dir, f"{split}_target.csv")

        # Read CSVs
        input_df = pd.read_csv(input_csv, header=None)
        target_df = pd.read_csv(target_csv, header=None)

        # xVel = col index 4, yVel = col index 5 (0-based indexing)
        mask = (input_df.iloc[:, 4].between(-5, 5)) & (input_df.iloc[:, 5].between(-5, 5))

        # Apply mask to both
        filtered_input = input_df[mask].reset_index(drop=True)
        filtered_target = target_df[mask].reset_index(drop=True)

        # Save filtered versions
        filtered_input.to_csv(os.path.join(output_dir, f"{split}_input.csv"), index=False, header=False)
        filtered_target.to_csv(os.path.join(output_dir, f"{split}_target.csv"), index=False, header=False)

        print(f"[{seed}] {split}: removed {len(input_df) - len(filtered_input)} rows")

print("✅ Filtering complete")
