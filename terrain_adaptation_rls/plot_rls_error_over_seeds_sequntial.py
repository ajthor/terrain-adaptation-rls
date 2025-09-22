import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import format_fig
from scipy.signal import savgol_filter


# Config
model_types = ["neural_ode", "maml", "rls"]  
platform = 'warty'

# Parse command line arguments.
args = argparse.ArgumentParser()
args.add_argument("--scene", type=str, default='scene0_to_scene1')
args = args.parse_args()

# Plotting
fig, colors, names = format_fig()
save_path = f"plots/{platform}/single_step_errors_over_full_scenes/{args.scene}"

def smooth_log(y, window_length=31, polyorder=3):
    y_log = np.log(y)
    y_smooth_log = savgol_filter(y_log, window_length, polyorder)
    return np.exp(y_smooth_log)

for mt in model_types:

    csv_file = os.path.join(save_path, f"{mt}_errors.csv")

    # Load CSV
    timesteps, med, p10, p90 = [], [], [], []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # if float(row["timestep"]) < 95:
            #     continue
            timesteps.append(float(row["timestep"]))
            med.append(float(row["median"]))
            p10.append(float(row["p10"]))
            p90.append(float(row["p90"]))
            if timesteps[-1] > 330:
                break

    # Apply smoothing (window length must be odd and < len(timesteps))
    window_length = min(21, len(timesteps) - (len(timesteps) + 1) % 2)  # adaptive odd size
    polyorder = 3  # cubic smoothing
    med = smooth_log(med, window_length, polyorder)
    p10 = smooth_log(p10, window_length, polyorder)
    p90 = smooth_log(p90, window_length, polyorder)

    # Plot median, p10, and p90. 
    plt.plot(np.array(timesteps) , med, label=names[mt], color=colors[mt])
    plt.fill_between(
        np.array(timesteps),
        p10,
        p90,
        alpha=0.2,
        color=colors[mt],
        edgecolor="none",
        linewidth=0.0,
    )

# Plot terrain changes
file_path = f"terrain_adaptation_rls/data/{platform}/{args.scene}/triggers.csv"
with open(file_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # if float(row["Time"]) > 140:
        #     break
        plt.axvline(x=float(row["Time"]), color="gray", linestyle="--", linewidth=0.75)


# Add a text box (anchored to axes coordinates, so it stays in corner)
plt.text(
    0.17, 0.95, "Ice",
    transform=plt.gca().transAxes,
    fontsize=8,
    verticalalignment="top",
)
plt.text(
    0.335, 0.95, "Pavement",
    transform=plt.gca().transAxes,
    fontsize=8,
    verticalalignment="top",
)
plt.text(
    0.60, 0.95, "Ice",
    transform=plt.gca().transAxes,
    fontsize=8,
    verticalalignment="top",
)
plt.text(
    0.77, 0.95, "Pavement",
    transform=plt.gca().transAxes,
    fontsize=8,
    verticalalignment="top",
)


plt.yscale("log")
plt.xlabel("Time (s)")
plt.ylabel(f"Prediction Error Magnitude")
fig.legend(
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=4,
    frameon=False,
)
plt.tight_layout()

# Save the plot
plot_file = os.path.join(save_path, f"plot.png")
plt.savefig(plot_file, bbox_inches="tight", dpi=300)
plt.close()
# plt.show()