import argparse
import csv
import os
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import savgol_filter
from plot_utils import format_fig

# Config
model_types = ["maml", "rls"]
platform = 'jackal_0770'

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default='ice_autonomy_11')
parser.add_argument("--fps", type=int, default=8, help="Frames per second for MP4")
args = parser.parse_args()

# Plotting setup
fig, colors, names = format_fig()
fig.subplots_adjust(top=0.90, bottom=0.25, left=0.2)
save_path = f"plots/{platform}/single_step_errors_over_full_scenes/{args.scene}"

def smooth_log(y, window_length=31, polyorder=3):
    y_log = np.log(y)
    y_smooth_log = savgol_filter(y_log, window_length, polyorder)
    return np.exp(y_smooth_log)

# Load all model data into memory first
all_data = {}
for mt in model_types:
    csv_file = os.path.join(save_path, f"{mt}_errors.csv")
    timesteps, med, p10, p90 = [], [], [], []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps.append(float(row["timestep"]))
            med.append(float(row["median"]))
            p10.append(float(row["p10"]))
            p90.append(float(row["p90"]))

    # Smooth
    window_length = min(21, len(timesteps) - (len(timesteps) + 1) % 2)
    med = smooth_log(med, window_length, 3)
    p10 = smooth_log(p10, window_length, 3)
    p90 = smooth_log(p90, window_length, 3)

    all_data[mt] = {
        "t": np.array(timesteps),
        "med": np.array(med),
        "p10": np.array(p10),
        "p90": np.array(p90),
    }

# Set up figure
ax = fig.add_subplot(111)
lines = {}
fills = {}

for mt in model_types:
    (line,) = ax.plot([], [], label=names[mt], color=colors[mt])
    fill = ax.fill_between([], [], [], alpha=0.2, color=colors[mt])
    lines[mt] = line
    fills[mt] = fill

ax.set_yscale("log")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Prediction Error Magnitude")

fig.legend(
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=4,
    frameon=False,
)

# Precompute axis limits
max_t = max(np.max(d["t"]) for d in all_data.values())
ymin = min(np.min(d["p10"]) for d in all_data.values())
ymax = max(np.max(d["p90"]) for d in all_data.values())


# Animation update function
def update(frame):
    ax.clear()
    ax.set_xlim(0, max_t)  # full time range fixed
    ax.set_ylim(ymin, ymax)  # fixed vertical range
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Prediction Error Magnitude")

    for mt in model_types:
        t = all_data[mt]["t"][:frame]
        med = all_data[mt]["med"][:frame]
        p10 = all_data[mt]["p10"][:frame]
        p90 = all_data[mt]["p90"][:frame]

        ax.plot(t, med, label=names[mt], color=colors[mt])
        ax.fill_between(t, p10, p90, alpha=0.2, color=colors[mt])

    return ax

# Number of frames = number of timesteps
n_frames = max(len(d["t"]) for d in all_data.values())

ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/args.fps, blit=False)

# Save to MP4
mp4_file = os.path.join(save_path, f"plot.mp4")
writer = FFMpegWriter(fps=args.fps, metadata={"artist": "ICRA Hero Plot"})
ani.save(mp4_file, writer=writer, dpi=300)

print(f"Saved animation to {mp4_file}")
