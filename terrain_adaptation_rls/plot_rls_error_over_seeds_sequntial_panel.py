import csv
import os
import matplotlib.pyplot as plt

# Config
model_types = ["function_encoder", "neural_ode", "rls", "maml"]  
platform = 'jackal_0770'

# Four short bags
scenes = [
    "short_bags/grass",
    "short_bags/gym_floor",
    "short_bags/mulch",
    "short_bags/ice",
]

# Plotting style
plt.rcParams.update({
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

colors = {
    "neural_ode": "#D62728",
    "function_encoder": "#1F77B4",
    "rls": "#2ca02c",
    "maml": "#A200FF",
}
names = {
    "neural_ode": "NODE",
    "function_encoder": "FE",
    "rls": "FE-RLS",
    "maml": "MAML",
}
scene_names = {
    "short_bags/grass": "Grass",
    "short_bags/gym_floor": "Gym Floor",
    "short_bags/mulch": "Mulch",
    "short_bags/ice": "Ice",
}

# Create a 2x2 panel figure
fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.0), sharex=True)
axs = axs.flatten()

for i, (ax, scene) in enumerate(zip(axs, scenes)):
    csv_path = f"plots/{platform}/accum_single_step_errors_over_full_scenes/{scene}"

    for mt in model_types:
        csv_file = os.path.join(csv_path, f"{mt}_errors.csv")

        # Load CSV
        times, med, p10, p90 = [], [], [], []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row["timestep"]))
                med.append(float(row["median"]))
                p10.append(float(row["p10"]))
                p90.append(float(row["p90"]))
                if times[-1] > 25:
                    break

        # Plot median + shaded region
        ax.plot(times, med, label=names[mt], color=colors[mt])
        ax.fill_between(
            times,
            p10,
            p90,
            alpha=0.2,
            color=colors[mt],
            edgecolor="none",
            linewidth=0.0,
        )

    # Add in-plot title (scene name, simplified)
    ax.text(
        0.05, 0.93, scene_names[scene],
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8
    )

# One shared x/y axis label for the figure
fig.supxlabel("Time (s)", fontsize=8)
fig.supylabel("Accumulated Prediction Error", fontsize=8)

# Shared legend outside
fig.legend(
    handles=[plt.Line2D([0], [0], color=colors[mt], label=names[mt]) for mt in model_types],
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=4,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
outdir = f"plots/{platform}/accum_single_step_errors_over_full_scenes"
os.makedirs(outdir, exist_ok=True)
plot_file = os.path.join(outdir, "four_panel_short_bags_25s_labeled_8pt_short.png")
plt.savefig(plot_file, bbox_inches="tight", dpi=300)
plt.close()
# plt.show()
