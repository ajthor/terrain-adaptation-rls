import os

import numpy
import torch
import matplotlib.pyplot as plt

logdir = "logs"
algs = ["neural_ode", "function_encoder"]
colors = ["#D62728", "#1F77B4"]
names = ["Neural ODE", "Function Encoder"]

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

fig, axs = plt.subplots(1,3, figsize=(7.1, 2.25), sharey=True)

for alg, color, name in zip(algs, colors, names):
    load_dir = f"{logdir}/{alg}"
    subdirs = os.listdir(load_dir)

    # load all subdir training_data.pth
    train_losses, eval_losses, interpolation_losses, extrapolation_losses = [], [], [], []
    for subdir in subdirs:
        path = os.path.join(load_dir, subdir, "training_data.pth")
        data = torch.load(path)
        train_losses.append(data["training_loss"])
        eval_losses.append(data["eval_loss"])
        interpolation_losses.append(data["interpolation_loss"])
        extrapolation_losses.append(data["extrapolation_loss"])

    # plot the training data
    # train_losses = numpy.array(train_losses)
    # mins, maxs, median = numpy.min(train_losses, axis=0), numpy.max(train_losses, axis=0), numpy.median(train_losses, axis=0)
    # axs[0].plot(median, label=name, color=color)
    # axs[0].fill_between(range(len(mins)), mins, maxs, alpha=0.2, color=color)

    # plot the eval data
    eval_losses = numpy.array(eval_losses)
    mins, maxs, median = numpy.min(eval_losses, axis=0), numpy.max(eval_losses, axis=0), numpy.median(eval_losses, axis=0)
    axs[0].plot(median, label=name, color=color)
    axs[0].fill_between(range(len(mins)), mins, maxs, alpha=0.2, color=color)

    # plot the interpolation data
    interpolation_losses = numpy.array(interpolation_losses)
    mins, maxs, median = numpy.min(interpolation_losses, axis=0), numpy.max(interpolation_losses, axis=0), numpy.median(interpolation_losses, axis=0)
    axs[1].plot(median, label=name, color=color)
    axs[1].fill_between(range(len(mins)), mins, maxs, alpha=0.2, color=color)

    # plot the extrapolation data
    extrapolation_losses = numpy.array(extrapolation_losses)
    mins, maxs, median = numpy.min(extrapolation_losses, axis=0), numpy.max(extrapolation_losses, axis=0), numpy.median(extrapolation_losses, axis=0)
    axs[2].plot(median, label=name, color=color)
    axs[2].fill_between(range(len(mins)), mins, maxs, alpha=0.2, color=color)

# axs[0].set_title("Training Loss")
axs[0].set_title("Eval Loss")
axs[1].set_title("Interpolation Loss")
axs[2].set_title("Extrapolation Loss")

# axs[0].set_ylabel("MSE")
# axs[0].set_xlabel("Grad steps")
axs[0].set_xlabel("Grad steps")
axs[1].set_xlabel("Grad steps")
axs[2].set_xlabel("Grad steps")

# axs[0].set_yscale("log")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
axs[2].set_yscale("log")

# remove the legend labels except for the first 2
# Get handles and labels from the axes
handles, labels = axs[1].get_legend_handles_labels()

# Keep only the first two unique entries
seen = set()
filtered = []
for h, l in zip(handles, labels):
    if l not in seen:
        filtered.append((h, l))
        seen.add(l)
    if len(filtered) == 2:
        break

# Unzip handles and labels
handles, labels = zip(*filtered)

# Now plot the filtered legend
fig.legend(
    handles, labels,
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.07),
    ncol=2,
    frameon=False,
)


plt.tight_layout()
plt.savefig("logs/training_curves.png", dpi=300, bbox_inches='tight', )

