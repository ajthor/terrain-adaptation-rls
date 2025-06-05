
import torch
import os
import matplotlib.pyplot as plt



fe_load_path = "logs/function_encoder/seed=0/online_losses.pt"
node_load_path = "logs/neural_ode/seed=0/online_losses.pt"
assert os.path.exists(fe_load_path), f"Path {fe_load_path} does not exist."
assert os.path.exists(node_load_path), f"Path {node_load_path} does not exist."

# Load losses
fe_losses = torch.load(fe_load_path)
node_losses = torch.load(node_load_path)

fe_xs = fe_losses.keys()
fe_ys = [fe_losses[k] for k in fe_xs]
node_val = list(node_losses.values())[0] # it only has one since it doesnt use examples

# plot them
# set font size to 8
plt.rcParams.update({'font.size': 8})
sc = 1.0
fig, ax = plt.subplots(figsize=(3.5 * sc, 1.75 * sc), dpi=300)
ax.plot(fe_xs, fe_ys, label="Function Encoder", color="#1F77B4")
# plot node as a horizontal line
ax.axhline(y=node_val, color='#D62728', linestyle='--', label="Neural ODE")
ax.set_xlabel("Number of Example Points")
ax.set_ylabel("Online MSE")

# set x ticks to the values we have
ax.set_xticks(list(fe_xs))

# make y ticks more sparse
ax.set_yscale("log")

ax.set_yticks([2e-4, 3e-4])
ax.set_yticklabels(["2e-4", "3e-4"])

# legend
fig.legend(
    loc="outside upper center",
    bbox_to_anchor=(0.55, 1.07),
    ncol=2,
    frameon=False,
)

# give more space to the top of the box since the legend is cut off
plt.tight_layout()
plt.savefig("logs/online_losses.png", bbox_inches='tight')