import csv
import os
import torch
import numpy as np
import pandas as pd
from plot_utils import load_model, format_fig
from data.load_data import load_scenes, fullBagDataset, fullBagDatasetOnline
from function_encoder.coefficients import recursive_least_squares_update
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth_log(y, window_length=31, polyorder=3):
    y_smooth = savgol_filter(y, window_length, polyorder)
    return y_smooth

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Config
n_basis = 8
hidden_size = 128
torch.manual_seed(30)
model_types = ["rls"]  

# Choose the evaluation scene
pavement_scene = 'turf'
ice_scene = 'ice'
platform = 'jackal_0770'
scene_data = load_scenes([pavement_scene, ice_scene], platform)
training_set = 'grass_gym_ice_mulch_pavement_turf'

# Create a dataset for testing prediction errors. 
ice_scene_input, ice_scene_target = scene_data[ice_scene]
ice_dataset = fullBagDataset(  
    inputs=[ice_scene_input],
    targets=[ice_scene_target],
    n_example_points=100,
)
pave_scene_input, pave_scene_target = scene_data[pavement_scene]
pave_dataset = fullBagDataset(  
    inputs=[pave_scene_input],
    targets=[pave_scene_target],
    n_example_points=100,
)

# Load the FE model and compute coeffs for each scene. 
fe_path = f"logs/{platform}/{training_set}/function_encoder/seed=0/n_basis={n_basis}/hidden_size={hidden_size}/function_encoder_model.pth"
fe_model, fe_loss_fn = load_model("function_encoder", device, n_basis, fe_path, hidden_size)

# Evaluate the function encoder and neural ode models. 
for batch in ice_dataset:
    # Ensure data is properly shaped. 
    if len(batch[0].shape) == 2:
        batch = [b.unsqueeze(0) for b in batch]

    # Extract the data. 
    xs, dt, ys, ex_xs, ex_dt, ex_ys, times = batch
    xs, dt, ys, ex_xs, ex_dt, ex_ys, times = [t.to(device) for t in [xs, dt, ys, ex_xs, ex_dt, ex_ys, times]]
    
    with torch.no_grad():
        # Compute baseline fe coefficients.
        ice_coefficients, _ = fe_model.compute_coefficients((ex_xs, ex_dt), ex_ys)

for batch in pave_dataset:
    # Ensure data is properly shaped. 
    if len(batch[0].shape) == 2:
        batch = [b.unsqueeze(0) for b in batch]

    # Extract the data. 
    xs, dt, ys, ex_xs, ex_dt, ex_ys, times = batch
    xs, dt, ys, ex_xs, ex_dt, ex_ys, times = [t.to(device) for t in [xs, dt, ys, ex_xs, ex_dt, ex_ys, times]]
    
    with torch.no_grad():
        # Compute baseline fe coefficients.
        pave_coefficients, _ = fe_model.compute_coefficients((ex_xs, ex_dt), ex_ys)


# Load the RLS coeffs
scene = 'ice_autonomy/12'
save_path = f"plots/{platform}/{training_set}/coefficients_over_time/{scene}"
csv_file = os.path.join(save_path, f"rls_coeffs.csv")
data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)

# Split into time and coefficients
time_array = data[:, 0]       # shape (N,)
coeffs = data[:, 1:]    # shape (N, n_basis)


# Plot the coeffs over time.
# norms = torch.norm(rls_coeffs, dim=1, keepdim=True)
fig, ax = plt.subplots()
# for jj in range(n_basis):
jj = 4
# plot the coefficient trajectory
window_length = min(21, len(time_array) - (len(time_array) + 1) % 2)  # adaptive odd size
polyorder = 3  # cubic smoothing
smooth = smooth_log(coeffs[:, jj], window_length, polyorder)
line, = ax.plot(time_array, smooth, label=f"Value {jj}")

# get the same color as the curve
color = line.get_color()

# plot horizontal reference line in that color
ax.hlines(
    y=pave_coefficients[:, jj].detach().cpu().item(),  # ensure scalar
    xmin=time_array[0],
    xmax=160,
    linestyles="dashed",
    colors='k'
)

# plot horizontal reference line in that color
ax.hlines(
    y=ice_coefficients[:, jj].detach().cpu().item(),  # ensure scalar
    xmin=time_array[0],
    xmax=160,
    linestyles="dashed",
    colors=color
)
# plt.show()

ax.set_xlabel("Time")
ax.set_ylabel("Coefficient Value")
# ax.legend()
# plt.savefig(f'{save_path}/coeffs_over_time_smooth.png', bbox_inches="tight", dpi=300)
# plt.close()
plt.show()
# exit()



# Plot the norm of the coeffs over time. 
rls_norms = np.linalg.norm(coeffs, axis=1, keepdims=True)
ice_norm = torch.norm(ice_coefficients)
pave_norm = torch.norm(pave_coefficients)

# Filter time and norms
mask = (time_array >= 0)# & (time_array <= 140)
time_filtered = time_array[mask]
rls_norms_filtered = rls_norms[mask]

fig, ax = plt.subplots()

# plot the coefficient trajectory
# window_length = min(21, len(time_array) - (len(time_array) + 1) % 2)  # adaptive odd size
# polyorder = 3  # cubic smoothing
# smooth = smooth_log(coeffs[:, jj], window_length, polyorder)
line, = ax.plot(np.array(time_filtered)-time_filtered[0], rls_norms_filtered, label=f"Value {jj}")

# get the same color as the curve
color = line.get_color()

# plot horizontal reference line in that color
ax.hlines(
    y=pave_norm.detach().cpu().item(),  # ensure scalar
    xmin=0,
    xmax=130,
    linestyles="dashed",
    colors='k'
)

# plot horizontal reference line in that color
ax.hlines(
    y=ice_norm.detach().cpu().item(),  # ensure scalar
    xmin=0,
    xmax=130,
    linestyles="dashed",
    colors=color
)

ax.set_xlabel("Time")
ax.set_ylabel("Coefficient Norm")
# ax.legend()
# plt.savefig(f'{save_path}/normed_coeffs_over_time_trimmed.png', bbox_inches="tight", dpi=300)
# plt.close()
plt.show()