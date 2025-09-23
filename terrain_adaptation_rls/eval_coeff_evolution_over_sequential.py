import csv
import os
import torch
import numpy as np
from plot_utils import load_model
from data.load_data import load_scenes, fullBagDataset
from function_encoder.coefficients import recursive_least_squares_update
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
seed = 0

# Choose the evaluation scene
scene = 'scene0_to_scene1_mission'
platform = 'warty'
scene_data = load_scenes([scene], platform)

# Create a dataset for testing prediction errors. 
scene_input, scene_target = scene_data[f'{scene}']
dataset = fullBagDataset(  
    inputs=[scene_input],
    targets=[scene_target],
    n_example_points=100,
)

# Evaluate the function encoder and neural ode models. 
for batch in dataset:
    # Ensure data is properly shaped. 
    if len(batch[0].shape) == 2:
        batch = [b.unsqueeze(0) for b in batch]

    # Extract the data. 
    xs, dt, ys, ex_xs, ex_dt, ex_ys, times = batch
    xs, dt, ys, ex_xs, ex_dt, ex_ys, times = [t.to(device) for t in [xs, dt, ys, ex_xs, ex_dt, ex_ys, times]]
    num_steps = times.shape[1]
    
    # Load the models.
    fe_path = f"logs/{platform}/function_encoder/seed={seed}/hidden_size={hidden_size}/n_basis={n_basis}/function_encoder_model.pth"
    fe_model, fe_loss_fn = load_model("function_encoder", device, n_basis, fe_path, hidden_size)

    # Initialize the RLS problem. 
    P = torch.eye(n_basis, device=device).repeat(1, 1, 1)
    coeffs = torch.zeros(1, n_basis, device=device)
    rls_coeffs = torch.zeros((num_steps, n_basis), device=device)

    for i in range(num_steps):
        if i % 100 == 0:
            print(f"\tStep: {i}")

        # Get the next point in the RLS update data.
        x_step = xs[:, i, :].unsqueeze(0)
        dt_step = dt[:, i].unsqueeze(0)
        y_step = ys[:, i, :].unsqueeze(0)

        # Predict the next state and save the prediction. 
        with torch.no_grad():
            del_x = fe_model((x_step, dt_step), coefficients=coeffs)

        # Compute new coeffs using RLS update. 
        g = fe_model.basis_functions((x_step, dt_step))
        L = torch.linalg.cholesky(P)
        coeffs, P = recursive_least_squares_update(
            method='qr', g=g, y=y_step, P=L, coefficients=coeffs, forgetting_factor=0.95
        )

        rls_coeffs[i, :] = coeffs


# Save the data
save_path = f"plots/{platform}/coefficients_over_time/{scene}"
os.makedirs(save_path, exist_ok=True)

# Move to numpy
time_array = times[0].detach().cpu().numpy()      
coeffs_array = rls_coeffs.detach().cpu().numpy()    

# Combine time and coefficients
output_array = np.hstack([time_array[:, None], coeffs_array])

# Save to CSV
save_path = f"plots/{platform}/coefficients_over_time/{scene}"
os.makedirs(save_path, exist_ok=True)
save_path = os.path.join(save_path, "rls_coeffs.csv")

header = ["time"] + [f"c{i}" for i in range(n_basis)]

with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(output_array)

print(f"Saved coefficients to {save_path}")