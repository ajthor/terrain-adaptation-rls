import csv
import os
import torch
import numpy as np
from plot_utils import load_model, format_fig
from data.load_data import load_scenes, fullBagDataset, fullBagDatasetOnline
from function_encoder.coefficients import recursive_least_squares_update
from meta_learning.maml import adapt_model


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
seeds = list(range(5))
model_types = ["function_encoder", "neural_ode", "rls", "maml"]  

# Meta-learning hyperparameters
inner_lr = 1e-2
inner_steps = 5 # for sim and 1 for hardware

# Choose the evaluation scene
scene = 'scene0_to_scene1'
ex_scene = 'scene0'
platform = 'warty'
scene_data = load_scenes([scene], platform)
ex_scene_data = load_scenes([ex_scene], platform)

# Create a dataset for testing prediction errors. 
scene_input, scene_target = scene_data[f'{scene}']
ex_scene_input, ex_scene_target = ex_scene_data[ex_scene]
dataset = fullBagDatasetOnline( 
    inputs=[scene_input],
    targets=[scene_target],
    example_inputs=[ex_scene_input],
    example_targets=[ex_scene_target],
    n_example_points=100,
)

# Evaluate
all_results = {mt: {seed: [] for seed in seeds} for mt in model_types}

# Evaluate the function encoder and neural ode models. 
for batch in dataset:
    # Ensure data is properly shaped. 
    if len(batch[0].shape) == 2:
        batch = [b.unsqueeze(0) for b in batch]

    # Extract the data. 
    xs, dt, ys, ex_xs, ex_dt, ex_ys, times = batch
    xs, dt, ys, ex_xs, ex_dt, ex_ys, times = [t.to(device) for t in [xs, dt, ys, ex_xs, ex_dt, ex_ys, times]]
    num_steps = times.shape[1]

    for seed in seeds:
        print(f"Seed: {seed}")
        
        # Load the models.
        fe_path = f"logs/{platform}/function_encoder/seed={seed}/hidden_size={hidden_size}/n_basis={n_basis}/function_encoder_model.pth"
        node_path = f"logs/{platform}/neural_ode/seed={seed}/hidden_size={hidden_size}/n_basis={n_basis}/neural_ode_model.pth"
        fe_model, fe_loss_fn = load_model("function_encoder", device, n_basis, fe_path, hidden_size)
        node_model, node_loss_fn = load_model("neural_ode", device, n_basis, node_path, hidden_size)

        with torch.no_grad():
            # Compute baseline fe coefficients.
            coefficients, _ = fe_model.compute_coefficients((ex_xs, ex_dt), ex_ys)

            # Compute the FE and NODE prediction loss over the entire dataset.
            fe_pred = fe_model((xs, dt), coefficients=coefficients)
            all_results['function_encoder'][seed] = torch.norm(ys - fe_pred, dim=-1).cpu().numpy()

            node_pred = node_model((xs, dt))
            all_results['neural_ode'][seed] = torch.norm(ys - node_pred, dim=-1).cpu().numpy()


        # Initialize the RLS problem. 
        P = torch.eye(n_basis, device=device).repeat(1, 1, 1)
        coeffs = torch.zeros(1, n_basis, device=device)
        rls_error = torch.zeros((num_steps), device=device)

        # Initialize the MAML problem.
        model_path = f"logs/{platform}/maml/seed={seed}/hidden_size={hidden_size}/n_basis={n_basis}/maml_model.pth"
        maml_model, maml_loss_fn = load_model("maml", device, n_basis, model_path, hidden_size)
        maml_error = torch.zeros((num_steps), device=device)

        adapted_model = maml_model

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
                rls_error[i] = torch.norm(y_step - del_x, dim=-1).squeeze()


            del_x = adapted_model((x_step, dt_step))
            maml_error[i] = torch.norm(y_step - del_x, dim=-1).squeeze()

            # Compute new coeffs using RLS update. 
            g = fe_model.basis_functions((x_step, dt_step))
            L = torch.linalg.cholesky(P)
            coeffs, P = recursive_least_squares_update(
                method='qr', g=g, y=y_step, P=L, coefficients=coeffs, forgetting_factor=0.95
            )

            # Adapt the MAML model to the scene.
            example_data = (x_step, dt_step, y_step)
            adapted_model = adapt_model(
                model=adapted_model,
                example_data=example_data,
                loss_fn=maml_loss_fn,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
            )
                
        # Save the results from this seed.
        all_results['rls'][seed] = rls_error.unsqueeze(0).cpu().numpy()
        all_results['maml'][seed] = maml_error.unsqueeze(0).cpu().detach().numpy()



# Save the data
save_path = f"plots/{platform}/single_step_errors_over_full_scenes/{scene}"
accum_save_path = f"plots/{platform}/accum_single_step_errors_over_full_scenes/{scene}"
os.makedirs(save_path, exist_ok=True)
os.makedirs(accum_save_path, exist_ok=True)

for mt in model_types:
    # Collect the final accumulated errors from all seeds and all rollouts
    errors = np.concatenate([all_results[mt][seed] for seed in seeds], axis=0)

    # Calculate the accumulated errors from all seeds.
    accum_errors = np.cumsum(errors, axis=1)

    # Compute statistics
    time_array = times.cpu().numpy()[0,:]
    med = np.median(errors, axis=0)
    p10 = np.percentile(errors, 10, axis=0)
    p90 = np.percentile(errors, 90, axis=0)

    # Compute accumulated statistics
    accum_med = np.median(accum_errors, axis=0)
    accum_p10 = np.percentile(accum_errors, 10, axis=0)
    accum_p90 = np.percentile(accum_errors, 90, axis=0)

    # Save data to a CSV file.
    csv_file = os.path.join(save_path, f"{mt}_errors.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "median", "p10", "p90"])
        for t, m, lo, hi in zip(time_array, med, p10, p90):
            writer.writerow([t, m, lo, hi])

    # Save accumulated data to a CSV file.
    csv_file = os.path.join(accum_save_path, f"{mt}_errors.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "median", "p10", "p90"])
        for t, m, lo, hi in zip(time_array, accum_med, accum_p10, accum_p90):
            writer.writerow([t, m, lo, hi])
