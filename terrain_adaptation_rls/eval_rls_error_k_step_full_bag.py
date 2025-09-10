import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.load_data import load_scenes, MultiRolloutFullBagDataset
from train_utils import inertial_to_body, inertial_to_body_XL
from plot_utils import load_model, format_fig
from torch.utils.data import DataLoader
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
k_steps = 15 # 30
n_rollouts = -1
batchsize = 1 # 100
torch.manual_seed(30)
seeds = list(range(1))
model_types = ["function_encoder", "neural_ode"]  

# Meta-learning hyperparameters
inner_lr = 1e-2
inner_steps = 1

# Choose the evaluation scene
platform = 'jackal_0770'
scene = 'short_bags/grass'
scene_data = load_scenes([scene], platform)
training_set = 'grass_gym_ice_mulch_pavement_turf'

# Create a dataset for testing prediction errors. 
scene_input, scene_target = scene_data[scene]
dataset = MultiRolloutFullBagDataset( # use MultiRolloutFullBagDataset for full bag
    inputs=[scene_input],
    targets=[scene_target],
    n_example_points=100,
    k_steps=k_steps,
    # n_rollouts=n_rollouts,
)
dataloader = DataLoader(dataset, batch_size=batchsize)

# Evaluate
all_results = {mt: {seed: [] for seed in seeds} for mt in model_types}
rls_results = {seed: [] for seed in seeds}
adaptive_results = {mt: {seed: [] for seed in seeds} for mt in ['rls', 'maml']}


with torch.no_grad():
    for seed in seeds:
        
        # Get a batch of data.
        x0_seq, dt_seq, u_seq, y_seq, ex_xs, ex_dt, ex_ys = next(iter(dataloader))
        x0_seq, dt_seq, u_seq, y_seq = [t.to(device) for t in [x0_seq, dt_seq, u_seq, y_seq]]
        ex_xs, ex_dt, ex_ys = [t.to(device) for t in [ex_xs, ex_dt, ex_ys]]

        # Get the number of time steps
        n_timesteps = x0_seq.shape[1]
        print(n_timesteps)

        # Evaluate the baseline FE and NODE on the batched data.
        for mt in model_types:
            # Load the model. 
            model_path = f"logs/{platform}/{training_set}/{mt}/seed={seed}/n_basis={n_basis}/hidden_size={hidden_size}/{mt}_model.pth"
            if not os.path.exists(model_path):
                print(f"Missing: {model_path}")
                exit()
            model, loss_fn = load_model(mt, device, n_basis, model_path, hidden_size)

            if mt == "function_encoder":
                # Compute the basis coefficients
                coefficients, _ = model.compute_coefficients((ex_xs, ex_dt), ex_ys)
            
            # Make a copy of the current state for processing.
            _x = x0_seq.clone()
            total_error = torch.zeros((batchsize, n_timesteps), device=device)

            # Loop over the k steps and predict the next state.
            for k in range(k_steps):

                # Predict the next state and save the prediction. 
                if mt == "function_encoder":
                    del_x = model((torch.cat((_x, u_seq[:,:,k,:]), dim=2), dt_seq[:,:,k]), coefficients=coefficients)
                elif mt == "neural_ode":
                    del_x = model((torch.cat((_x, u_seq[:,:,k,:]), dim=2), dt_seq[:,:,k]))

                # Get the next velocity in the initial body frame.
                next_vel_Bi = _x[:,:,3:6] + del_x[:,:,3:6]

                # Transform the velocity back to the body frame.
                next_vel_B = inertial_to_body_XL(
                    bIMat=del_x[:,:,:3],
                    xIMat=next_vel_Bi,
                    device=device
                )

                # Prepare the new current state. 
                _x = torch.cat((torch.zeros((batchsize, n_timesteps, 3), device=device), next_vel_B), dim=-1)
                
                # Calculate and accumulate the error. 
                pred = torch.cat((del_x[:,:,:3], next_vel_Bi), dim=-1)
                total_error += torch.norm(y_seq[:,:,k,:] - pred, dim=-1)

            # Save results from this model. 
            all_results[mt][seed] = total_error.cpu().numpy()

        
        # Initialize the RLS problem.
        model_path = f"logs/{platform}/{training_set}/function_encoder/seed={seed}/n_basis={n_basis}/hidden_size={hidden_size}/function_encoder_model.pth"
        rls_model, _ = load_model("function_encoder", device, n_basis, model_path, hidden_size)
        P = torch.eye(n_basis, device=device).repeat(1, 1, 1)
        coeffs = torch.zeros(batchsize, n_basis, device=device)
        rls_error = torch.zeros((batchsize, n_timesteps), device=device)

        for i in range(n_timesteps):

            print(f"seed={seed}, rollout={i}")

            # Get the next point in the RLS update data.
            x_step = x0_seq[:, i, :].unsqueeze(1)
            u_step = u_seq[:, i, 0, :].unsqueeze(1)
            dt_step = dt_seq[:, i, 0].unsqueeze(1)
            y_step = y_seq[:, i, 0, :].unsqueeze(1) - x_step

            # Compute new coeffs using RLS update. 
            g = rls_model.basis_functions((torch.cat((x_step, u_step), dim=-1), dt_step))
            L = torch.linalg.cholesky(P)
            coeffs, P = recursive_least_squares_update(
                method='qr', g=g, y=y_step, P=L, coefficients=coeffs, forgetting_factor=0.95
            )

            _x = x0_seq[:,i,:].clone()
            for k in range(k_steps):

                # Predict the next state and save the prediction. 
                del_x = rls_model((torch.cat((_x.unsqueeze(1), u_seq[:,i,k,:].unsqueeze(1)), dim=-1),
                                   dt_seq[:,i,k].unsqueeze(1)), coefficients=coeffs)

                # Get the next velocity in the initial body frame.
                next_vel_Bi = _x[:,3:6] + del_x[:,:,3:6].squeeze(1)

                # Transform the velocity back to the body frame.
                next_vel_B = inertial_to_body(
                    bIMat=del_x[:,:,:3].squeeze(1),
                    xIMat=next_vel_Bi,
                    device=device
                )

                # Prepare the new current state. 
                _x = torch.cat((torch.zeros((batchsize, 3), device=device), next_vel_B), dim=-1)

                # Calculate and accumulate the error. 
                pred = torch.cat((del_x[:,:,:3].squeeze(1), next_vel_Bi), dim=-1)
                # rls_error[:,i] += torch.nn.functional.mse_loss(pred, y_seq[:,i,k,:])
                rls_error[:,i] += torch.norm(y_seq[:,i,k,:] - pred, dim=-1)

        # Save the results from this rollout.
        rls_results[seed] = rls_error.cpu().numpy()




# Plotting
fig, colors, names = format_fig()
save_path = f"plots/{platform}/{training_set}/rls_error_k_step/{scene}_k={k_steps}_bs={batchsize}"
os.makedirs(save_path, exist_ok=True)

for mt in model_types + ["rls"]:
    # Collect the final accumulated errors from all seeds and all rollouts
    if mt == "rls":
        errors = np.concatenate([rls_results[seed] for seed in seeds], axis=0)
    else:
        errors = np.concatenate([all_results[mt][seed] for seed in seeds], axis=0)

    # Compute statistics
    med = np.median(errors, axis=0)
    p10 = np.percentile(errors, 10, axis=0)
    p90 = np.percentile(errors, 90, axis=0)

    # Plot median, min, and max. 
    plt.plot(med, label=names[mt], color=colors[mt])
    plt.fill_between(
        np.arange(n_timesteps),
        p10,
        p90,
        alpha=0.2,
        color=colors[mt],
        edgecolor="none",
        linewidth=0.0,
    )

    # Save data to a CSV file.
    # csv_file = os.path.join(save_path, f"{mt}_errors.csv")
    # with open(csv_file, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["timestep", "median", "p10", "p90"])
    #     for t, m, lo, hi in zip(np.arange(n_timesteps), med, p10, p90):
    #         writer.writerow([t, m, lo, hi])

plt.yscale("log")
plt.xlabel("Number of Time Steps")
plt.ylabel(f"Accumulated Rollout Error")
fig.legend(
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=3,
    frameon=False,
)
plt.tight_layout()

# Save the plot
plot_file = os.path.join(save_path, f"plot.png")
# plt.savefig(plot_file, bbox_inches="tight", dpi=300)
# plt.close()
plt.show()