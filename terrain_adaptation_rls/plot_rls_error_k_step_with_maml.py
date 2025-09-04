import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.load_data import load_scenes, MultiRolloutDataset
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
k_steps = 30
n_rollouts = 100
batchsize = 100
torch.manual_seed(30)
seeds = list(range(10))
model_types = ["function_encoder", "neural_ode"]  

# Meta-learning hyperparameters
inner_lr = 1e-2
inner_steps = 5

# Choose the evaluation scene
platform = 'warthog_sim'
scene = 'scene5'
scene_data = load_scenes([scene], platform)

# Create a dataset for testing prediction errors. 
scene_input, scene_target = scene_data[scene]
dataset = MultiRolloutDataset(
    inputs=[scene_input],
    targets=[scene_target],
    n_example_points=100,
    k_steps=k_steps,
    n_rollouts=n_rollouts,
)
dataloader = DataLoader(dataset, batch_size=batchsize)

# Evaluate
all_results = {mt: {seed: [] for seed in seeds} for mt in model_types}
# rls_results = {seed: [] for seed in seeds}
adaptive_results = {mt: {seed: [] for seed in seeds} for mt in ['rls', 'maml']}


# with torch.no_grad():
for seed in seeds:
    
    # Get a batch of data.
    x0_seq, dt_seq, u_seq, y_seq, ex_xs, ex_dt, ex_ys = next(iter(dataloader))
    x0_seq, dt_seq, u_seq, y_seq = [t.to(device) for t in [x0_seq, dt_seq, u_seq, y_seq]]
    ex_xs, ex_dt, ex_ys = [t.to(device) for t in [ex_xs, ex_dt, ex_ys]]

    # Evaluate the baseline FE and NODE on the batched data.
    with torch.no_grad():
        for mt in model_types:
            # Load the model. 
            model_path = f"logs/{platform}/{mt}/seed={seed}/{mt}_model.pth"
            if not os.path.exists(model_path):
                print(f"Missing: {model_path}")
                exit()
            model, loss_fn = load_model(mt, device, n_basis, model_path)

            if mt == "function_encoder":
                # Compute the basis coefficients
                coefficients, _ = model.compute_coefficients((ex_xs, ex_dt), ex_ys)
            
            # Make a copy of the current state for processing.
            _x = x0_seq.clone()
            total_error = torch.zeros((batchsize, n_rollouts), device=device)

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
                _x = torch.cat((torch.zeros((batchsize, n_rollouts, 3), device=device), next_vel_B), dim=-1)
                
                # Calculate and accumulate the error. 
                pred = torch.cat((del_x[:,:,:3], next_vel_Bi), dim=-1)
                total_error += torch.norm(y_seq[:,:,k,:] - pred, dim=-1)

            # Save results from this model. 
            all_results[mt][seed] = total_error.cpu().numpy()

    
    # Initialize the RLS problem.
    model_path = f"logs/{platform}/function_encoder/seed={seed}/function_encoder_model.pth"
    rls_model, _ = load_model("function_encoder", device, n_basis, model_path)
    P = torch.eye(n_basis, device=device).repeat(1, 1, 1)
    coeffs = torch.zeros(1, n_basis, device=device)
    rls_error = torch.zeros(batchsize, n_rollouts, device=device)

    # Initialize the MAML problem.
    model_path = f"logs/{platform}/maml/seed={seed}/maml_model.pth"
    maml_model, maml_loss_fn = load_model("maml", device, n_basis, model_path)
    maml_error = torch.zeros(batchsize, n_rollouts, device=device)

    for j in range(batchsize):

        adapted_model = maml_model

        # Reset the FE RLS problem.
        P = torch.eye(n_basis, device=device).repeat(1, 1, 1)
        coeffs = torch.zeros(1, n_basis, device=device)

        for i in range(n_rollouts):

            print(f"seed={seed}, batch={j}, rollout={i}")

            # Get the next point in the RLS update data.
            x_step = x0_seq[j, i, :].unsqueeze(0).unsqueeze(1)
            u_step = u_seq[j, i, 0, :].unsqueeze(0).unsqueeze(1)
            dt_step = dt_seq[j, i, 0].unsqueeze(0).unsqueeze(1)
            y_step = y_seq[j, i, 0, :].unsqueeze(0).unsqueeze(1) - x_step

            # Compute new coeffs using RLS update. 
            g = rls_model.basis_functions((torch.cat((x_step, u_step), dim=-1), dt_step))
            L = torch.linalg.cholesky(P)
            coeffs, P = recursive_least_squares_update(
                method='qr', g=g, y=y_step, P=L, coefficients=coeffs, forgetting_factor=0.95
            )

            # Adapt the MAML model to the scene.
            example_data = (torch.cat((x_step, u_step), dim=-1), dt_step, y_step)
            adapted_model = adapt_model(
                model=adapted_model,
                example_data=example_data,
                loss_fn=maml_loss_fn,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
            )

            for mt in ['rls', 'maml']:

                # Reset the initial condition for each model. 
                _x = x0_seq[j,i,:].unsqueeze(0).clone()
            
                for k in range(k_steps):

                    # Predict the next state and save the prediction. 
                    if mt == 'rls':
                        with torch.no_grad():
                            del_x = rls_model((torch.cat((_x.unsqueeze(1), u_seq[j,i,k,:].unsqueeze(0).unsqueeze(1)), dim=-1),
                                                dt_seq[j,i,k].unsqueeze(0).unsqueeze(1)), coefficients=coeffs)
                    elif mt == 'maml':
                        del_x = adapted_model((torch.cat((_x.unsqueeze(1), u_seq[j,i,k,:].unsqueeze(0).unsqueeze(1)), dim=-1),
                                                dt_seq[j,i,k].unsqueeze(0).unsqueeze(1)))

                    # Get the next velocity in the initial body frame.
                    next_vel_Bi = _x[:,3:6] + del_x[:,:,3:6].squeeze(1)

                    # Transform the velocity back to the body frame.
                    next_vel_B = inertial_to_body(
                        bIMat=del_x[:,:,:3].squeeze(1),
                        xIMat=next_vel_Bi,
                        device=device
                    )

                    # Prepare the new current state. 
                    _x = torch.cat((torch.zeros((1, 3), device=device), next_vel_B), dim=-1)

                    # Calculate and accumulate the error. 
                    pred = torch.cat((del_x[:,:,:3].squeeze(1), next_vel_Bi), dim=-1)

                    if mt == 'rls':
                        rls_error[j, i] += torch.norm(y_seq[j,i,k,:].unsqueeze(0) - pred, dim=-1).squeeze(0)
                    elif mt == 'maml':
                        maml_error[j, i] += torch.norm(y_seq[j,i,k,:].unsqueeze(0) - pred, dim=-1).squeeze(0)

    # Save the results from this rollout.
    adaptive_results['rls'][seed] = rls_error.cpu().numpy()
    adaptive_results['maml'][seed] = maml_error.cpu().detach().numpy()




# Plotting
fig, colors, names = format_fig()

for mt in model_types + ["rls", "maml"]:
    # Collect the final accumulated errors from all seeds and all rollouts
    if mt == "rls":
        errors = np.concatenate([adaptive_results['rls'][seed] for seed in seeds], axis=0)
    elif mt == "maml":
        errors = np.concatenate([adaptive_results['maml'][seed] for seed in seeds], axis=0)
    else:
        errors = np.concatenate([all_results[mt][seed] for seed in seeds], axis=0)

    # Compute statistics
    med = np.median(errors, axis=0)
    _min = np.percentile(errors, 10, axis=0)
    _max = np.percentile(errors, 90, axis=0)

    # Plot median, min, and max. 
    plt.plot(med, label=names[mt], color=colors[mt])
    plt.fill_between(
        np.arange(n_rollouts),
        _min,
        _max,
        alpha=0.2,
        color=colors[mt],
        edgecolor="none",
        linewidth=0.0,
    )

plt.yscale("log")
plt.xlabel("Number of Time Steps")
plt.ylabel(f"Accumulated Rollout Error")
fig.legend(
    loc="outside upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=4,
    frameon=False,
)
plt.tight_layout()
plt.savefig(f"plots/warthog_sim/rls_error_k_step_with_maml/k={k_steps}_bs={batchsize}_{scene}_single_maml_update.png", bbox_inches="tight", dpi=300)
plt.close()
# plt.show()