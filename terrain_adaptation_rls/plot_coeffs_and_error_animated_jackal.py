import argparse
import csv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import savgol_filter
from plot_utils import load_model, format_fig
from data.load_data import load_scenes, fullBagDataset


# -------------------------------
# Utility functions
# -------------------------------
def smooth_log(y, window_length=21, polyorder=3):
    """Apply Savitzky-Golay smoothing in log space."""
    y = np.log(y)
    w = min(window_length, len(y) - (len(y) + 1) % 2)  # ensure odd < len(y)
    return np.exp(savgol_filter(y, w, polyorder))


def load_errors(csv_file, tmin=95, tmax=140):
    """Load and trim errors from CSV between [tmin, tmax]."""
    ts, med, p10, p90 = [], [], [], []
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            t = float(row["timestep"])
            if t < tmin:
                continue
            if t > tmax:
                break
            ts.append(t)
            med.append(float(row["median"]))
            p10.append(float(row["p10"]))
            p90.append(float(row["p90"]))
    return np.array(ts), np.array(med), np.array(p10), np.array(p90)


# -------------------------------
# Animated combined plot
# -------------------------------
def animate_errors_and_coeffs(scene, platform="jackal_0770", n_basis=8, hidden_size=128, fps=8, seed=0):
    save_path = f"plots/{platform}/coefficients_over_time/{scene}"
    os.makedirs(save_path, exist_ok=True)

    # Prepare figure with subplots
    _, colors, _ = format_fig()
    fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=fig.get_size_inches(), dpi=fig.dpi, sharex=True, gridspec_kw={'height_ratios': [1,3]}
    )

    # -------------------------------
    # Load errors
    # -------------------------------
    csv_file = f"plots/{platform}/single_step_errors_over_full_scenes/{scene}/rls_errors.csv"
    t, med, p10, p90 = load_errors(csv_file, tmin=0, tmax=1e8)
    med, p10, p90 = map(smooth_log, (med, p10, p90))

    t0 = t[0]
    t -= t0

    # -------------------------------
    # Load coefficient norms
    # -------------------------------
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    fe_path = f"logs/{platform}/function_encoder/seed={seed}/hidden_size={hidden_size}/n_basis={n_basis}/function_encoder_model.pth"
    fe_model, _ = load_model("function_encoder", device, n_basis, fe_path, hidden_size)

    scene_data = load_scenes(["turf", "ice"], platform)
    ice_scene_input, ice_scene_target = scene_data["ice"]
    ice_dataset = fullBagDataset([ice_scene_input], [ice_scene_target], n_example_points=100)
    pave_scene_input, pave_scene_target = scene_data["turf"]
    pave_dataset = fullBagDataset([pave_scene_input], [pave_scene_target], n_example_points=100)

    def coeffs_for(scene_dataset):
        for batch in scene_dataset:
            if len(batch[0].shape) == 2:
                batch = [b.unsqueeze(0) for b in batch]
            _, _, _, ex_xs, ex_dt, ex_ys, _ = batch
            ex_xs, ex_dt, ex_ys = [t.to(device) for t in [ex_xs, ex_dt, ex_ys]]
            with torch.no_grad():
                coeffs, _ = fe_model.compute_coefficients((ex_xs, ex_dt), ex_ys)
            return coeffs

    pave_coeffs = coeffs_for(pave_dataset)
    ice_coeffs = coeffs_for(ice_dataset)

    coeff_file = f"plots/{platform}/coefficients_over_time/{scene}/rls_coeffs.csv"
    data = np.genfromtxt(coeff_file, delimiter=",", skip_header=1)
    time_array, coeffs = data[:, 0], data[:, 1:]
    rls_norms = np.linalg.norm(coeffs, axis=1)
    pave_norm = torch.norm(pave_coeffs).item()
    ice_norm = torch.norm(ice_coeffs).item()

    time_filtered = time_array - time_array[0]
    rls_norms_filtered = rls_norms
    rls_norms_filtered_smooth = smooth_log(rls_norms)

    # -------------------------------
    # Axis limits (fixed across frames)
    # -------------------------------
    max_t = max(t[-1], time_filtered[-1])
    ymin_err, ymax_err = np.min(p10), np.max(p90)
    ymin_coeff, ymax_coeff = np.min(rls_norms_filtered), np.max(rls_norms_filtered)


    # -------------------------------
    # Animation update function
    # -------------------------------
    def update(frame):
        ax1.clear()
        ax2.clear()

        # Errors subplot
        ax1.set_xlim(0, max_t)
        ax1.set_ylim(ymin_err, ymax_err)
        ax1.set_ylabel("Prediction Error", fontsize=16)
        ax1.tick_params(axis="both", which="major", labelsize=16)
        ax1.plot(t[:frame], med[:frame], label="RLS", color=colors["rls"])
        ax1.fill_between(t[:frame], p10[:frame], p90[:frame], alpha=0.2, color=colors["rls"])

        
        # Coefficient subplot
        ax2.set_xlim(0, max_t)
        ax2.set_ylim(ymin_coeff, ymax_coeff)
        ax2.set_xlabel("Time (s)", fontsize=16)
        ax2.set_ylabel("Coefficient Norm", fontsize=16)
        ax2.tick_params(axis="both", which="major", labelsize=16)
        ax2.hlines(pave_norm, xmin=0, xmax=max_t - time_array[0], 
                   linestyles="dashed", color='r', label="FE-Turf")
        ax2.hlines(ice_norm, xmin=0 - time_array[0], xmax=max_t,
                   linestyles="dashed", color='k', label="FE-Ice")
        ax2.plot(time_filtered[:frame], rls_norms_filtered[:frame], alpha=0.5, color="C0")
        ax2.plot(time_filtered[:frame], rls_norms_filtered_smooth[:frame], color="C0")

        return ax1, ax2

    # -------------------------------
    # Run animation
    # -------------------------------
    n_frames = max(len(t), len(time_filtered))
    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    mp4_file = os.path.join(save_path, f"rls_coeffs_and_error.mp4")
    writer = FFMpegWriter(fps=fps)
    ani.save(mp4_file, writer=writer, dpi=300)

    print(f"Saved animation to {mp4_file}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="ice_autonomy_11")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for MP4")
    args = parser.parse_args()

    animate_errors_and_coeffs(args.scene, fps=args.fps)
