import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# --- Config: Scene order, labels, and colors (original order preserved) ---
scene_info = [
    (1, '#1a1a1a', 'Scene 1'),  # darkest gray
    (4, '#404040', 'Scene 4'),
    (3, '#666666', 'Scene 3'),
    (5, '#7f7f7f', 'Scene 5'),
    (2, '#999999', 'Scene 2'),
    (7, '#b3b3b3', 'Scene 7'),
    (6, '#cccccc', 'Scene 6'),
    (0, '#e6e6e6', 'Scene 0'),  # lightest gray
]

# --- Load data ---
odom_dfs = []
cmdvel_dfs = []
path = 'terrain_adaptation_rls/data'
for scene_id, _, _ in scene_info:
    odom_dfs.append(pd.read_csv(f"{path}/scene{scene_id}_odom.csv"))
    cmdvel_dfs.append(pd.read_csv(f"{path}/scene{scene_id}_cmd_vel.csv"))

# --- Load real terrain CSVs ---
real_colors = {
    'grass': '#009E73',
    'gravel': '#E69F00',
    'mulch': '#0072B2',
    'ahg-gym': "#FF0000",
}
real_odom_dfs = []  # list of (df, color, label)
for terrain in ['grass', 'gravel', 'mulch','ahg-gym',]:
    for file in sorted(glob.glob(os.path.join(path, f"real_{terrain}_odom.csv"))):
        df = pd.read_csv(file)
        label = f"Real {terrain.capitalize()}"
        real_odom_dfs.append((df, real_colors[terrain], label))

# --- Styling ---
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
marker_size = 0.1
lbl_fs = tick_fs = 8

# --- Plot odometry scatter plots ---
# limits for plotting just real data.
xLims = [-2,2]
yLims = [-1.5, 1.5]
angLims = [-3.5, 3.7]
# limits for plotting just sim data. 
xLims = [-3, 5.2]
yLims = [-4, 3.8]
angLims = [-2.8, 2.5]
# limits for plotting real and sim data. 
xLims = [-3, 5.2]
yLims = [-4, 3.8]
angLims = [-3.5, 3.7]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7.05, 2), sharex=False, sharey=False)
plot_specs = [
    ('xVel', 'yVel', r'$v_x$', r'$v_y$', xLims, yLims),
    ('xVel', 'zAngVel', r'$v_x$', r'$\omega_z$', xLims, angLims),
    ('zAngVel', 'yVel', r'$\omega_z$', r'$v_y$', angLims, yLims),
]

for ax, (xk, yk, xlabel, ylabel, xlim, ylim) in zip(axes, plot_specs):
    # Simulated scenes
    for df, (_, color, label) in zip(odom_dfs, scene_info):
        ax.scatter(df[xk], df[yk], s=marker_size, color=color, marker='o', label=label if (xk, yk) == ('xVel', 'yVel') else None)
    # Real terrain data
    for df, color, label in real_odom_dfs:
        ax.scatter(df[xk], df[yk], s=marker_size*0.1, color=color, marker='o', label=label if (xk, yk) == ('xVel', 'yVel') else None)
    
    ax.set_xlabel(xlabel, fontsize=lbl_fs)
    ax.set_ylabel(ylabel, fontsize=lbl_fs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=tick_fs)
    ax.grid(False)

# --- Show only real scene labels in the legend ---
handles, labels = axes[0].get_legend_handles_labels()

# Filter to only include labels with "Real"
real_handles_labels = [(h, l) for h, l in zip(handles, labels) if "Real" in l]

# Unzip into separate lists
real_handles, real_labels = zip(*real_handles_labels)

fig.legend(
    real_handles,
    real_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.05),
    fontsize=8,
    ncol=len(real_labels),
    frameon=False,
    markerscale=20
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
# fig.savefig(f"{path}/velocity_scatter_with_real_and_better_control.png", dpi=300, bbox_inches='tight')
# plt.close(fig)
plt.show()

# --- Plot cmd_vel scatter ---
fig, ax = plt.subplots(figsize=(2.5, 2.5))

# Simulated scenes
for df, (_, color, label) in zip(cmdvel_dfs, scene_info):
    df_sampled = df.sample(frac=0.25, random_state=42)
    ax.scatter(df_sampled['linear.x'], df_sampled['angular.z'],
               s=marker_size, color=color, marker='o', label=label)

# Real terrains
for terrain in ['ahg-gym', 'grass', 'gravel', 'mulch']:
    real_cmd_path = os.path.join(path, f"real_{terrain}_cmd_vel.csv")
    if os.path.exists(real_cmd_path):
        df_real = pd.read_csv(real_cmd_path)
        df_sampled = df_real.sample(frac=0.25, random_state=42)
        ax.scatter(df_sampled['linear.x'], df_sampled['angular.z'],
                   s=marker_size, color=real_colors[terrain], marker='o',
                   label=f"Real {terrain.capitalize()}")

ax.set_xlabel(r'Command $v_x$', fontsize=lbl_fs)
ax.set_ylabel(r'Command $\omega_z$', fontsize=lbl_fs)
ax.tick_params(labelsize=tick_fs)
ax.grid(False)

# Legend to the right
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles[::-1],
    labels[::-1],
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    fontsize=8,
    frameon=False,
    markerscale=20,
    ncol=1
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
# fig.savefig(f"{path}/cmd_vel_with_real.png", dpi=300, bbox_inches='tight')
# plt.close(fig)
plt.show()
