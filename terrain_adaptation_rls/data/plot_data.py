import pandas as pd
import matplotlib.pyplot as plt
import os

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

# --- Config: Scene order, labels, and colors ---
warty_scene_info = [
    ('scene1', "#c447a4", 'Scene 1'), 
    ('scene4', "#56CAFF", 'Scene 4'),
    ('scene3', "#E5CE1E", 'Scene 3'),
    ('scene5', "#a63dd0", 'Scene 5'),
    ('scene2', "#EB0505", 'Scene 2'),
    ('scene7', "#11b00b", 'Scene 7'),
    ('scene6', "#ff8a24", 'Scene 6'),
    ('scene0', "#1f66ff", 'Scene 0'),  
]
jackal_0770_scene_info = [
    ('ice', "#00FBFF", 'Ice'),
    ('pavement', "#202020", 'Pavement'),
    ('turf', "#FF0000", 'Turf'),
    ('grass', "#009E73", 'Grass'),  
    ('mulch', "#7D3F00", 'Mulch'),
    ('gym_floor', "#FFB700", 'Gym Floor'),
]

# --- Load data ---
odom_dfs = []
cmdvel_dfs = []
platform = 'jackal_0770'
path = f'terrain_adaptation_rls/data/{platform}'

if platform == 'warty':
    scene_info = warty_scene_info
elif platform == 'jackal_0770':
    scene_info = jackal_0770_scene_info

for scene_id, _, _ in scene_info:
    odom_dfs.append(pd.read_csv(f"{path}/{scene_id}/odom.csv"))
    cmdvel_dfs.append(pd.read_csv(f"{path}/{scene_id}/cmd_vel.csv"))


# --- Plot odometry scatter plots --- 
if platform == 'warty':
    xLims = [-3, 5.2]
    yLims = [-4, 3.8]
    angLims = [-2.8, 2.5]
elif platform == 'jackal_0770':
    xLims = [-2, 2.5]
    yLims = [-2, 2]
    angLims = [-3.5, 3]
    

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
    
    ax.set_xlabel(xlabel, fontsize=lbl_fs)
    ax.set_ylabel(ylabel, fontsize=lbl_fs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=tick_fs)
    ax.grid(False)

# --- Create legend ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.05),
    fontsize=8,
    ncol=len(labels),
    frameon=False,
    markerscale=20
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(path, exist_ok=True)
fig.savefig(f"{path}/velocity_scatter.png", dpi=300, bbox_inches='tight')
plt.close(fig)
# plt.show()