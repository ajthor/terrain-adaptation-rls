import pandas as pd
import matplotlib.pyplot as plt

# === File paths for Scene 1 ===
path = "terrain_adaptation_rls/data/jackal_0770"
terrain = "ice"
odom_path = f"{path}/{terrain}/odom.csv"
cmd_path = f"{path}/{terrain}/cmd_vel.csv"

# === Load data ===
odom_df = pd.read_csv(odom_path)
cmd_df = pd.read_csv(cmd_path)

# === Sort by time (assumed in second column) ===
odom_df = odom_df.sort_values(by=odom_df.columns[1])
cmd_df = cmd_df.sort_values(by=cmd_df.columns[0])

# Extract time and relevant fields
t_odom = odom_df.iloc[:, 1]
xVel = odom_df["xVel"] #[:, 4] 
zAngVel = odom_df["zAngVel"] #[:,6] 

t_cmd = cmd_df.iloc[:, 0]
lin_cmd = cmd_df["linear.x"] #[:, 7] 
ang_cmd = cmd_df["angular.z"] #[:, 8] 

# === Styling ===
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
})

# === Plotting ===
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6.5, 4), sharex=False)

# --- Linear velocity ---
ax1.plot(t_cmd, lin_cmd, color='g', linestyle='--', label="cmd_vel")
ax1.plot(t_odom, xVel, color='k', linestyle='-')#, marker='o'), markersize='2', label="odom")
ax1.set_ylabel(r"Linear $v_x$ (m/s)")
ax1.set_title("Linear Velocity Over Time")
ax1.grid(True)
ax1.legend(loc="upper right", fontsize=8)

# --- Angular velocity ---
ax2.plot(t_cmd, ang_cmd, color='g', linestyle='--', label="cmd_vel")
ax2.plot(t_odom, zAngVel, color='k', linestyle='-')#, marker='o', markersize='2', label="odom")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel(r"Angular $\omega_z$ (rad/s)")
ax2.set_title("Angular Velocity Over Time")
ax2.grid(True)
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
# plt.savefig(f"{path}/{terrain}_over_time.png", bbox_inches="tight", dpi=300)
plt.show()