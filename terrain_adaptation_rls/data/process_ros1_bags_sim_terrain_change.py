import rosbag
import csv
import numpy as np
import os

"""Extracts and processes data from ROS 1 bags.
Outputs CSV files for odom and cmd_vel topics in
the format required by this FE training code."""

def quaternion_to_yaw(x, y, z, w):
    """
    Convert a quaternion (x, y, z, w) to yaw (rotation around Z axis) in radians.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

# === Paths ===
ssd_path = "path/to/bag/file"
platform = "warty"
terrain = "terrain"
bag_name = "bag_name"
bag_file = f"{ssd_path}/{bag_name}.bag" 

# Topics
odom_topic = f"/warty/odom_processed_full2D"
cmd_topic = f"/warty/cmd_vel"
trigger_topic = f"/trigger_terrain_change"

cmd_clean = []
odom_clean = []
triggers = []

with rosbag.Bag(bag_file, "r") as bag:

    # === Step 2: process messages relative to earliest timestamp ===
    for topic, msg, t in bag.read_messages(topics=[odom_topic, cmd_topic, trigger_topic]):

        if topic == cmd_topic:
            cmd_clean.append({
                "Time": t.to_sec(),
                "linear.x": msg.linear.x,
                "linear.y": msg.linear.y,
                "linear.z": msg.linear.z,
                "angular.x": msg.angular.x,
                "angular.y": msg.angular.y,
                "angular.z": msg.angular.z
            })

        elif topic == odom_topic:
            odom_clean.append({
                "Time": msg.time,
                "time": msg.time,
                "xPos": msg.xPos,
                "yPos": msg.yPos,
                "yaw": msg.yaw,
                "xVel": msg.xVel,
                "yVel": msg.yVel,
                "zAngVel": msg.zAngVel
            })
        
        elif topic == trigger_topic:
            triggers.append({
                "Time": t.to_sec(),
                "Scene": msg.data
            })

# === Step 3: write CSVs ===
def write_csv(filepath, rows):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

data_path = f"/home/arl/terrain-adaptation-rls/terrain_adaptation_rls/data/{platform}"
terrain_path = os.path.join(data_path, f'{terrain}/{bag_name}')
os.makedirs(terrain_path, exist_ok=True)
write_csv(os.path.join(terrain_path, "cmd_vel.csv"), cmd_clean)
write_csv(os.path.join(terrain_path, "odom.csv"), odom_clean)
write_csv(os.path.join(terrain_path, "triggers.csv"), triggers)

print(f"Wrote {len(cmd_clean)} cmd_vel rows and {len(odom_clean)} odom rows.")
print(f"Wrote {len(triggers)} trigger rows.")
