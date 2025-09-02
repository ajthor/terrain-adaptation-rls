import rosbag
import csv
import numpy as np

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
ssd_path = "/media/arl/ASG-3/8-Hardware-Datasets"
platform = "jackal_0770"
terrain = "gym-floor"
location = "AHG"
bag_name = "2025-08-22-11-48-58"
bag_file = f"{ssd_path}/{platform}/{terrain}-{location}/{bag_name}.bag" 

# Topics
odom_topic = f"/{platform}/odom"
cmd_topic = f"/{platform}/joy_teleop/cmd_vel"

# === Step 1: find earliest timestamp ===
earliest_time = None
with rosbag.Bag(bag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[odom_topic, cmd_topic]):
        t_float = t.to_sec()
        if earliest_time is None or t_float < earliest_time:
            earliest_time = t_float

print(f"Reference time (first message in bag): {earliest_time}")

# === Step 2: process messages relative to earliest timestamp ===
cmd_clean = []
odom_clean = []

with rosbag.Bag(bag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[odom_topic, cmd_topic]):
        time_sec = t.to_sec() - earliest_time

        if topic == cmd_topic:
            cmd_clean.append({
                "Time": time_sec,
                "linear.x": msg.linear.x,
                "linear.y": msg.linear.y,
                "linear.z": msg.linear.z,
                "angular.x": msg.angular.x,
                "angular.y": msg.angular.y,
                "angular.z": msg.angular.z
            })

        elif topic == odom_topic:
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            yaw = quaternion_to_yaw(qx, qy, qz, qw)

            odom_clean.append({
                "Time": time_sec,
                "time": time_sec,
                "xPos": msg.pose.pose.position.x,
                "yPos": msg.pose.pose.position.y,
                "yaw": yaw,
                "xVel": msg.twist.twist.linear.x,
                "yVel": msg.twist.twist.linear.y,
                "zAngVel": msg.twist.twist.angular.z
            })

# === Step 3: write CSVs ===
def write_csv(filepath, rows):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

data_path = f"/home/arl/terrain-adaptation-rls/terrain_adaptation_rls/data/{platform}"
write_csv(f"{data_path}/{terrain}_cmd_vel.csv", cmd_clean)
write_csv(f"{data_path}/{terrain}_odom.csv", odom_clean)

print(f"Wrote {len(cmd_clean)} cmd_vel rows and {len(odom_clean)} odom rows.")
