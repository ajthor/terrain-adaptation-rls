from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import csv
import numpy as np

"""Extracts and processes data from ROS 2 bags.
Outputs CSV files for odom and cmd_vel topics in
the format required by this FE training code."""

def quaternion_to_yaw(x, y, z, w):
    """
    Convert a quaternion (x, y, z, w) to yaw (rotation around Z axis) in radians.
    """
    # Formula: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

ssd_path = "/ssd_path/"
platform = "bluebonnet"
terrain = "ice"
location = "rink-the-crossover"
bag_name = "rosbag2_2025_08_08-16_17_55"
bag_file = f"{ssd_path}/{platform}/{terrain}-{location}/{bag_name}/{bag_name}_0.mcap" 
topic_types = {
    f"/bluebonnet/odometry/local": "nav_msgs/msg/Odometry",
    f"/bluebonnet/joy_teleop/cmd_vel": "geometry_msgs/msg/TwistStamped"
}

# === Step 1: find earliest timestamp ===
earliest_time = None
with open(bag_file, "rb") as f:
    reader = make_reader(f)
    for _, channel, message in reader.iter_messages():
        if channel.topic not in topic_types:
            continue
        msg_type = get_message(topic_types[channel.topic])
        msg = deserialize_message(message.data, msg_type)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if earliest_time is None or t < earliest_time:
            earliest_time = t

print(f"Reference time (first message in bag): {earliest_time}")

# === Step 2: process messages relative to earliest timestamp ===
cmd_clean = []
odom_clean = []

with open(bag_file, "rb") as f:
    reader = make_reader(f)
    for _, channel, message in reader.iter_messages():
        if channel.topic not in topic_types:
            continue
        msg_type = get_message(topic_types[channel.topic])
        msg = deserialize_message(message.data, msg_type)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        time_sec = t - earliest_time  # relative time

        if channel.topic == "/bluebonnet/joy_teleop/cmd_vel":
            cmd_clean.append({
                "Time": time_sec,
                "linear.x": msg.twist.linear.x,
                "linear.y": msg.twist.linear.y,
                "linear.z": msg.twist.linear.z,
                "angular.x": msg.twist.angular.x,
                "angular.y": msg.twist.angular.y,
                "angular.z": msg.twist.angular.z
            })
        elif channel.topic == "/bluebonnet/odometry/local":
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
