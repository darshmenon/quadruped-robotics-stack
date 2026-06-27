"""
Converts Go2 URDF to SDF with standing joint positions and
a joint position controller plugin so it holds the pose in Gazebo Garden.

Usage:
    python3 scripts/make_go2_stand.py
    # outputs /tmp/go2_stand.sdf
"""

import subprocess
import xml.etree.ElementTree as ET
import sys
import os
import argparse
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.join(REPO, "urdf", "go2_unitree", "urdf", "go2.urdf")
DEFAULT_OUT_SDF = os.path.join(tempfile.gettempdir(), "go2_stand.sdf")

# Go2 standing joint angles (radians)
# Hip (abduction): 0, Thigh (hip flex): 0.8, Calf (knee): -1.5
STANDING_POSE = {
    "FL_hip_joint":   0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint":  -1.5,
    "FR_hip_joint":   0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint":  -1.5,
    "RL_hip_joint":   0.0,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint":  -1.5,
    "RR_hip_joint":   0.0,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint":  -1.5,
}

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default=DEFAULT_URDF)
parser.add_argument("--out", default=DEFAULT_OUT_SDF)
args = parser.parse_args()

# Convert URDF -> SDF
print("Converting URDF to SDF...")
result = subprocess.run(["gz", "sdf", "-p", args.urdf], capture_output=True, text=True)
if result.returncode != 0:
    print("Error:", result.stderr)
    sys.exit(1)

sdf_text = result.stdout

# Parse and modify SDF
ET.register_namespace("", "")
tree = ET.ElementTree(ET.fromstring(sdf_text))
root = tree.getroot()
model = root.find("model")

# Add initial_position to each joint axis
for joint in model.findall("joint"):
    jname = joint.get("name")
    if jname in STANDING_POSE:
        axis = joint.find("axis")
        if axis is not None:
            ip = ET.SubElement(axis, "initial_position")
            ip.text = str(STANDING_POSE[jname])

# Add joint position controller plugin for each leg joint
for jname, angle in STANDING_POSE.items():
    plugin = ET.SubElement(model, "plugin")
    plugin.set("filename", "gz-sim-joint-position-controller-system")
    plugin.set("name", "gz::sim::systems::JointPositionController")
    ET.SubElement(plugin, "joint_name").text = jname
    ET.SubElement(plugin, "topic").text = f"/go2/cmd/{jname}"
    ET.SubElement(plugin, "p_gain").text = "200"
    ET.SubElement(plugin, "i_gain").text = "0.5"
    ET.SubElement(plugin, "d_gain").text = "10"
    ET.SubElement(plugin, "i_max").text = "1"
    ET.SubElement(plugin, "i_min").text = "-1"
    ET.SubElement(plugin, "cmd_max").text = "1000"
    ET.SubElement(plugin, "cmd_min").text = "-1000"
    ET.SubElement(plugin, "target").text = str(angle)
    ET.SubElement(plugin, "use_velocity_commands").text = "false"

imu_link = model.find("link[@name='imu']")
if imu_link is None:
    imu_link = model.find("link[@name='base']")
if imu_link is not None and imu_link.find("sensor[@name='imu_sensor']") is None:
    sensor = ET.SubElement(imu_link, "sensor")
    sensor.set("name", "imu_sensor")
    sensor.set("type", "imu")
    ET.SubElement(sensor, "always_on").text = "1"
    ET.SubElement(sensor, "update_rate").text = "100"
    ET.SubElement(sensor, "topic").text = "/imu/data"
    ET.SubElement(sensor, "imu")

if hasattr(ET, "indent"):
    ET.indent(tree, space="  ")
tree.write(args.out, encoding="unicode", xml_declaration=False)

print(f"Standing SDF written to: {args.out}")
print()
print("To spawn in Gazebo Garden:")
print("  gz sim -r empty.sdf &")
print("  sleep 4")
print(f"  gz service -s /world/empty/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 5000 --req \"sdf_filename: '{args.out}', name: 'go2', pose: {{position: {{z: 0.45}}}}\"")
