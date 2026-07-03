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

# Go2 standing joint angles (radians). These match the stable pose used by
# stand_go2_gz.py and the CHAMP adapter.
STANDING_POSE = {
    "FL_hip_joint":   0.10,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint":  -1.5,
    "FR_hip_joint":   -0.10,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint":  -1.5,
    "RL_hip_joint":   0.10,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint":  -1.5,
    "RR_hip_joint":   -0.10,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint":  -1.5,
}

FOOT_LINKS = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"}
CONTROLLER_PLUGIN = "gz::sim::systems::JointPositionController"

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

# go2_gz.urdf already carries JointPositionController plugins. Remove them
# before adding this script's tuned controllers, otherwise duplicate
# controllers fight over the same joints in Gazebo.
for plugin in list(model.findall("plugin")):
    if plugin.get("name") == CONTROLLER_PLUGIN:
        model.remove(plugin)


def ensure_child(parent, tag):
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def set_text(parent, tag, text):
    child = ensure_child(parent, tag)
    child.text = str(text)
    return child


def tune_collision_surface(collision):
    surface = ensure_child(collision, "surface")
    friction = ensure_child(surface, "friction")
    ode = ensure_child(friction, "ode")
    set_text(ode, "mu", "2.5")
    set_text(ode, "mu2", "2.5")
    set_text(ode, "fdir1", "1 0 0")

    contact = ensure_child(surface, "contact")
    contact_ode = ensure_child(contact, "ode")
    set_text(contact_ode, "kp", "100000")
    set_text(contact_ode, "kd", "1")
    set_text(contact_ode, "max_vel", "0.1")
    set_text(contact_ode, "min_depth", "0.001")


for link in model.findall("link"):
    for collision in link.findall("collision"):
        collision_name = collision.get("name", "")
        is_foot_link = link.get("name") in FOOT_LINKS
        is_lumped_foot = "foot_collision" in collision_name
        if is_foot_link or is_lumped_foot:
            tune_collision_surface(collision)

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
    ET.SubElement(plugin, "p_gain").text = "80"
    ET.SubElement(plugin, "i_gain").text = "0"
    ET.SubElement(plugin, "d_gain").text = "4"
    ET.SubElement(plugin, "i_max").text = "0"
    ET.SubElement(plugin, "i_min").text = "0"
    ET.SubElement(plugin, "cmd_max").text = "120"
    ET.SubElement(plugin, "cmd_min").text = "-120"
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

has_scan_sensor = any(
    sensor.findtext("topic", default="").strip().lstrip("/") == "scan"
    for sensor in model.findall(".//sensor")
)
base_link = model.find("link[@name='base']")
if base_link is not None and not has_scan_sensor:
    sensor = ET.SubElement(base_link, "sensor")
    sensor.set("name", "lidar_sensor")
    sensor.set("type", "gpu_lidar")
    ET.SubElement(sensor, "pose").text = "0.25 0 0.12 0 0 0"
    ET.SubElement(sensor, "always_on").text = "1"
    ET.SubElement(sensor, "update_rate").text = "10"
    ET.SubElement(sensor, "topic").text = "/scan"
    ET.SubElement(sensor, "gz_frame_id").text = "base"

    ray = ET.SubElement(sensor, "ray")
    scan = ET.SubElement(ray, "scan")
    horizontal = ET.SubElement(scan, "horizontal")
    ET.SubElement(horizontal, "samples").text = "720"
    ET.SubElement(horizontal, "resolution").text = "1"
    ET.SubElement(horizontal, "min_angle").text = "-3.14159"
    ET.SubElement(horizontal, "max_angle").text = "3.14159"

    range_tag = ET.SubElement(ray, "range")
    ET.SubElement(range_tag, "min").text = "0.08"
    ET.SubElement(range_tag, "max").text = "8.0"
    ET.SubElement(range_tag, "resolution").text = "0.01"

if hasattr(ET, "indent"):
    ET.indent(tree, space="  ")
tree.write(args.out, encoding="unicode", xml_declaration=False)

print(f"Standing SDF written to: {args.out}")
print()
print("To spawn in Gazebo Garden:")
print("  gz sim -r empty.sdf &")
print("  sleep 4")
print(f"  gz service -s /world/empty/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 5000 --req \"sdf_filename: '{args.out}', name: 'go2', pose: {{position: {{z: 0.32}}}}\"")
