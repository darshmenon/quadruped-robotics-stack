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
    # Manipulator arm (see ros2/champ_description/urdf/arm.urdf.xacro), raised
    # up and out so it's visible above the body instead of drooping under
    # gravity with no controller holding it.
    "base_joint":       0.0,
    "lower_arm_joint":  1.4,
    "upper_arm_joint":  0.8,
    "wrist1_joint":     0.3,
    "wrist2_joint":     0.0,
    # Gripper closed by default (see ros2/champ_description/urdf/arm.urdf.xacro:
    # 0.0 = closed, 0.025 = fully open for left_finger_joint, mirrored for
    # right_finger_joint since its <mimic> isn't enforced by this sim path).
    "left_finger_joint":  0.0,
    "right_finger_joint": 0.0,
}

FOOT_LINKS = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"}
CONTROLLER_PLUGIN = "gz::sim::systems::JointPositionController"

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default=DEFAULT_URDF)
parser.add_argument("--out", default=DEFAULT_OUT_SDF)
parser.add_argument("--mu", default="2.5", help="foot/ground friction (MuJoCo floor uses 1.0)")
parser.add_argument("--p-gain", default="300", help="JointPositionController p_gain (MuJoCo-trained policies use 20)")
parser.add_argument("--d-gain", default="30", help="JointPositionController d_gain (MuJoCo-trained policies use 0.5)")
args = parser.parse_args()

# Convert URDF -> SDF
print("Converting URDF to SDF...")
result = subprocess.run(["gz", "sdf", "-p", args.urdf], capture_output=True, text=True)
if result.returncode != 0:
    print("Error:", result.stderr)
    sys.exit(1)

sdf_text = result.stdout

# go2_gz.urdf.xacro can't use $(find training) to locate go2_ros2_control.yaml
# (training/ has no package.xml, so that's unresolvable within this repo) --
# it bakes in a __REPO_ROOT__ placeholder instead, resolved here to wherever
# this checkout actually lives.
sdf_text = sdf_text.replace("__REPO_ROOT__", REPO)

# Parse and modify SDF
ET.register_namespace("", "")
tree = ET.ElementTree(ET.fromstring(sdf_text))
root = tree.getroot()
model = root.find("model")

# A model without the arm (scripts/make_go2_noarm_urdf.py) has no arm joints: only pose/control what exists.
_present = {j.get("name") for j in model.findall("joint")}
STANDING_POSE = {k: v for k, v in STANDING_POSE.items() if k in _present}

# go2_gz.urdf already carries JointPositionController plugins. Remove them
# before adding this script's tuned controllers, otherwise duplicate
# controllers fight over the same joints in Gazebo.
for plugin in list(model.findall("plugin")):
    if plugin.get("name") == CONTROLLER_PLUGIN:
        model.remove(plugin)

# right_finger_joint's URDF <mimic> survives `gz sdf -p`'s URDF->SDF
# conversion into an <axis><mimic> block, even though the physics engine
# here can't honor it (startup log: "physics engine does not support mimic
# constraints, so no constraint will be created" -- right_finger_joint is
# driven independently by its own JointPositionController instead, see
# intelligence/manipulation/arm_reach_node.py). Empirically, leaving that
# dead <mimic> tag in place makes the JointPositionController fail to
# actually track its target for that joint (it gets stuck near its open
# limit instead of closing) -- stripping the tag fixes it.
for joint in model.findall("joint"):
    axis = joint.find("axis")
    if axis is not None:
        mimic = axis.find("mimic")
        if mimic is not None:
            axis.remove(mimic)


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
    set_text(ode, "mu", args.mu)
    set_text(ode, "mu2", args.mu)
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

# left_finger/right_finger (ros2/champ_description/urdf/arm.urdf.xacro) both
# have their prismatic joint's zero position at the same local (x=0.04,y=0)
# offset from gripper_base, so at anything near a "closed" commanded target
# their collision boxes are driving toward fully overlapping -- causing them
# to jam against EACH OTHER rather than an actual grasped object (confirmed
# empirically: with nothing between them, gripper close always stalls a few
# mm short, symmetric with an object absent or present). Distinct
# collide_bitmasks that share no bits disable finger-vs-finger collision
# while leaving each finger's default (all-bits) mask for colliding with
# everything else, so they can still grip a real object between them.
FINGER_COLLIDE_BITMASKS = {"left_finger": "0x01", "right_finger": "0x02"}
for link in model.findall("link"):
    bitmask = FINGER_COLLIDE_BITMASKS.get(link.get("name"))
    if bitmask is None:
        continue
    for collision in link.findall("collision"):
        surface = ensure_child(collision, "surface")
        contact = ensure_child(surface, "contact")
        set_text(contact, "collide_bitmask", bitmask)

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
    # p_gain/d_gain raised from 80/4: at 80 the legs were too compliant to hold
    # the commanded stance rigidly under the arm's added weight, so the base
    # slowly sank (0.32 -> ~0.25) and crept sideways at ~0.07-0.08 m/s even
    # with all IMU posture feedback disabled. 300/15 holds height near the
    # commanded 0.32 and roughly halves the steady-state creep (measured
    # ~0.036 m/s); it does not eliminate it -- that's a genuine stance-pose/
    # CoM equilibrium issue (see STANDING_POSE above), not a gain problem.
    ET.SubElement(plugin, "p_gain").text = args.p_gain
    ET.SubElement(plugin, "i_gain").text = "0"
    # d_gain raised further to 30 (vs. the 15 that matched p_gain's stand-pose
    # scaling above): these same per-joint controllers also track CHAMP's
    # swing/stance trajectory during walking, and 300/15 is stiff enough that
    # each foot's ground contact behaves like an underdamped spring -- visibly
    # a hop/bounce on touchdown rather than a settled stance. Doubling d_gain
    # alone (p_gain unchanged, so the stand-pose height/creep fix is kept)
    # raises the damping ratio and absorbs that contact energy instead of
    # bouncing it back.
    ET.SubElement(plugin, "d_gain").text = args.d_gain
    ET.SubElement(plugin, "i_max").text = "0"
    ET.SubElement(plugin, "i_min").text = "0"
    ET.SubElement(plugin, "cmd_max").text = "120"
    ET.SubElement(plugin, "cmd_min").text = "-120"
    ET.SubElement(plugin, "target").text = str(angle)
    ET.SubElement(plugin, "use_velocity_commands").text = "false"

# NOT adding gz::sim::systems::JointStatePublisher here: it would fix
# /joint_states never publishing (see arm_reach_node.py's docstring on the
# TF/joint-feedback gap), but empirically it causes a real leg-joint
# resonance with the JointPositionController plugins above -- e.g. FL_hip
# oscillating ~0.07 rad every ~1ms once it's added, confirmed stable again
# (rock-solid across repeated pose samples) once removed. A shaking robot is
# worse than missing joint feedback, so this stays off until that's root
# caused (possibly plugin publish-rate tuning, or an update-loop ordering
# issue between the two systems).

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
