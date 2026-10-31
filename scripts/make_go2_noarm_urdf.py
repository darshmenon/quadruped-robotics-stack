#!/usr/bin/env python3
"""Write a copy of go2_gz.urdf without the manipulator arm.

    python3 scripts/make_go2_noarm_urdf.py --out /tmp/go2_gz_noarm.urdf
    ros2 launch training/launch/gazebo_rl.launch.py urdf:=/tmp/go2_gz_noarm.urdf

Drops every link/joint under `arm_base` (arm, wrist, gripper, fingers) and their entries in the
ros2_control block; legs, lidar, IMU and the <gazebo> plugins stay. The arm is ~0.5 kg, so the
MuJoCo-trained walker's dynamics barely change (see README: 0% falls with and without it).
"""
import argparse
import os
import xml.etree.ElementTree as ET

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ARM_ROOT = "arm_base"


def strip_arm(root):
    joints = {j.get("name"): (j.find("parent").get("link"), j.find("child").get("link"))
              for j in root.findall("joint")}
    arm_links = {ARM_ROOT}
    grew = True
    while grew:
        grew = False
        for parent, child in joints.values():
            if parent in arm_links and child not in arm_links:
                arm_links.add(child)
                grew = True
    arm_joints = {n for n, (p, c) in joints.items() if p in arm_links or c in arm_links}
    for el in list(root):
        if (el.tag == "link" and el.get("name") in arm_links) or \
           (el.tag == "joint" and el.get("name") in arm_joints) or \
           (el.tag == "gazebo" and el.get("reference") in arm_links):
            root.remove(el)
    for rc in root.findall("ros2_control"):
        for j in rc.findall("joint"):
            if j.get("name") in arm_joints:
                rc.remove(j)
    return sorted(arm_links), sorted(arm_joints)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=os.path.join(REPO, "urdf", "go2_unitree", "urdf", "go2_gz.urdf"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tree = ET.parse(args.urdf)
    links, joints = strip_arm(tree.getroot())
    tree.write(args.out, encoding="unicode", xml_declaration=True)
    print(f"removed {len(links)} links, {len(joints)} joints -> {args.out}")


if __name__ == "__main__":
    main()
