#!/usr/bin/env python3
"""Film the Go2 indoor autonomy demo from the Gazebo showcase camera and write a README GIF.

    python3 scripts/record_go2_gif.py --out docs/images/indoor_autonomy.gif --seconds 60

Starts launch/indoor_autonomy_go2.launch.py headless in its own ROS_DOMAIN_ID / GZ_PARTITION,
bridges the in-world `showcase_cam` (defined in training/envs/go2_gz_world_room.sdf; off unless
subscribed), saves frames to --frames-dir, then builds the GIF with scripts/make_gif.py.
Frames are kept so the GIF can be rebuilt with another crop or speed.
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def kill(p):
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(p.pid), sig)
            p.wait(timeout=20)
            return
        except (subprocess.TimeoutExpired, ProcessLookupError):
            continue


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to record once the camera is live")
    ap.add_argument("--warmup", type=float, default=40.0, help="seconds to let the stack come up before recording")
    ap.add_argument("--domain", type=int, default=147)
    ap.add_argument("--period", type=float, default=0.4)
    ap.add_argument("--frames-dir", default="/tmp/go2_showcase_frames")
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--max-mb", type=float, default=3.0)
    ap.add_argument("--stack-cmd", default=None,
                    help="shell command to run instead of the indoor autonomy launch, "
                         "e.g. './scripts/walk_quadsdk_go2.sh 3.0 0.0 headless go2_room.sdf'")
    ap.add_argument("--follow", action="store_true",
                    help="move showcase_cam to track the robot (NMPC stack: /robot_1/state/ground_truth)")
    ap.add_argument("--world", default="default", help="Gazebo world name, for the set_pose service")
    ap.add_argument("--follow-offset", type=float, nargs=3, default=[-2.4, -2.4, 2.0],
                    help="camera position relative to the robot (m)")
    ap.add_argument("--launch-args", nargs="*", default=["headless:=true", "rviz:=false"])
    args = ap.parse_args()

    env = dict(os.environ, ROS_DOMAIN_ID=str(args.domain), GZ_PARTITION=f"go2_gif_{args.domain}")
    src = ("source /opt/ros/humble/setup.bash && "
           f"source {REPO}/ros2/install/setup.bash && ")
    frames = Path(args.frames_dir)
    subprocess.run(["rm", "-rf", str(frames)])
    frames.mkdir(parents=True)
    log = REPO / "training" / "logs" / "gif_launch.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    os.environ.update(env)
    import cv2
    import numpy as np
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    def spawn(cmd, out):
        return subprocess.Popen(["bash", "-c", src + "exec " + cmd], env=env, stdout=out,
                                stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    stack = args.stack_cmd or ("ros2 launch " + str(REPO / "launch" / "indoor_autonomy_go2.launch.py")
                               + " " + " ".join(args.launch_args))
    sim = spawn(stack, open(log, "w"))
    print(f"launch log: {log} (domain {args.domain})", flush=True)
    bridge = spawn("ros2 run ros_gz_image image_bridge /showcase_cam/image", subprocess.DEVNULL)

    rclpy.init()
    node = Node("record_go2_gif")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    state = {"last": 0.0, "n": 0, "on": False, "seen": 0}

    def on_image(m):
        state["seen"] += 1
        now = time.time()
        if not state["on"] or now - state["last"] < args.period:
            return
        state["last"] = now
        im = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        cv2.imwrite(str(frames / f"{state['n']:05d}.png"),
                    cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else im)
        state["n"] += 1

    node.create_subscription(Image, "/showcase_cam/image", on_image, 5)

    if args.follow:
        from quad_msgs.msg import RobotState
        # Fixed look-at orientation (yaw toward the robot, pitched down), position
        # translated with the robot, so the view pans smoothly instead of rotating.
        ox, oy, oz = args.follow_offset
        yaw = math.atan2(-oy, -ox)
        pitch = math.atan2(oz, math.hypot(ox, oy))
        qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
        qy2, qw2 = math.sin(pitch / 2), math.cos(pitch / 2)
        quat = (-qz * qy2, qw * qy2, qz * qw2, qw * qw2)  # qz(yaw) * qy(pitch) as (x, y, z, w)
        follow = {"last": 0.0, "proc": None}

        def on_state(m):
            now = time.time()
            if now - follow["last"] < 0.25:
                return
            proc = follow["proc"]
            if proc is not None and proc.poll() is None:
                return  # previous set_pose still in flight
            follow["last"] = now
            p = m.body.pose.position
            req = (f'name: "showcase_cam" position: {{x: {p.x + ox}, y: {p.y + oy}, z: {oz}}} '
                   f'orientation: {{x: {quat[0]}, y: {quat[1]}, z: {quat[2]}, w: {quat[3]}}}')
            follow["proc"] = subprocess.Popen(
                ["gz", "service", "-s", f"/world/{args.world}/set_pose", "--reqtype", "gz.msgs.Pose",
                 "--reptype", "gz.msgs.Boolean", "--timeout", "500", "--req", req],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        node.create_subscription(RobotState, "/robot_1/state/ground_truth", on_state, 5)
    try:
        time.sleep(args.warmup)
        print(f"camera frames seen during warmup: {state['seen']}", flush=True)
        state["on"] = True
        time.sleep(args.seconds)
        state["on"] = False
    finally:
        kill(bridge)
        kill(sim)
        executor.shutdown()
        rclpy.shutdown()
    print(f"{state['n']} frames in {frames}", flush=True)
    if state["n"] > 5:
        subprocess.run([sys.executable, str(REPO / "scripts" / "make_gif.py"), str(frames), args.out,
                        "--width", str(args.width), "--fps", str(args.fps), "--max-mb", str(args.max_mb)])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
