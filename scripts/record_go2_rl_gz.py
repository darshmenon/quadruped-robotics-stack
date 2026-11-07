#!/usr/bin/env python3
"""Film the RL walker in Gazebo from a follow camera and write a GIF.

    python3 scripts/record_go2_rl_gz.py --out docs/images/go2_gazebo_open.gif --seconds 40

Starts gazebo_rl.launch.py headless on the open-ground world (training/envs/go2_gz_world_open.sdf, arm-free
URDF) in its own ROS_DOMAIN_ID / GZ_PARTITION, runs scripts/rl_cmd_vel_go2_gz.py, bridges the in-world
`showcase_cam` and moves it with the set_pose service so it tracks the robot. /cmd_vel is vx plus a yaw
P-controller on /odom heading (what a navigation layer would do), so the clip shows straight-line walking if the
policy has the turning authority for it. Frames are saved on sim time (Gazebo may run below real time under load),
then scripts/make_gif.py builds the GIF. Everything started here is stopped on exit.
"""
import argparse
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def stop(p):
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(p.pid), sig)
            p.wait(timeout=20)
            return
        except (subprocess.TimeoutExpired, ProcessLookupError):
            continue


def start_gui_capture(args, env):
    """Find the Gazebo window, make its camera follow the robot, and start recording only that window with ffmpeg."""
    wid = None
    for _ in range(60):
        tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True).stdout
        best = 0
        for line in tree.splitlines():
            if "Gazebo" in line and "x" in line and line.strip().startswith("0x"):
                m = __import__("re").search(r"(\d+)x(\d+)\+", line)
                if m and int(m.group(1)) * int(m.group(2)) > best:
                    best, wid = int(m.group(1)) * int(m.group(2)), line.split()[0]
        if wid:
            break
        time.sleep(1.0)
    if not wid:
        print("no Gazebo window found -- not recording", flush=True)
        return None
    ox, oy, oz = args.gui_offset
    try:  # bring the Gazebo window to the front (EWMH _NET_ACTIVE_WINDOW), so the capture is not of whatever covers it
        from Xlib import X, display, protocol
        d = display.Display(os.environ.get("DISPLAY", ":0"))
        root = d.screen().root
        win = d.create_resource_object("window", int(wid, 16))
        ev = protocol.event.ClientMessage(window=win, client_type=d.intern_atom("_NET_ACTIVE_WINDOW"), data=(32, [1, X.CurrentTime, 0, 0, 0]))
        root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        d.flush()
    except Exception as e:
        print(f"could not raise the Gazebo window: {e}", flush=True)
    for svc, rtype, req in (("/gui/follow", "gz.msgs.StringMsg", 'data: "go2"'),
                            ("/gui/follow/offset", "gz.msgs.Vector3d", f"x: {ox} y: {oy} z: {oz}")):
        r = subprocess.run(["gz", "service", "-s", svc, "--reqtype", rtype, "--reptype", "gz.msgs.Boolean", "--timeout", "3000",
                            "--req", req], env=env, capture_output=True, text=True)
        print(f"{svc}: {(r.stdout + r.stderr).strip()[:60] or 'no reply'}", flush=True)
    time.sleep(1.5)
    mp4 = str(Path(args.out).with_suffix(".mp4"))
    print(f"recording window {wid} -> {mp4}", flush=True)
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", str(args.gui_fps),
                             "-window_id", wid, "-i", os.environ.get("DISPLAY", ":0"), "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                             "-c:v", "libx264", "-preset", "veryfast",
                             "-pix_fmt", "yuv420p", mp4], stdin=subprocess.PIPE), mp4


def stop_gui_capture(grab):
    proc, mp4 = grab
    try:
        proc.stdin.write(b"q")
        proc.stdin.flush()
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    print(f"saved {mp4}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(REPO / "training/pretrained/go2_walk_gentle/go2_flat_walk.zip"))
    ap.add_argument("--seconds", type=float, default=40.0, help="sim seconds to record while walking")
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--steer-gain", type=float, default=1.5, help="wz per rad of heading error (0 = open loop)")
    ap.add_argument("--wz-max", type=float, default=0.4)
    ap.add_argument("--yaw-fb", type=float, default=0.0)
    ap.add_argument("--domain", type=int, default=random.randint(60, 230))
    ap.add_argument("--stand-wait", type=float, default=25.0, help="wall seconds for the stack to come up and stand")
    ap.add_argument("--period", type=float, default=0.25, help="sim seconds between saved frames")
    ap.add_argument("--offset", type=float, nargs=3, default=[-1.2, -2.6, 1.1], help="camera offset from robot (m)")
    ap.add_argument("--frames-dir", default="/tmp/go2_rl_gz_frames")
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--max-mb", type=float, default=3.0)
    ap.add_argument("--world", default=str(REPO / "training/envs/go2_gz_world_open.sdf"))
    ap.add_argument("--policy-args", default="", help="extra arguments for rl_cmd_vel_go2_gz.py, e.g. '--scan-steps 0.8 0.05 0.3 4 6 --vecnorm ...'")
    ap.add_argument("--gui", action="store_true", help="open the Gazebo window (follows the robot) and screen-record that window only")
    ap.add_argument("--gui-fps", type=int, default=15)
    ap.add_argument("--gui-offset", type=float, nargs=3, default=[-1.6, -1.6, 1.0], help="GUI follow-camera offset from the robot (m)")
    args = ap.parse_args()

    urdf = "/tmp/go2_gz_noarm.urdf"
    subprocess.run([sys.executable, str(REPO / "scripts/make_go2_noarm_urdf.py"), "--out", urdf], check=True)
    d = args.domain
    env = dict(os.environ, ROS_DOMAIN_ID=str(d), GZ_PARTITION=f"go2_rec_{d}", GO2_TROT_PRIOR="1")
    src = ("source /opt/ros/humble/setup.bash >/dev/null 2>&1; "
           f"export AMENT_PREFIX_PATH={REPO}/ros2/install/champ_description:{REPO}/ros2/install/champ_config:$AMENT_PREFIX_PATH; ")
    log = REPO / "training/logs/run_logs/record_gz_launch.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    frames = Path(args.frames_dir)
    subprocess.run(["rm", "-rf", str(frames)])
    frames.mkdir(parents=True)

    def spawn(cmd, out=subprocess.DEVNULL):
        return subprocess.Popen(["bash", "-c", src + "exec " + cmd], env=env, stdout=out,
                                stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    procs = []
    os.environ.update(env)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")  # gz.msgs10 _pb2 vs system protobuf
    import cv2
    import gz.transport13 as gz_transport
    import numpy as np
    import rclpy
    from gz.msgs10.image_pb2 import Image as GzImage
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter

    try:
        procs.append(spawn(f"ros2 launch {REPO}/training/launch/gazebo_rl.launch.py urdf:={urdf} "
                           f"world:={args.world} headless:={'false' if args.gui else 'true'} stand_duration:=9.0 enable_lidar3d:=false enable_octomap:=false "
                           "p_gain:=20 d_gain:=1.0 mu:=1.0", open(log, "w")))
        print(f"domain {d}, launch log {log}", flush=True)
        time.sleep(args.stand_wait)
        plog = REPO / "training/logs/run_logs/record_gz_policy.log"
        procs.append(spawn(f"python3 -u {REPO}/scripts/rl_cmd_vel_go2_gz.py --gyro-scale 1.0 --yaw-fb {args.yaw_fb} "
                           f"--model {args.model} {args.policy_args}", open(plog, "w")))
        t_ready = time.time()
        while "rl_cmd_vel ready" not in plog.read_text() and time.time() - t_ready < 300:
            time.sleep(1.0)  # the policy node's torch/tensorflow import can take minutes on a loaded machine
        print(f"policy node ready after {time.time() - t_ready:.0f} s", flush=True)

        rclpy.init()
        node = Node("record_go2_rl_gz", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        threading.Thread(target=ex.spin, daemon=True).start()
        st = {"pose": None, "t0": None, "ref": None, "last_frame": -1e9, "n": 0, "cam": None, "path": []}
        pub = node.create_publisher(Twist, "/cmd_vel", 10)

        def on_odom(m):
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            p = m.pose.pose.position
            st["pose"] = (p.x, p.y, p.z, yaw, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)

        def on_image(m):  # gz-transport thread; frames are cut on the image's own (sim-time) stamp
            if st["t0"] is None:
                return
            t = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
            if t - st["last_frame"] < args.period:
                return
            st["last_frame"] = t
            im = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            cv2.imwrite(str(frames / f"{st['n']:05d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            st["n"] += 1

        node.create_subscription(Odometry, "/odom", on_odom, 10)
        gz_node = gz_transport.Node()
        if not gz_node.subscribe(GzImage, "/showcase_cam/image", on_image):
            print("failed to subscribe to /showcase_cam/image", flush=True)

        ox, oy, oz = args.offset

        def move_camera():
            p = st["pose"]
            if args.gui or p is None or (st["cam"] is not None and st["cam"].poll() is None):
                return
            cx, cy, cz = p[0] + ox, p[1] + oy, oz
            yaw = math.atan2(p[1] - cy, p[0] - cx)
            pitch = math.atan2(cz - 0.25, math.hypot(p[0] - cx, p[1] - cy))
            qz, qw, qy2, qw2 = math.sin(yaw / 2), math.cos(yaw / 2), math.sin(pitch / 2), math.cos(pitch / 2)
            q = (-qz * qy2, qw * qy2, qz * qw2, qw * qw2)
            req = (f'name: "showcase_cam" position: {{x: {cx}, y: {cy}, z: {cz}}} '
                   f'orientation: {{x: {q[0]}, y: {q[1]}, z: {q[2]}, w: {q[3]}}}')
            st["cam"] = subprocess.Popen(
                ["gz", "service", "-s", "/world/go2_rl/set_pose", "--reqtype", "gz.msgs.Pose", "--reptype",
                 "gz.msgs.Boolean", "--timeout", "500", "--req", req], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # wait for odom, then walk for --seconds of sim time
        t_wait = time.time()
        while st["pose"] is None and time.time() - t_wait < 120:
            time.sleep(0.5)
        if st["pose"] is None:
            print("no /odom -- Gazebo did not come up", flush=True)
            return 1
        time.sleep(2.0)
        start = st["pose"]
        st["ref"], st["t0"] = start[3], start[4]
        print(f"start pose x={start[0]:.2f} y={start[1]:.2f} yaw={math.degrees(start[3]):.0f} deg", flush=True)
        grab = None
        gui_prepared = False
        wall0 = time.time()
        while st["pose"][4] - st["t0"] < args.seconds and time.time() - wall0 < args.seconds * 12:
            p = st["pose"]
            err = math.atan2(math.sin(st["ref"] - p[3]), math.cos(st["ref"] - p[3]))
            m = Twist()
            m.linear.x = args.vx
            m.angular.z = max(-args.wz_max, min(args.wz_max, args.steer_gain * err))
            pub.publish(m)
            st["path"].append((p[4] - st["t0"], p[0], p[1], p[2], p[3]))
            if args.gui and grab is None and math.hypot(p[0] - start[0], p[1] - start[1]) > 0.1:
                grab = start_gui_capture(args, env)  # the clip starts when the robot starts moving
            move_camera()
            time.sleep(0.1)
        if grab is not None:
            stop_gui_capture(grab)
        end = st["pose"]
        dt = end[4] - st["t0"]
        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        dyaw = math.degrees(math.atan2(math.sin(end[3] - start[3]), math.cos(end[3] - start[3])))
        print(f"walked {dist:.2f} m in {dt:.1f} sim s ({dist / max(dt, 1e-6):.2f} m/s), heading change {dyaw:+.0f} deg, "
              f"min z {min(q[3] for q in st['path']):.2f}, max z {max(q[3] for q in st['path']):.2f}, "
              f"end x={st['path'][-1][1]:.2f} y={st['path'][-1][2]:.2f}, {st['n']} frames", flush=True)
        ex.shutdown()
        rclpy.shutdown()
    finally:
        for p in reversed(procs):
            stop(p)
        # gz sim and its helpers leave the launch's process group: stop whatever still carries this run's partition
        mine = f"GZ_PARTITION=go2_rec_{d}"
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                if mine in open(f"/proc/{pid}/environ").read().split("\0") and int(pid) != os.getpid():
                    os.kill(int(pid), signal.SIGKILL)
            except (OSError, ValueError):
                pass
    if args.gui:
        mp4 = str(Path(args.out).with_suffix(".mp4"))
        if not os.path.exists(mp4):
            return 1
        vf = f"fps={args.fps},scale={args.width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", vf, args.out])
        print(f"gif: {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)", flush=True)
        return 0
    if st["n"] > 5:
        subprocess.run([sys.executable, str(REPO / "scripts/make_gif.py"), str(frames), args.out,
                        "--width", str(args.width), "--fps", str(args.fps), "--max-mb", str(args.max_mb)])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
