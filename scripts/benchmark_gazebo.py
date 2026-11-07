#!/usr/bin/env python3
"""Benchmark the Quad-SDK NMPC and the RL walker on the same Go2 task in Gazebo: walk to a goal straight ahead.

    scripts/benchmark_gazebo.sh --controller both --trials 3 --goal 5.0

Each trial starts a fresh Gazebo + controller stack in its own ROS_DOMAIN_ID / GZ_PARTITION, waits for the robot to
stand, then measures from simulator ground truth (not the controller's own estimate) until the goal is reached, the
robot falls, or the sim-time budget runs out. Everything it starts is stopped afterwards. One row per trial is
appended to --csv; a summary table is printed (and written to --md when given).

nmpc: ros2 launch quad_utils quad_gazebo.py (world big_flat.sdf) + quad_plan.py goal_state:=[goal, 0]
rl:   training/launch/gazebo_rl.launch.py (open world) + scripts/rl_cmd_vel_go2_gz.py, /cmd_vel = vx plus a yaw
      P-controller on heading (what a navigation layer would send), stopped at the goal.
"""
import argparse
import csv
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIELDS = ["controller", "trial", "reached", "fell", "time_s", "dist_m", "mean_speed", "max_lateral_m",
          "final_lateral_m", "heading_change_deg", "min_z", "path_m", "sim_s", "wall_s"]


def stop(p):
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(p.pid), sig)
            p.wait(timeout=20)
            return
        except (subprocess.TimeoutExpired, ProcessLookupError):
            continue


def kill_partition(tag):
    """gz sim and helpers leave the launch's process group: stop whatever still carries this run's partition."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if f"GZ_PARTITION={tag}" in open(f"/proc/{pid}/environ").read().split("\0") and int(pid) != os.getpid():
                os.kill(int(pid), signal.SIGKILL)
        except (OSError, ValueError):
            pass


def run_trial(ctl, trial, args):
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import UInt8

    d = random.randint(60, 230)
    tag = f"go2_bench_{d}"
    env = dict(os.environ, ROS_DOMAIN_ID=str(d), GZ_PARTITION=tag, GO2_TROT_PRIOR="1")
    os.environ.update(env)
    base = "source /opt/ros/humble/setup.bash >/dev/null 2>&1; " \
           f"source {REPO}/ros2/install/setup.bash >/dev/null 2>&1; source {REPO}/ros2/quad_sdk_external/setup_env.sh >/dev/null 2>&1; "
    if ctl == "rl":
        base += f"export AMENT_PREFIX_PATH={REPO}/ros2/install/champ_description:{REPO}/ros2/install/champ_config:$AMENT_PREFIX_PATH; "
    logdir = REPO / "training/logs/run_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    procs = []

    def spawn(cmd, logname):
        return subprocess.Popen(["bash", "-c", base + "exec " + cmd], env=env, stdout=open(logdir / logname, "w"),
                                stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    st = {"x": None, "y": None, "z": None, "yaw": None, "t": None}
    row = dict(controller=ctl, trial=trial, reached=0, fell=0)
    wall0 = time.time()
    try:
        if ctl == "nmpc":
            procs.append(spawn(f"ros2 launch quad_utils quad_gazebo.py gui:=false rviz:=false world:={args.nmpc_world}",
                               f"bench_nmpc_gz_{d}.log"))
        else:
            urdf = "/tmp/go2_gz_noarm.urdf"
            subprocess.run([sys.executable, str(REPO / "scripts/make_go2_noarm_urdf.py"), "--out", urdf], check=True)
            procs.append(spawn(f"ros2 launch {REPO}/training/launch/gazebo_rl.launch.py headless:=true urdf:={urdf} "
                               f"world:={args.rl_world} stand_duration:=9.0 enable_lidar3d:=false enable_octomap:=false "
                               "p_gain:=20 d_gain:=1.0 mu:=1.0", f"bench_rl_gz_{d}.log"))
        print(f"[{ctl} #{trial}] domain {d}", flush=True)

        rclpy.init()
        node = Node("benchmark_gazebo", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        threading.Thread(target=ex.spin, daemon=True).start()

        def put(p, q, t):
            st["x"], st["y"], st["z"] = p.x, p.y, p.z
            st["yaw"] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            st["t"] = t

        if ctl == "nmpc":
            from quad_msgs.msg import RobotState
            node.create_subscription(RobotState, "/robot_1/state/ground_truth", lambda m: put(
                m.body.pose.position, m.body.pose.orientation, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9), 10)
            mode = node.create_publisher(UInt8, "/robot_1/control/mode", 10)
        else:
            node.create_subscription(Odometry, "/odom", lambda m: put(
                m.pose.pose.position, m.pose.pose.orientation, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9), 10)
            cmd = node.create_publisher(Twist, "/cmd_vel", 10)

        t0 = time.time()
        while st["x"] is None and time.time() - t0 < args.boot_wait:
            time.sleep(0.5)
        if st["x"] is None:
            print(f"[{ctl} #{trial}] no ground truth within {args.boot_wait:.0f} s -- sim did not come up", flush=True)
            return None

        if ctl == "nmpc":
            time.sleep(10.0)  # let the robot settle from its drop
            t1 = time.time()
            while time.time() - t1 < 5.0:  # hold stand: a one-shot publish can be lost to a discovery race
                mode.publish(UInt8(data=1))
                time.sleep(0.1)
            time.sleep(2.0)
            procs.append(spawn(f"ros2 launch quad_utils quad_plan.py goal_state:=\"[{args.goal}, 0.0]\"", f"bench_nmpc_plan_{d}.log"))
        else:
            time.sleep(args.stand_wait)
            plog = logdir / f"bench_rl_policy_{d}.log"
            procs.append(spawn(f"python3 -u {REPO}/scripts/rl_cmd_vel_go2_gz.py --gyro-scale 1.0 --yaw-fb 0.0 --model {args.model}",
                               plog.name))
            tr = time.time()
            while "rl_cmd_vel ready" not in plog.read_text() and time.time() - tr < 300:
                time.sleep(1.0)  # torch import can take minutes on a loaded machine
            time.sleep(2.0)

        x0, y0, yaw0, tstart = st["x"], st["y"], st["yaw"], st["t"]
        row_t0 = None
        path, last, minz, maxlat = 0.0, (x0, y0), st["z"], 0.0
        wall_start = time.time()
        moved_at = None
        while True:
            x, y, z, yaw, t = st["x"], st["y"], st["z"], st["yaw"], st["t"]
            if ctl == "rl":
                err = math.atan2(math.sin(yaw0 - yaw), math.cos(yaw0 - yaw))
                m = Twist()
                m.linear.x = args.rl_vx if x - x0 < args.goal else 0.0
                m.angular.z = max(-args.rl_wz_max, min(args.rl_wz_max, args.rl_steer * err))
                cmd.publish(m)
            path += math.hypot(x - last[0], y - last[1])
            last = (x, y)
            minz = min(minz, z)
            maxlat = max(maxlat, abs(y - y0))
            if row_t0 is None and math.hypot(x - x0, y - y0) > 0.15:
                row_t0 = t  # time starts when the robot first leaves its spot, not when it was commanded
            sim = t - tstart
            if x - x0 >= args.goal - args.tol:
                row["reached"] = 1
                break
            if z < args.fall_z:
                row["fell"] = 1
                break
            if sim > args.timeout or time.time() - wall_start > args.timeout * 15:
                break
            time.sleep(0.05)
        x, y, yaw, t = st["x"], st["y"], st["yaw"], st["t"]
        dist = math.hypot(x - x0, y - y0)
        el = (t - row_t0) if row_t0 is not None else float("nan")
        row.update(time_s=round(el, 2), dist_m=round(dist, 2), mean_speed=round(dist / el, 3) if el == el and el > 0 else 0.0,
                   max_lateral_m=round(maxlat, 2), final_lateral_m=round(y - y0, 2),
                   heading_change_deg=round(math.degrees(math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))), 1),
                   min_z=round(minz, 2), path_m=round(path, 2), sim_s=round(t - tstart, 1), wall_s=round(time.time() - wall0))
        print(f"[{ctl} #{trial}] {row}", flush=True)
        ex.shutdown()
        rclpy.shutdown()
        return row
    finally:
        for p in reversed(procs):
            stop(p)
        kill_partition(tag)
        try:
            import rclpy as _r
            if _r.ok():
                _r.shutdown()
        except Exception:
            pass


def summarize(rows):
    out = ["| Controller | Trials | Reached goal | Falls | Time to goal (s) | Mean speed (m/s) | Max lateral drift (m) | Final heading change (deg) | Path / straight |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ctl in ("nmpc", "rl"):
        r = [x for x in rows if x["controller"] == ctl]
        if not r:
            continue
        ok = [x for x in r if x["reached"]]

        def mean(key, src=r):
            v = [float(x[key]) for x in src if x.get(key) not in (None, "") and float(x[key]) == float(x[key])]
            return f"{statistics.mean(v):.2f}" if v else "-"
        eff = [float(x["path_m"]) / max(float(x["dist_m"]), 1e-6) for x in r if float(x["dist_m"]) > 0.5]
        out.append(f"| {'Quad-SDK NMPC' if ctl == 'nmpc' else 'RL walker (lat2)'} | {len(r)} | {len(ok)}/{len(r)} | "
                   f"{sum(int(x['fell']) for x in r)}/{len(r)} | {mean('time_s', ok)} | {mean('mean_speed', ok or r)} | "
                   f"{mean('max_lateral_m')} | {mean('heading_change_deg')} | {statistics.mean(eff):.2f} |" if eff else
                   f"| {ctl} | {len(r)} | {len(ok)}/{len(r)} | - | - | - | - | - | - |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=["nmpc", "rl", "both"], default="both")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--goal", type=float, default=5.0, help="forward distance to walk (m)")
    ap.add_argument("--tol", type=float, default=0.25, help="goal reached within this distance (m)")
    ap.add_argument("--timeout", type=float, default=60.0, help="sim seconds allowed per trial")
    ap.add_argument("--fall-z", type=float, default=0.12, help="base height below which the robot counts as fallen")
    ap.add_argument("--boot-wait", type=float, default=180.0)
    ap.add_argument("--stand-wait", type=float, default=25.0, help="RL: wall seconds for the stack to come up and stand")
    ap.add_argument("--model", default=str(REPO / "training/pretrained/go2_walk_lat2/go2_flat_walk.zip"))
    ap.add_argument("--rl-vx", type=float, default=0.4)
    ap.add_argument("--rl-steer", type=float, default=4.0)
    ap.add_argument("--rl-wz-max", type=float, default=0.6)
    ap.add_argument("--rl-world", default=str(REPO / "training/envs/go2_gz_world_open.sdf"))
    ap.add_argument("--nmpc-world", default="big_flat.sdf")
    ap.add_argument("--csv", default=str(REPO / "docs/data/gazebo_benchmark.csv"))
    ap.add_argument("--md", default=None, help="also write the summary table to this file")
    args = ap.parse_args()

    ctls = ["nmpc", "rl"] if args.controller == "both" else [args.controller]
    rows = []
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    for ctl in ctls:
        for k in range(1, args.trials + 1):
            r = run_trial(ctl, k, args)
            if r is None:
                r = dict(controller=ctl, trial=k, reached=0, fell=0, time_s="", dist_m=0, mean_speed=0, max_lateral_m="",
                         final_lateral_m="", heading_change_deg="", min_z="", path_m=0, sim_s="", wall_s="")
            rows.append(r)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                    new = False
                w.writerow(r)
    table = summarize(rows)
    print("\n" + table)
    if args.md:
        Path(args.md).write_text(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
