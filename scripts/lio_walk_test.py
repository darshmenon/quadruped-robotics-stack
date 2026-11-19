#!/usr/bin/env python3
"""Straight-walk test for LiDAR-inertial odometry on the Go2 RL walker in Gazebo.

Starts the RL stack (lidar3d on) in its own ROS_DOMAIN_ID / GZ_PARTITION, runs rko_lio on /points + /imu/data,
walks the robot --goal metres forward and compares the odometry estimate with simulator ground truth:
  distance ratio  = estimated net distance / true distance   (1.0 is perfect; resets excluded)
  heading error   = estimated vs true heading change
  resets          = odometry resets (jumps back to the origin) counted from the estimate
Writes one CSV row per run.

  python3 scripts/lio_walk_test.py --goal 5 --world training/envs/go2_gz_world_arena.sdf
"""
import argparse, csv, math, os, random, signal, subprocess, sys, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def yaw_of(q): return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
def wrap(a): return math.atan2(math.sin(a), math.cos(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", type=float, default=5.0)
    ap.add_argument("--world", default=str(REPO / "training/envs/go2_gz_world_arena.sdf"))
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--timeout", type=float, default=90.0, help="sim seconds")
    ap.add_argument("--model", default=str(REPO / "training/pretrained/go2_walk_lat2/go2_flat_walk.zip"))
    ap.add_argument("--csv", default=str(REPO / "docs/data/lio_walk.csv"))
    ap.add_argument("--label", default="rko_lio")
    ap.add_argument("--weave", action="store_true", help="sinusoidal yaw command (turns) instead of holding heading; ends after --duration sim s")
    ap.add_argument("--weave-wz", type=float, default=0.2, help="--weave only: peak yaw rate (rad/s); 0.45 made the walker spin in place")
    ap.add_argument("--duration", type=float, default=60.0, help="--weave only: sim seconds to walk")
    ap.add_argument("--bag", default=None, help="also record /points /imu/data /odom /rko_lio/odom /clock to this bag dir "
                    "(then: ~/fusion_tools/odom_to_tum.py + eval_traj.sh for evo APE/RPE)")
    ap.add_argument("--boot-wait", type=float, default=180.0)
    ap.add_argument("--stand-wait", type=float, default=30.0)
    a = ap.parse_args()

    d = random.randint(60, 230); tag = f"go2_lio_{d}"
    env = dict(os.environ, ROS_DOMAIN_ID=str(d), GZ_PARTITION=tag, GO2_TROT_PRIOR="1")
    os.environ.update(env)
    base = ("source /opt/ros/humble/setup.bash >/dev/null 2>&1; "
            f"source {REPO}/ros2/install/setup.bash >/dev/null 2>&1; "
            f"export AMENT_PREFIX_PATH={REPO}/ros2/install/champ_description:{REPO}/ros2/install/champ_config:$AMENT_PREFIX_PATH; ")
    logdir = REPO / "training/logs/run_logs"; logdir.mkdir(parents=True, exist_ok=True)
    procs = []

    def spawn(cmd, log):
        p = subprocess.Popen(["bash", "-c", base + "exec " + cmd], env=env, stdout=open(logdir / log, "w"),
                             stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        procs.append(p); return p

    urdf = "/tmp/go2_gz_noarm.urdf"
    subprocess.run([sys.executable, str(REPO / "scripts/make_go2_noarm_urdf.py"), "--out", urdf], check=True)
    print(f"domain {d}, partition {tag}", flush=True)
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    try:
        spawn(f"ros2 launch {REPO}/training/launch/gazebo_rl.launch.py headless:=true urdf:={urdf} world:={a.world} "
              "stand_duration:=9.0 enable_lidar3d:=true enable_octomap:=false p_gain:=20 d_gain:=1.0 mu:=1.0", f"lio_walk_gz_{d}.log")
        rclpy.init()
        node = Node("lio_walk_test", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        ex = SingleThreadedExecutor(); ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()
        gt, est = [], []     # (t, x, y, yaw)
        def add(buf):
            def cb(m):
                p = m.pose.pose.position
                buf.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.x, p.y, yaw_of(m.pose.pose.orientation)))
            return cb
        node.create_subscription(Odometry, "/odom", add(gt), 50)
        node.create_subscription(Odometry, "/rko_lio/odom", add(est), 50)
        cmd = node.create_publisher(Twist, "/cmd_vel", 10)

        t0 = time.time()
        while not gt and time.time() - t0 < a.boot_wait: time.sleep(0.5)
        if not gt: print("no ground truth: sim did not come up"); return 1
        time.sleep(a.stand_wait)
        plog = logdir / f"lio_walk_policy_{d}.log"
        spawn(f"python3 -u {REPO}/scripts/rl_cmd_vel_go2_gz.py --gyro-scale 1.0 --yaw-fb 0.0 --model {a.model}", plog.name)
        t1 = time.time()
        while "rl_cmd_vel ready" not in plog.read_text() and time.time() - t1 < 300: time.sleep(1.0)
        spawn("ros2 launch " + str(REPO / "launch/lio_go2.launch.py"), f"lio_walk_lio_{d}.log")
        if a.bag:
            if os.path.exists(a.bag): raise SystemExit(f"{a.bag} exists; pick a new --bag dir")
            spawn(f"ros2 bag record -o {a.bag} /points /imu/data /odom /rko_lio/odom /clock /joint_states", f"lio_walk_bag_{d}.log")
        time.sleep(8.0)        # let the odometry initialise while standing
        i_gt, i_est = len(gt), len(est)
        x0, y0, yaw0, tstart = gt[-1][1], gt[-1][2], gt[-1][3], gt[-1][0]
        print(f"start gt=({x0:.2f},{y0:.2f}) est samples so far={len(est)}", flush=True)
        wall = time.time()
        while True:
            x, y, yaw, t = gt[-1][1], gt[-1][2], gt[-1][3], gt[-1][0]
            m = Twist()
            if a.weave:      # S-curves: yaw rate +-weave_wz with a 30 s period
                m.linear.x = a.vx; m.angular.z = a.weave_wz * math.sin(2 * math.pi * (t - tstart) / 30.0)
                done = t - tstart > a.duration
            else:
                m.linear.x = a.vx if x - x0 < a.goal else 0.0
                m.angular.z = max(-0.6, min(0.6, 4.0 * wrap(yaw0 - yaw)))
                done = x - x0 >= a.goal - 0.1
            cmd.publish(m)
            if done or t - tstart > a.timeout or time.time() - wall > a.timeout * 15: break
            time.sleep(0.05)
        time.sleep(1.0)
        g = gt[i_gt:]; e = est[i_est:]
        true_dist = math.hypot(g[-1][1] - g[0][1], g[-1][2] - g[0][2])
        # estimated distance: net displacement of each reset-free segment (a >0.5 m step is a reset jump back to the
        # origin); summing every step would count sensor jitter as motion
        cuts = [0] + [i for i in range(1, len(e)) if math.hypot(e[i][1] - e[i - 1][1], e[i][2] - e[i - 1][2]) > 0.5] + [len(e)]
        resets = len(cuts) - 2
        est_dist = sum(math.hypot(e[b - 1][1] - e[a_][1], e[b - 1][2] - e[a_][2]) for a_, b in zip(cuts, cuts[1:]) if b - a_ > 1)
        dh_gt = math.degrees(wrap(g[-1][3] - g[0][3])); dh_est = math.degrees(wrap(e[-1][3] - e[0][3])) if e else float("nan")
        row = dict(label=a.label, world=Path(a.world).name + ("+weave" if a.weave else ""), goal=a.goal, true_dist=round(true_dist, 2), est_samples=len(e),
                   est_dist=round(est_dist, 2), dist_ratio=round(est_dist / true_dist, 3) if true_dist > 0.05 else None,
                   resets=resets, heading_gt_deg=round(dh_gt, 1), heading_est_deg=round(dh_est, 1),
                   sim_s=round(g[-1][0] - g[0][0], 1))
        print(row, flush=True)
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
        return 0
    finally:
        for p in reversed(procs):
            try: os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except Exception: pass
        time.sleep(6)
        for p in procs:
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception: pass
        # gz sim leaves the launch's process group: stop only what carries this run's partition
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                if f"GZ_PARTITION={tag}" in open(f"/proc/{pid}/environ").read().split("\0") and int(pid) != os.getpid():
                    os.kill(int(pid), signal.SIGKILL)
            except Exception: pass


if __name__ == "__main__":
    sys.exit(main())
