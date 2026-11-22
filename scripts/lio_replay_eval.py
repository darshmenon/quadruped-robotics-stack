#!/usr/bin/env python3
"""Replay a recorded Go2 bag through RKO-LIO, LIO-SAM, or LIORF and score it with evo (needs ~/fusion_tools and ~/.venvs/evo).

Same bag, same walker, only the LIO settings change, so before/after comparisons are fair (live runs vary a lot).
  python3 scripts/lio_replay_eval.py training/logs/bags/go2_outdoor_weave2 --extrinsics on   # real lidar offset
  python3 scripts/lio_replay_eval.py training/logs/bags/go2_outdoor_weave2                  # identity extrinsics (default)
"""
import argparse, os, random, shutil, signal, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EST_TOPIC = {
    "rko_lio": "/rko_lio/odom",
    "lio_sam": "/lio_sam/mapping/odometry",
    "liorf": "/liorf/mapping/odometry",
}
LAUNCH = {
    "rko_lio": f"ros2 launch {REPO}/launch/lio_go2.launch.py",
    "lio_sam": f"bash -c 'source ~/lio_ws/install/setup.bash; exec ros2 launch {REPO}/launch/lio_sam_go2.launch.py'",
    "liorf": f"bash -c 'source ~/ws_liorf/install_clean/setup.bash; exec ros2 launch {REPO}/launch/liorf_go2.launch.py'",
}
FUS = Path.home() / "fusion_tools"


def _n_poses(est_dir):
    import glob, sqlite3
    try:
        return sum(sqlite3.connect(f).execute("select count(*) from messages").fetchone()[0] for f in glob.glob(str(est_dir / "*.db3")))
    except Exception:
        return 0


def run_once(a, d, env, src, out):
    procs = []
    def spawn(cmd, log):
        p = subprocess.Popen(["bash", "-c", src + "exec " + cmd], env=env, stdout=open(out / log, "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        procs.append(p); return p
    try:
        spawn(LAUNCH[a.method], "lio.log")
        spawn(f"ros2 bag record -o {out}/est {EST_TOPIC[a.method]} --use-sim-time", "rec.log")
        time.sleep(6)
        subprocess.run(["bash", "-c", src + f"ros2 bag play {a.bag} --clock --rate {a.rate} --topics /points /imu/data"],
                       env=env, stdout=open(out / "play.log", "w"), stderr=subprocess.STDOUT)
        time.sleep(3)
    finally:
        for p in reversed(procs):
            try: os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except Exception: pass
        time.sleep(12)      # give the bag recorder time to flush and write metadata.yaml before SIGKILL
        for p in procs:
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag"); ap.add_argument("--extrinsics", choices=["on", "off"], default="off", help="off = identity (default), on = real lidar offset")
    ap.add_argument("--method", choices=["rko_lio", "lio_sam", "liorf"], default="rko_lio")
    ap.add_argument("--rate", type=float, default=0.5); ap.add_argument("--out", default="/tmp/lio_replay")
    a = ap.parse_args()
    d = random.randint(60, 230)
    env = dict(os.environ, ROS_DOMAIN_ID=str(d))
    src = "source /opt/ros/humble/setup.bash >/dev/null 2>&1; "
    tag = f"{Path(a.bag).name}_{a.extrinsics}" if a.method == "rko_lio" else f"{Path(a.bag).name}_{a.method}_{a.extrinsics}"
    out = Path(a.out) / tag
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    env["LIO_EXTRINSICS_FILE"] = str(REPO / "config" / ("lio_go2_extrinsics.yaml" if a.extrinsics == "on" else "lio_go2_identity.yaml"))
    for attempt in range(1, 4):      # retry when too few poses were recorded instead of scoring nothing
        shutil.rmtree(out / "est", ignore_errors=True)
        run_once(a, d, env, src, out)
        if _n_poses(out / "est") >= 100:     # online_node always segfaults at shutdown (exit -11), so judge by the recorded poses
            break
        print(f"attempt {attempt}: only {_n_poses(out / 'est')} poses recorded, retrying", flush=True)
    tum = lambda bag, topic, f: subprocess.run(["bash", "-c", src + f"python3 {FUS}/odom_to_tum.py {bag} {topic} {f}"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tum(a.bag, "/odom", out / "gt.tum"); tum(out / "est", EST_TOPIC[a.method], out / "est.tum")
    print(f"== {tag}")
    print(subprocess.run([str(FUS / "eval_traj.sh"), str(out / "gt.tum"), str(out / "est.tum")], capture_output=True, text=True).stdout)
    resets = (out / "lio.log").read_text().count("reset count")
    print(f"odometry resets logged: {resets}")


if __name__ == "__main__":
    main()
