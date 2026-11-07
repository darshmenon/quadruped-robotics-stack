"""Evaluate the newest checkpoint of a running training job at random intervals.

    GO2_WIDE_CMD=1 GO2_TROT_PRIOR=1 python3 training/watch_checkpoints.py \
        --ckpt-dir training/logs/mujoco_curriculum/flat/checkpoints \
        --min-minutes 6 --max-minutes 15 --log training/logs/run_logs/watch_flat.log

Each wake-up: if there is a checkpoint it hasn't scored yet, run eval_checkpoints.py on stop /
forward / turn-in-place / forward+turn (fall %, forward speed, yaw rate) and append one block to
--log. Exits once training has stopped (no train_curriculum process) and the last checkpoint has
been scored, or after --max-hours. Set the same GO2_* env vars the training run used.
"""
import argparse
import glob
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def newest(ckpt_dir):
    zips = glob.glob(os.path.join(ckpt_dir, "go2_*_steps.zip"))
    return max(zips, key=lambda p: int(re.search(r"_(\d+)_steps\.zip$", p).group(1))) if zips else None


def training_alive():
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if b"train_curriculum" in f.read() and int(pid) != os.getpid():
                    return True
        except OSError:
            pass
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--min-minutes", type=float, default=6.0)
    ap.add_argument("--max-minutes", type=float, default=15.0)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--log", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    t_end, last = time.time() + 3600 * args.max_hours, None
    while time.time() < t_end:
        ckpt = newest(args.ckpt_dir)
        if ckpt and ckpt != last:
            out = subprocess.run(
                [sys.executable, os.path.join(HERE, "eval_checkpoints.py"), "--models", ckpt, "--scenes", "flat",
                 "--cmds", "0.0", "0.3", "--wz", "0.0", "0.4", "--episodes", str(args.episodes),
                 "--max-steps", str(args.max_steps)], capture_output=True, text=True).stdout
            rows = [l for l in out.splitlines() if "flat" in l and "steps.zip" in l]
            with open(args.log, "a") as f:
                f.write(f"\n[{time.strftime('%m-%d %H:%M')}] {os.path.basename(ckpt)}\n" + "\n".join(rows) + "\n")
            last = ckpt
        elif not training_alive():
            break  # training gone and nothing new to score
        time.sleep(random.uniform(args.min_minutes, args.max_minutes) * 60)
    with open(args.log, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M')}] watcher done (last scored: {last})\n")


if __name__ == "__main__":
    main()
