"""Speed tracking error and per-term reward of SB3 checkpoints on a clean MuJoCo env.

eval_checkpoints.py reports speed from net displacement, which hides overshoot and wobble. This logs, every control
step, the velocity along the commanded direction and the yaw rate, plus the mean of every reward term, so a policy
that is paid for running faster than commanded shows up as a positive bias next to a large `progress` term.

    GO2_TROT_PRIOR=1 python3 training/eval_tracking.py \
        --models training/pretrained/go2_walk_trotprior/go2_flat_walk.zip --cmds 0.2 0.35 --episodes 4 --csv /tmp/tracking.csv

The first --skip steps of each episode (start-up from standing) are left out of the means. Falls are counted over all
steps. Env settings (GO2_* variables) must match the ones the checkpoint was trained with; when the checkpoint has a
run_config.json beside it, differences are printed. Each row gets a verdict: `stands` (mean speed below --min-speed-frac
of the command), `falls` (half or more of the episodes), `ok`, or `not_judged` (command of 0.1 m/s or less).
--fail-on-stall exits with status 2 if any row is `stands` or `falls`.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_checkpoints as ec  # noqa: E402  (imports play_policy first: import order segfaults mujoco)
from stable_baselines3 import PPO  # noqa: E402
import tracking_check as tc  # noqa: E402


def load(model_path, scene, cmd, wz):
    vn = ec._find_vecnorm(model_path)
    last = None
    for layout in ec.LAYOUTS:
        try:
            env, core = ec._build(scene, cmd, layout, vn, wz)
            return env, core, PPO.load(model_path, env=env)
        except (AssertionError, ValueError) as e:  # obs layout mismatch
            last = e
    raise SystemExit(f"{model_path}: no matching obs layout ({last})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scene", default="flat", choices=["flat", "stairs", "rough"])
    ap.add_argument("--cmds", type=float, nargs="+", default=[0.2, 0.35])
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--skip", type=int, default=50)
    ap.add_argument("--min-speed-frac", type=float, default=tc.MIN_SPEED_FRAC)
    ap.add_argument("--fail-on-stall", action="store_true")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    rows = []
    for mp in a.models:
        saved = tc.find_run_config(mp)
        for name, trained, now in tc.env_mismatches(saved) if saved else []:
            print(f"WARNING {mp}: {name} was {trained!r} in training, is {now!r} now", flush=True)
        for cmd in a.cmds:
            env, _, model = load(mp, a.scene, cmd, a.wz)
            row = dict(model=os.path.relpath(mp), scene=a.scene, cmd=cmd, wz=a.wz, seed=a.seed,
                       **tc.run(env, model, cmd, a.wz, a.episodes, a.seed, a.max_steps, a.skip))
            row["verdict"] = tc.verdict(row, cmd, a.min_speed_frac)
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, (float, np.floating)) else f"{k}={v}"
                           for k, v in row.items()), flush=True)
    if a.csv and rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    if a.fail_on_stall and any(r["verdict"] in ("stands", "falls") for r in rows):
        sys.exit(2)


if __name__ == "__main__":
    main()
