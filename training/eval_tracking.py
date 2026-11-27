"""Speed tracking error and per-term reward of SB3 checkpoints on a clean MuJoCo env.

eval_checkpoints.py reports speed from net displacement, which hides overshoot and wobble. This logs, every control
step, the velocity along the commanded direction and the yaw rate, plus the mean of every reward term, so a policy
that is paid for running faster than commanded shows up as a positive bias next to a large `progress` term.

    GO2_TROT_PRIOR=1 python3 training/eval_tracking.py \
        --models training/pretrained/go2_walk_trotprior/go2_flat_walk.zip --cmds 0.2 0.35 --episodes 4 --csv /tmp/tracking.csv

The first --skip steps of each episode (start-up from standing) are left out of the means. Falls are counted over all
steps. Env settings (GO2_* variables) must match the ones the checkpoint was trained with.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_checkpoints as ec  # noqa: E402  (imports play_policy first: import order segfaults mujoco)
from stable_baselines3 import PPO  # noqa: E402


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


def run(env, model, cmd, wz, episodes, seed0, max_steps, skip):
    vx, yaw, terms, falls = [], [], {}, 0
    for ep in range(episodes):
        env.seed(seed0 + ep)
        obs = env.reset()
        for n in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            info = infos[0]
            if done[0]:
                falls += not info.get("TimeLimit.truncated", False)
            if n >= skip:
                # lin_vel is the body-frame sensor reading taken during this step, before any auto-reset
                vx.append(float(info["privileged"]["lin_vel"][0]))
                if not done[0]:  # after a reset the sim state belongs to the next episode
                    yaw.append(float(env.envs[0].unwrapped.data.qvel[5]))
                for k, v in info["reward_components"].items():
                    terms.setdefault(k, []).append(float(v))
            if done[0]:
                break
    if not vx:
        return dict(episodes=episodes, falls=int(falls), samples=0)
    vx = np.array(vx)
    row = dict(episodes=episodes, falls=int(falls), samples=len(vx),
               vx_mean=vx.mean(), vx_bias=vx.mean() - cmd, vx_rmse=np.sqrt(np.mean((vx - cmd) ** 2)),
               over_pct=100.0 * np.mean(vx > cmd),
               wz_mean=np.mean(yaw) if yaw else float("nan"),
               wz_rmse=np.sqrt(np.mean((np.array(yaw) - wz) ** 2)) if yaw else float("nan"))
    row.update({f"r_{k}": float(np.mean(v)) for k, v in sorted(terms.items())})
    return row


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
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    rows = []
    for mp in a.models:
        for cmd in a.cmds:
            env, _, model = load(mp, a.scene, cmd, a.wz)
            row = dict(model=os.path.relpath(mp), scene=a.scene, cmd=cmd, wz=a.wz, seed=a.seed,
                       **run(env, model, cmd, a.wz, a.episodes, a.seed, a.max_steps, a.skip))
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, (float, np.floating)) else f"{k}={v}"
                           for k, v in row.items()), flush=True)
    if a.csv and rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
