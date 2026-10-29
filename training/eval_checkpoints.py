"""Fixed-difficulty sweep over SB3 checkpoints: fall rate, distance, speed.

ep_rew_mean from training is confounded by the curriculum (harder commands and
pushes as it climbs), so this is the check that says whether a policy walks.
Env is clean (no domain randomization, no pushes, no curriculum) and the
commands should stay inside what the checkpoint was trained on (training samples
vx up to 0.15 + 1.05 * curriculum_level).

    python3 training/eval_checkpoints.py --models a/go2_x_final.zip b/best_model.zip \
        --scenes flat stairs --cmds 0.2 0.35 --episodes 4 --csv /tmp/sweep.csv

The obs layout (asymmetric / history / legs-only) isn't stored in the zip, so
each model is probed against the known layouts and the first that fits is used.
VecNormalize stats are taken from the checkpoint's dir or its parent.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import play_policy as pp  # noqa: E402  (must precede stable_baselines3: import order segfaults mujoco)
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

# (asymmetric, obs_history, legs_only[, gait_conditioned])
LAYOUTS = [(True, 5, True), (True, 5, True, True), (False, 5, True), (False, 1, True),
           (True, 5, False), (False, 5, False), (False, 1, False)]


def _find_vecnorm(model_path):
    d = os.path.dirname(os.path.abspath(model_path))
    # CheckpointCallback(save_vecnormalize=True): go2_mixed_123_steps.zip <-> go2_mixed_vecnormalize_123_steps.pkl
    stem = os.path.basename(model_path)
    if stem.endswith("_steps.zip"):
        head, _, steps = stem[:-len("_steps.zip")].rpartition("_")
        c = os.path.join(d, f"{head}_vecnormalize_{steps}_steps.pkl")
        if os.path.exists(c):
            return c
    for c in (os.path.join(d, "vecnorm_final.pkl"), os.path.join(d, "..", "vecnorm_final.pkl")):
        if os.path.exists(c):
            return os.path.normpath(c)
    return None


def _build(scene, cmd, layout, vecnorm):
    asym, hist, legs, *rest = layout
    raw = pp._make_env(scene, (cmd, 0.0, 0.0), use_vision=False, obs_history=hist,
                       asymmetric=asym, legs_only=legs, gait_conditioned=bool(rest and rest[0]))
    env = DummyVecEnv([lambda: Monitor(raw)])
    if vecnorm:
        env = VecNormalize.load(vecnorm, env)
        env.training, env.norm_reward = False, False
    core = raw
    while hasattr(core, "env"):
        core = core.env
    return env, core


def _episodes(env, model, core, n, max_steps):
    out = []
    for _ in range(n):
        obs = env.reset()
        x0, y0 = pp._base_xy(core)
        x1, y1, fell, steps = x0, y0, False, 0
        for steps in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            if done[0]:
                fell = not infos[0].get("TimeLimit.truncated", False)
                break
            x1, y1 = pp._base_xy(core)
        dist = float(np.hypot(x1 - x0, y1 - y0))
        out.append((steps, dist, fell))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scenes", nargs="+", default=["flat"], choices=["flat", "stairs", "rough"])
    ap.add_argument("--cmds", type=float, nargs="+", default=[0.2, 0.35])
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    dt = pp.SIM_DT * pp.CTRL_DECIMATION
    rows = []
    print(f"{'model':<58}{'scene':<7}{'vx':>5}{'fall%':>7}{'steps':>7}{'dist_m':>8}{'vx_act':>8}")
    for mp in args.models:
        vn = _find_vecnorm(mp)
        for scene in args.scenes:
            for cmd in args.cmds:
                res = None
                for layout in LAYOUTS:
                    try:
                        env, core = _build(scene, cmd, layout, vn)
                        model = PPO.load(mp, env=env)
                        res = _episodes(env, model, core, args.episodes, args.max_steps)
                        break
                    except (AssertionError, ValueError) as e:  # obs layout mismatch
                        last = e
                label = os.path.relpath(mp)[-56:]
                if res is None:
                    print(f"{label:<58}{scene:<7}{cmd:>5.2f}  no matching obs layout ({last})")
                    continue
                steps = np.mean([r[0] for r in res])
                dist = np.mean([r[1] for r in res])
                fall = 100.0 * np.mean([r[2] for r in res])
                vx = dist / max(steps * dt, 1e-6)
                print(f"{label:<58}{scene:<7}{cmd:>5.2f}{fall:>7.0f}{steps:>7.0f}{dist:>8.2f}{vx:>8.2f}", flush=True)
                rows.append(dict(model=mp, scene=scene, cmd=cmd, fall_pct=fall, steps=steps,
                                 dist_m=dist, vx=vx, layout=layout))
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
