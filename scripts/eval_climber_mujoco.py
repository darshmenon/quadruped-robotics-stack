#!/usr/bin/env python3
"""Measured episodes for the step-climbing claims in the README (MuJoCo only, no viewer).

    python3 scripts/eval_climber_mujoco.py --height 3 --episodes 10 --seed 0 --csv docs/data/climber_eval.csv

Runs the go2_climb_scan policy for the given riser height on the four-step Unitree scene
(training/envs/go2_steps_h0N_unitree.xml, steps start at x = 0.8 m, tread 0.30 m, see make_step_scene.py),
deterministic actions, fixed seeds (seed, seed+1, ...). One CSV row per episode:

  height_cm, seed, steps, terminated, stalled, fell, max_x, steps_cleared, reached_top, ep_len_s
  steps_cleared = number of riser lines (x = 0.8 + 0.3 * i, i = 0..3) the base centre passed by more than 0.10 m
  reached_top   = max_x >= 2.0 (past the last riser onto the top platform)
  fell          = terminated and not stalled (tilt or base height out of range, per the env)
  stalled       = terminated by the env's stall rule (parked in front of a step)

The env variables below mirror training/office_demo.py --steps (3 cm) and its --h branch (5 cm, mixture-of-experts
climber). If a checkpoint does not load with these settings the script exits with the error instead of guessing.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(HERE), "training")

RISER_X = [0.8 + 0.3 * i for i in range(4)]
TOP_X = 2.0


def configure(height_cm, scene="unitree", moe=None):
    os.environ.setdefault("GO2_TROT_PRIOR", "1")
    os.environ.setdefault("GO2_WIDE_CMD", "1")
    os.environ["GO2_SCENE_XML"] = os.path.join(TRAIN, "envs", "go2_plain_scene.xml")
    os.environ.update(GO2_STAIRS_VISION="1", GO2_NO_STAND_SUBSIDY="1", GO2_LIFT_SCALE="2.0",
                      GO2_STAIRS_SCENE=os.path.join(TRAIN, "envs", f"go2_steps_h0{height_cm}{'_unitree' if scene == 'unitree' else ''}.xml"))
    if moe is None:
        moe = height_cm != 3
    if moe:
        os.environ.update(GO2_MOE="3", GO2_STD_FLOOR="0.45", GO2_STABILITY_W="0.02", GO2_FOOTHOLD_W="0.5",
                          GO2_STALL_PENALTY="120")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=int, choices=(3, 5), required=True, help="riser height in cm")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", choices=("unitree", "plain"), default="unitree",
                    help="unitree = go2_steps_h0N_unitree.xml (default), plain = go2_steps_h0N.xml (box-only robot geometry)")
    ap.add_argument("--moe", choices=("auto", "on", "off"), default="auto",
                    help="training-time MoE/reward env vars; auto = on for 5 cm. They do not change how a saved policy loads.")
    ap.add_argument("--variant", default="", help="free-text label stored in the CSV")
    ap.add_argument("--vx", type=float, default=0.35)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--model", default=None)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--csv", default=None, help="append rows here (header written if the file is new)")
    a = ap.parse_args()

    configure(a.height, a.scene, None if a.moe == "auto" else a.moe == "on")
    sys.path.insert(0, TRAIN)
    sys.path.insert(0, os.path.join(TRAIN, "envs"))
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from play_policy import _make_env

    pre = os.path.join(TRAIN, "pretrained", "go2_climb_scan")
    model_path = a.model or os.path.join(pre, f"go2_stairs_h0{a.height}.zip")
    # office_demo.py loads vecnorm_final.pkl next to the model for both heights; vecnorm_h05.pkl gave 3/10 at 5 cm vs 8/10 here
    vn_path = a.vecnorm or os.path.join(pre, "vecnorm_final.pkl")

    raw = _make_env("stairs", (a.vx, 0, 0), use_vision=True, gait_conditioned=True, obs_history=5,
                    asymmetric=True, legs_only=True)
    core = raw
    while hasattr(core, "env"):
        core = core.env
    env = DummyVecEnv([lambda: raw])
    env = VecNormalize.load(vn_path, env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(model_path, env=env)

    dt = core.model.opt.timestep * 4
    rows = []
    for ep in range(a.episodes):
        seed = a.seed + ep
        env.seed(seed)
        obs = env.reset()
        max_x, steps, term, stalled = float(core.data.qpos[0]), 0, False, False
        for steps in range(1, a.max_steps + 1):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(act)
            max_x = max(max_x, float(infos[0].get("x", core.data.qpos[0])))
            if done[0]:
                tr = bool(infos[0].get("TimeLimit.truncated", False))
                term = not tr
                stalled = bool(getattr(core, "_stalled", False)) and term
                break
        cleared = sum(1 for r in RISER_X if max_x > r + 0.10)
        row = dict(height_cm=a.height, variant=a.variant, seed=seed, steps=steps, terminated=int(term), stalled=int(stalled),
                   fell=int(term and not stalled), max_x=round(max_x, 3), steps_cleared=cleared,
                   reached_top=int(max_x >= TOP_X), ep_len_s=round(steps * dt, 2))
        rows.append(row)
        print(row, flush=True)

    n = len(rows)
    print(f"h={a.height} cm: {n} episodes, falls {sum(r['fell'] for r in rows)}, "
          f"stalls {sum(r['stalled'] for r in rows)}, reached top {sum(r['reached_top'] for r in rows)}, "
          f"mean steps cleared {sum(r['steps_cleared'] for r in rows) / n:.2f}")
    if a.csv:
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
