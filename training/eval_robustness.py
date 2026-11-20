"""Robustness sweep for a Go2 MuJoCo checkpoint: which single perturbation breaks the walker?

Each perturbation is applied alone at a fixed level, on the same episode seeds, so the fall rates are comparable
across levels and across checkpoints. The levels are the known sim-to-sim gaps: the roll/pitch gyro gain (Gazebo's IMU
reads 2-2.4x MuJoCo's), actuation latency (Gazebo runs below real time), floor friction, base mass and base-velocity
pushes. Falling is rare for these walkers, so the speed column (vx, and vx relative to the baseline row) is the one
that shows degradation: a walker that survives but stalls is flagged "slowed". Fall rate comes with a 95% Wilson interval, so a 2/16 vs 5/16 difference is not read as a real one.

    python3 training/eval_robustness.py --models training/pretrained/go2_walk_lat2/go2_flat_walk.zip \
        --cmd 0.3 --episodes 16 --csv /tmp/robustness.csv
    python3 training/eval_robustness.py --models a.zip b.zip --only gyro latency   # compare two checkpoints

Perturbation levels can be overridden, e.g. --gyro 1 2 3 --latency 0 2 4 --push 0.2 0.5. Like eval_checkpoints.py the
env is otherwise clean (no randomization, no curriculum) and the obs layout is probed from the known layouts.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_checkpoints as ec  # noqa: E402  (imports play_policy first: import order segfaults mujoco)
import play_policy as pp  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

PUSH_EVERY_S = 2.0


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials, as (low, high) fractions."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def perturbation_plan(a):
    """[(family, level, label)] with the baseline first."""
    plan = [("baseline", 0, "none")]
    for fam, levels, unit in (("gyro", a.gyro, "x"), ("latency", a.latency, " steps"), ("friction", a.friction, "x"),
                              ("mass", a.mass, "x"), ("push", a.push, " m/s")):
        if a.only and fam not in a.only:
            continue
        plan += [(fam, lv, f"{lv:g}{unit}") for lv in levels]
    return plan


def apply_static(core, base, fam, level):
    """Set the per-episode perturbation on the raw env (before reset). Pushes are applied in the step loop."""
    core.rand_gyro, core._gyro_gain = (fam == "gyro"), np.array([level, level, 1.0], np.float32) if fam == "gyro" else np.ones(3, np.float32)
    core.rand_lat, core._lat, core._ctrl_hist = (fam == "latency"), int(level) if fam == "latency" else 0, []
    core.model.geom_friction[core._floor_geom_id] = base["fric"] * (level if fam == "friction" else 1.0)
    core.model.body_mass[core._base_body_id] = base["mass"] * (level if fam == "mass" else 1.0)


def run_level(env, model, core, base, fam, level, episodes, seed0, max_steps, dt):
    falls, steps_l, dist_l = 0, [], []
    for ep in range(episodes):
        apply_static(core, base, fam, level)
        env.seed(seed0 + ep)
        rng = np.random.default_rng(seed0 + ep)
        obs = env.reset()
        x0, y0 = pp._base_xy(core)
        x1, y1, fell, n = x0, y0, False, 0
        for n in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            if fam == "push" and n % int(PUSH_EVERY_S / dt) == 0:
                ang = rng.uniform(0, 2 * math.pi)
                core.data.qvel[0] += level * math.cos(ang)
                core.data.qvel[1] += level * math.sin(ang)
            if done[0]:
                fell = not infos[0].get("TimeLimit.truncated", False)
                break
            x1, y1 = pp._base_xy(core)
        falls += fell
        steps_l.append(n)
        dist_l.append(float(np.hypot(x1 - x0, y1 - y0)))
    return falls, float(np.mean(steps_l)), float(np.mean(dist_l))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scene", default="flat", choices=["flat", "stairs", "rough"])
    ap.add_argument("--cmd", type=float, default=0.3, help="forward command, m/s")
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--gyro", type=float, nargs="+", default=[1.0, 2.0, 2.5])
    ap.add_argument("--latency", type=float, nargs="+", default=[0, 1, 2])
    ap.add_argument("--friction", type=float, nargs="+", default=[0.5, 1.5])
    ap.add_argument("--mass", type=float, nargs="+", default=[0.85, 1.15])
    ap.add_argument("--push", type=float, nargs="+", default=[0.3, 0.6])
    ap.add_argument("--only", nargs="+", choices=["gyro", "latency", "friction", "mass", "push"])
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    dt = pp.SIM_DT * pp.CTRL_DECIMATION
    plan = perturbation_plan(a)
    rows = []
    for mp in a.models:
        vn = ec._find_vecnorm(mp)
        env = core = model = layout = None
        for lay in ec.LAYOUTS:
            try:
                env, core = ec._build(a.scene, a.cmd, lay, vn, a.wz)
                model = PPO.load(mp, env=env)
                layout = lay
                break
            except (AssertionError, ValueError):
                env = None
        label = os.path.relpath(mp)
        if env is None:
            print(f"{label}: no matching obs layout")
            continue
        base = dict(fric=core.model.geom_friction[core._floor_geom_id].copy(), mass=float(core.model.body_mass[core._base_body_id]))
        print(f"\n{label}  scene={a.scene} cmd={a.cmd} m/s, {a.episodes} episodes, seeds {a.seed}..{a.seed + a.episodes - 1}")
        print(f"{'perturbation':<12}{'level':>11}{'fall%':>7}{'95% CI':>14}{'steps':>7}{'dist_m':>8}{'vx':>7}{'vs base':>9}")
        base_vx = None
        for fam, level, lab in plan:
            falls, steps, dist = run_level(env, model, core, base, fam, level, a.episodes, a.seed, a.max_steps, dt)
            lo, hi = wilson(falls, a.episodes)
            vx = dist / max(steps * dt, 1e-6)
            base_vx = vx if base_vx is None else base_vx          # the baseline row comes first
            rel = vx / max(base_vx, 1e-6)
            flag = "  <- slowed" if rel < 0.75 else ""
            print(f"{fam:<12}{lab:>11}{100 * falls / a.episodes:>7.0f}{f'{100 * lo:.0f}-{100 * hi:.0f}':>14}{steps:>7.0f}{dist:>8.2f}{vx:>7.2f}{100 * rel:>8.0f}%{flag}", flush=True)
            rows.append(dict(model=mp, scene=a.scene, cmd=a.cmd, family=fam, level=lab, episodes=a.episodes, falls=falls,
                             fall_pct=100 * falls / a.episodes, ci_lo=100 * lo, ci_hi=100 * hi, steps=steps, dist_m=dist, vx=vx, vx_vs_baseline=rel,
                             layout=layout))
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
