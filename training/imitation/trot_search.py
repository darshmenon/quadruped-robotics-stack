"""Random search for an open-loop trot (joint-space sinusoids) that walks in Go2MujocoEnv.

Used as a cheap MuJoCo-side expert for behavior cloning: the env action is
(target - ACT_DEFAULT) / ACT_SCALE clipped to [-1, 1], so every gait param here
lives inside +/-0.25 rad of the default pose.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from envs.go2_mujoco_env import Go2MujocoEnv  # noqa: E402

# actuator order: hips[FL,FR,RL,RR], thighs[...], calves[...]
DIAG = np.array([0.0, np.pi, np.pi, 0.0])  # FL,RR in phase; FR,RL opposite


def trot_action(t, p):
    """p: f (Hz), a_thigh, a_calf (rad), s_thigh (+-1), lift_phase (rad), a_hip."""
    ph = 2 * np.pi * p["f"] * t + DIAG
    thigh = p["s_thigh"] * p["a_thigh"] * np.sin(ph)
    calf = p["a_calf"] * np.maximum(0.0, np.sin(ph + p["lift_phase"]))
    hip = p["a_hip"] * np.sin(ph) * np.array([1, -1, 1, -1])
    return np.clip(np.concatenate([hip, thigh, -calf]) / 0.25, -1.0, 1.0).astype(np.float32)


def rollout(env, p, steps=400):
    env.reset()
    x0 = float(env.data.qpos[0])
    for k in range(steps):
        _, _, term, trunc, _ = env.step(trot_action(k * 0.02, p))
        if term:
            return k, float(env.data.qpos[0]) - x0, float(env.data.qpos[1])
    return steps, float(env.data.qpos[0]) - x0, float(env.data.qpos[1])


def main(n_trials=150, seed=0, out=os.path.join(os.path.dirname(os.path.abspath(__file__)), "trot_params.json")):
    rng = np.random.default_rng(seed)
    env = Go2MujocoEnv(cmd=(0.3, 0, 0), render_mode=None, legs_only=True,
                       randomize_domain=False, use_curriculum=False, push_robots=False)
    best, results = None, []
    for i in range(n_trials):
        p = dict(f=float(rng.uniform(1.5, 3.5)), a_thigh=float(rng.uniform(0.08, 0.25)),
                 a_calf=float(rng.uniform(0.08, 0.25)), s_thigh=float(rng.choice([-1, 1])),
                 lift_phase=float(rng.uniform(-1.2, 1.2)), a_hip=float(rng.uniform(0, 0.05)))
        steps, dx, dy = rollout(env, p)
        score = dx - 0.5 * abs(dy) if steps >= 400 else -1.0 + 0.001 * steps
        results.append((score, p))
        if best is None or score > best[0]:
            best = (score, p, steps, dx, dy)
            print(f"trial {i}: score={score:.2f} steps={steps} dx={dx:.2f} dy={dy:.2f} {p}", flush=True)
    results.sort(key=lambda r: -r[0])
    json.dump([dict(score=s, **p) for s, p in results[:10]], open(out, "w"), indent=1)
    print("saved top-10 ->", out)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
