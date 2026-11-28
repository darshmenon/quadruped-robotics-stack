"""Run settings saved next to checkpoints, and a speed-tracking verdict that catches a policy that only stands.

Only numpy is imported here, so train_mujoco.py, eval_tracking.py and the tests can share it.
"""
import json
import os
import subprocess
import time

import numpy as np

CONFIG_NAME = "run_config.json"
MIN_SPEED_FRAC = 0.5  # mean speed below this share of the command counts as not walking
MIN_CMD = 0.1         # m/s; smaller commands are too close to standing to judge


def go2_env(environ=None):
    environ = os.environ if environ is None else environ
    return {k: v for k, v in sorted(environ.items()) if k.startswith("GO2_")}


def save_run_config(dirs, options, environ=None):
    """Append this run's GO2_* variables and options to run_config.json in each dir (resumes keep older entries)."""
    try:
        commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    entry = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), git_commit=commit, env=go2_env(environ), options=options)
    for d in dirs:
        path = os.path.join(d, CONFIG_NAME)
        runs = []
        if os.path.exists(path):
            with open(path) as f:
                runs = json.load(f)["runs"]
        with open(path, "w") as f:
            json.dump({"runs": runs + [entry]}, f, indent=2)
    return entry


def find_run_config(model_path):
    """Latest entry saved beside the checkpoint or one directory up; None for checkpoints without one."""
    d = os.path.dirname(os.path.abspath(model_path))
    for path in (os.path.join(d, CONFIG_NAME), os.path.join(d, "..", CONFIG_NAME)):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["runs"][-1]
    return None


def env_mismatches(entry, environ=None):
    """[(name, trained_value, current_value)] for GO2_* variables that differ; None means unset."""
    saved, now = entry["env"], go2_env(environ)
    return [(k, saved.get(k), now.get(k)) for k in sorted(set(saved) | set(now)) if saved.get(k) != now.get(k)]


def verdict(row, cmd, min_frac=MIN_SPEED_FRAC):
    """'ok', 'falls', 'stands' or 'not_judged' for one evaluation row from run()."""
    if row["falls"] * 2 >= row["episodes"] or row["samples"] == 0:
        return "falls"
    if cmd <= MIN_CMD:
        return "not_judged"
    return "stands" if row["vx_mean"] < min_frac * cmd else "ok"


def run(env, model, cmd, wz, episodes, seed0, max_steps, skip):
    """Roll out `model` on a one-env VecEnv and summarise speed tracking and reward terms."""
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
