#!/usr/bin/env python3
"""Evaluate a Go2 Gazebo RL policy with simple locomotion metrics."""

import argparse
import csv
import os
import time

import numpy as np

from envs.go2_gazebo_env import Go2GazeboEnv


def _load_model(path, env):
    if not path:
        return None
    from stable_baselines3 import PPO
    return PPO.load(path, env=env)


def _action(model, obs, action_shape):
    if model is None:
        return np.zeros(action_shape, dtype=np.float32)
    act, _ = model.predict(obs, deterministic=True)
    return act


def evaluate(args):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    qpos_reference = None
    if args.recovery_pose:
        from envs.go2_mujoco_recovery_env import ACT_CURL
        qpos_reference = ACT_CURL

    raw = Go2GazeboEnv(
        cmd=tuple(args.cmd),
        auto_launch=not args.no_launch,
        ros_domain_id=args.ros_domain_id,
        gz_partition=args.gz_partition,
        qpos_reference=qpos_reference,
    )
    env = DummyVecEnv([lambda: raw])
    if args.vecnorm:
        env = VecNormalize.load(args.vecnorm, env)
        env.training = False
        env.norm_reward = False
        print(f"VecNormalize stats: {args.vecnorm}")
    else:
        print("No VecNormalize stats given -- running without obs normalisation "
              "(policies trained with VecNormalize will behave incorrectly)")

    model = _load_model(args.model, env)
    rows = []

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            start = time.monotonic()
            start_pos = raw._node.get_base_xy()
            rewards = []
            speeds = []
            terminated = False
            truncated = False

            while not (terminated or truncated):
                act = _action(model, obs, raw.action_space.shape)
                obs, reward, done, _ = env.step(act)
                odom_vel = raw._node.get_lin_vel()  # after the step, so it matches this tick's reward
                terminated = bool(done[0])
                rewards.append(float(reward[0]))
                speeds.append(float(np.linalg.norm(odom_vel[:2])))
                if len(rewards) >= args.max_steps:
                    truncated = True

            elapsed = time.monotonic() - start
            mean_speed = float(np.mean(speeds)) if speeds else 0.0
            commanded_speed = float(np.linalg.norm(args.cmd[:2]))
            # Net displacement from /odom, so rocking in place doesn't count as distance.
            distance = float(np.linalg.norm(raw._node.get_base_xy() - start_pos))
            cmd_error = abs(mean_speed - commanded_speed)
            rows.append({
                "episode": ep + 1,
                "steps": len(rewards),
                "reward": round(sum(rewards), 3),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "elapsed_s": round(elapsed, 3),
                "distance_m": round(distance, 3),
                "mean_speed_mps": round(mean_speed, 3),
                "command_speed_mps": round(commanded_speed, 3),
                "speed_error_mps": round(cmd_error, 3),
            })
    finally:
        env.close()

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="SB3 PPO checkpoint; omitted evaluates zero actions")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    parser.add_argument("--no-launch", action="store_true",
                        help="use an already running Gazebo RL sim")
    parser.add_argument("--vecnorm", default=None,
                        help="VecNormalize .pkl stats matching --model -- required for "
                             "correct behavior on any policy trained with VecNormalize "
                             "(obs are otherwise fed in raw, unnormalised)")
    parser.add_argument("--recovery-pose", action="store_true",
                        help="use the fall-recovery env's curled ACT_CURL pose as the "
                             "dof_pos reference instead of this env's own standing "
                             "DEFAULT_QPOS -- needed when --model is a recovery policy")
    parser.add_argument("--ros-domain-id", default="177")
    parser.add_argument("--gz-partition", default="go2rleval")
    parser.add_argument("--csv", default=None,
                        help="optional CSV output path")
    args = parser.parse_args()

    rows = evaluate(args)
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        print(row)
    if rows:
        print(_benchmark_row(rows, args))


def _benchmark_row(rows, args):
    """Aggregate episodes into a row for the docs/BENCHMARKS.md Gazebo RL table."""
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    fall = sum(r["terminated"] for r in rows) / n
    model = os.path.basename(args.model) if args.model else "zero-action"
    cmd = " ".join(f"{c:g}" for c in args.cmd)
    return (f"| {time.strftime('%Y-%m-%d')} | {model} | `{cmd}` | {n} | {mean('steps'):.0f} | "
            f"{fall:.0%} | {mean('mean_speed_mps'):.2f} | {mean('speed_error_mps'):.2f} | |")


if __name__ == "__main__":
    main()
