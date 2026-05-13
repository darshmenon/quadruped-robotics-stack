"""
Train Unitree Go2 walking policy using Gazebo (headless) + ROS2 + SB3 PPO.

Requires ROS2 sourced. Gazebo is auto-launched headlessly.

Usage:
    source /opt/ros/humble/setup.bash
    source ros2/install/setup.bash
    python3 training/train_gazebo.py
    python3 training/train_gazebo.py --timesteps 500000 --cmd 0.5 0.0 0.0
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from envs.go2_gazebo_env import Go2GazeboEnv

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "gazebo")
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                        metavar=("LIN_X", "LIN_Y", "ANG_YAW"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-launch", action="store_true",
                        help="don't auto-launch Gazebo (use externally running sim)")
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    cmd = tuple(args.cmd)

    print(f"Training Go2 (Gazebo) | cmd={cmd} | steps={args.timesteps}")
    print("Launching Gazebo headlessly..." if not args.no_launch else "Using existing Gazebo...")

    env = Monitor(Go2GazeboEnv(cmd=cmd, auto_launch=not args.no_launch))

    callbacks = [
        CheckpointCallback(save_freq=10_000, save_path=CKPT_DIR, name_prefix="go2_gazebo"),
        EvalCallback(env, best_model_save_path=LOG_DIR,
                     log_path=LOG_DIR, eval_freq=10_000,
                     n_eval_episodes=3, deterministic=True, render=False),
    ]

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=LOG_DIR)
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            n_steps=500,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            tensorboard_log=LOG_DIR,
            verbose=1,
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=not bool(args.resume))
    model.save(os.path.join(LOG_DIR, "go2_gazebo_final"))
    print("Training done. Model saved to", LOG_DIR)
    env.close()


if __name__ == "__main__":
    main()
