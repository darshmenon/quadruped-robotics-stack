"""
Train Unitree Go2 walking policy using MuJoCo + Stable-Baselines3 PPO.

Usage:
    python3 training/train_mujoco.py
    python3 training/train_mujoco.py --timesteps 5000000 --cmd 1.0 0.0 0.0
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.go2_mujoco_env import Go2MujocoEnv

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "mujoco")
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")


def make_env(cmd, rank, seed=0):
    def _init():
        env = Go2MujocoEnv(cmd=cmd, render_mode=None)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                        metavar=("LIN_X", "LIN_Y", "ANG_YAW"))
    parser.add_argument("--resume", type=str, default=None, help="path to .zip checkpoint")
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    cmd = tuple(args.cmd)

    print(f"Training Go2 (MuJoCo) | cmd={cmd} | envs={args.n_envs} | steps={args.timesteps}")

    vec_env = make_vec_env(
        lambda: Go2MujocoEnv(cmd=cmd, render_mode=None),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = Monitor(Go2MujocoEnv(cmd=cmd, render_mode=None,
                                    randomize_domain=False, use_curriculum=False))

    callbacks = [
        CheckpointCallback(save_freq=50_000, save_path=CKPT_DIR, name_prefix="go2_mujoco"),
        EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                     log_path=LOG_DIR, eval_freq=50_000,
                     n_eval_episodes=5, deterministic=True, render=False),
    ]

    if args.resume:
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=LOG_DIR)
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-3,
            n_steps=1024,
            batch_size=args.n_envs * 128,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            tensorboard_log=LOG_DIR,
            verbose=1,
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=not bool(args.resume))
    model.save(os.path.join(LOG_DIR, "go2_mujoco_final"))
    print("Training done. Model saved to", LOG_DIR)
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
