"""
Train Unitree Go2 walking policy using MuJoCo + Stable-Baselines3 PPO.

Usage:
    python3 training/train_mujoco.py
    python3 training/train_mujoco.py --timesteps 5000000 --cmd 1.0 0.0 0.0
    python3 training/train_mujoco.py --resume training/logs/mujoco/checkpoints/go2_mujoco_500000_steps.zip
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.go2_mujoco_env import Go2MujocoEnv

LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs", "mujoco")
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #

class RewardComponentCallback(BaseCallback):
    """Logs each reward term to TensorBoard separately every log_interval steps."""

    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self._sums: dict = {}
        self._n: int = 0
        self._interval = log_interval

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            comps = info.get("reward_components")
            if comps:
                for k, v in comps.items():
                    self._sums[k] = self._sums.get(k, 0.0) + float(v)
                self._n += 1
        if self._n >= self._interval:
            for k, v in self._sums.items():
                self.logger.record(f"reward/{k}", v / self._n)
            self._sums = {}
            self._n = 0
        return True


class VecNormSaveCallback(BaseCallback):
    """Saves VecNormalize running stats alongside every model checkpoint."""

    def __init__(self, vec_env: VecNormalize, save_path: str, save_freq: int):
        super().__init__()
        self._vec_env   = vec_env
        self._save_path = save_path
        self._save_freq = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self._save_freq < self.training_env.num_envs:
            path = os.path.join(self._save_path,
                                f"vecnorm_{self.num_timesteps}_steps.pkl")
            self._vec_env.save(path)
        return True


# --------------------------------------------------------------------------- #
# Env factory
# --------------------------------------------------------------------------- #

def make_env(cmd, rank, seed=0):
    def _init():
        env = Go2MujocoEnv(cmd=cmd, render_mode=None,
                           randomize_domain=True, use_curriculum=True)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs",    type=int, default=8)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                        metavar=("LIN_X", "LIN_Y", "ANG_YAW"))
    parser.add_argument("--resume", type=str, default=None,
                        help="path to .zip checkpoint")
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    cmd = tuple(args.cmd)

    print(f"Training Go2 (MuJoCo) | cmd={cmd} | envs={args.n_envs} | steps={args.timesteps}")

    # ---- training envs with obs + reward normalisation ----
    vec_env = SubprocVecEnv([make_env(cmd, i) for i in range(args.n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    # ---- eval env: obs normalised (frozen), no reward norm, no domain rand ----
    eval_raw = Monitor(Go2MujocoEnv(cmd=cmd, render_mode=None,
                                    randomize_domain=False, use_curriculum=False))
    eval_env = VecNormalize(DummyVecEnv([lambda: eval_raw]),
                            norm_obs=True, norm_reward=False, training=False)

    callbacks = [
        RewardComponentCallback(log_interval=1000),
        CheckpointCallback(save_freq=50_000, save_path=CKPT_DIR,
                           name_prefix="go2_mujoco"),
        VecNormSaveCallback(vec_env, CKPT_DIR, save_freq=50_000),
        EvalCallback(
            eval_env,
            best_model_save_path=LOG_DIR,
            log_path=LOG_DIR,
            eval_freq=50_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
    ]

    if args.resume:
        # try to load matching vecnorm stats (name: vecnorm_<steps>_steps.pkl)
        ckpt_stem  = os.path.basename(args.resume).replace(".zip", "")
        steps_part = ckpt_stem.split("_steps")[0].split("_")[-1]
        norm_path  = os.path.join(CKPT_DIR, f"vecnorm_{steps_part}_steps.pkl")
        if os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env.venv)
            vec_env.norm_reward = True
            print(f"Loaded VecNormalize stats from {norm_path}")
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=LOG_DIR)
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=args.n_envs * 128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            tensorboard_log=LOG_DIR,
            verbose=1,
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=not bool(args.resume),
    )

    model.save(os.path.join(LOG_DIR, "go2_mujoco_final"))
    vec_env.save(os.path.join(LOG_DIR, "vecnorm_final.pkl"))
    print("Training done. Model saved to", LOG_DIR)
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
