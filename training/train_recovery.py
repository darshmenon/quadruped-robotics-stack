"""
Train a Go2 fall-recovery policy in MuJoCo (FR-Net-style, SB3 PPO).

Usage:
    python3 training/train_recovery.py
    python3 training/train_recovery.py --timesteps 1000000 --n_envs 8
    python3 training/train_recovery.py --resume training/logs/recovery/checkpoints/go2_recovery_500000_steps.zip
"""

import argparse
import glob
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault("triton", None)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.go2_mujoco_recovery_env import Go2MujocoRecoveryEnv, ROUGH_SCENE_XML
from envs.obs_history import ObsHistoryWrapper

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "recovery")
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")
CURRICULUM_PATH = os.path.join(LOG_DIR, "curriculum_level.txt")


class RewardComponentCallback(BaseCallback):
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self._sums: dict = {}
        self._n = 0
        self._interval = log_interval
        self._upright = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            comps = info.get("reward_components")
            if comps:
                for k, v in comps.items():
                    self._sums[k] = self._sums.get(k, 0.0) + float(v)
                self._n += 1
            if info.get("upright"):
                self._upright += 1
        if self._n >= self._interval:
            for k, v in self._sums.items():
                self.logger.record(f"reward/{k}", v / self._n)
            self.logger.record("recovery/upright_frac",
                               self._upright / max(self._n, 1))
            self._sums = {}
            self._n = 0
            self._upright = 0
        return True


class VecNormSaveCallback(BaseCallback):
    def __init__(self, vec_env: VecNormalize, save_path: str, save_freq: int,
                 curriculum_path: str):
        super().__init__()
        self._vec_env = vec_env
        self._save_path = save_path
        self._save_freq = save_freq
        self._curriculum_path = curriculum_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self._save_freq < self.training_env.num_envs:
            path = os.path.join(
                self._save_path, f"vecnorm_{self.num_timesteps}_steps.pkl")
            self._vec_env.save(path)
            level = float(np.mean(self._vec_env.get_attr("curriculum_level")))
            with open(self._curriculum_path, "w") as f:
                f.write(str(level))
        return True


def make_env(rank, seed=0, curriculum_level=0.0, obs_history=1, rough=False):
    def _init():
        scene = ROUGH_SCENE_XML if rough else None
        env = Go2MujocoRecoveryEnv(
            render_mode=None, randomize_domain=True,
            initial_curriculum_level=curriculum_level,
            scene_xml=scene)
        if obs_history > 1:
            env = ObsHistoryWrapper(env, history_len=obs_history)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train Go2 MuJoCo fall recovery (FR-Net-style)")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--curriculum_level", type=float, default=None)
    parser.add_argument("--obs-history", type=int, default=1, metavar="N")
    parser.add_argument("--rough", action="store_true",
                        help="train recovery on go2_rough_scene.xml (flat spawn patch)")
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    log_dir = args.log_dir or LOG_DIR
    if args.log_dir is None:
        parts = []
        if args.rough:
            parts.append("rough")
        if args.obs_history > 1:
            parts.append(f"hist{args.obs_history}")
        if parts:
            log_dir = os.path.join(
                os.path.dirname(LOG_DIR), "recovery_" + "_".join(parts))
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    curriculum_path = os.path.join(log_dir, "curriculum_level.txt")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.curriculum_level is None:
        if os.path.exists(curriculum_path):
            with open(curriculum_path) as f:
                args.curriculum_level = float(f.read().strip())
            print(f"Loaded curriculum_level={args.curriculum_level:.3f}")
        else:
            args.curriculum_level = 0.0

    print(f"Recovery train | envs={args.n_envs} | steps={args.timesteps} "
          f"| curr={args.curriculum_level:.3f} | hist={args.obs_history} "
          f"| rough={args.rough} | log={log_dir}")

    vec_env = DummyVecEnv([
        make_env(i, curriculum_level=args.curriculum_level,
                 obs_history=args.obs_history, rough=args.rough)
        for i in range(args.n_envs)])
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    def _make_eval():
        scene = ROUGH_SCENE_XML if args.rough else None
        e = Go2MujocoRecoveryEnv(
            randomize_domain=False, initial_curriculum_level=args.curriculum_level,
            scene_xml=scene)
        if args.obs_history > 1:
            e = ObsHistoryWrapper(e, history_len=args.obs_history)
        return Monitor(e)

    eval_env = VecNormalize(
        DummyVecEnv([_make_eval]),
        norm_obs=True, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=log_dir, log_path=log_dir,
        # Both CheckpointCallback.save_freq and EvalCallback.eval_freq count
        # n_calls, one per vectorized env.step() (n_envs timesteps each) --
        # SB3 docs warn to divide by n_envs or checkpoints/evals land 8x less
        # often than the "50_000" naming implies (see train_mujoco.py, which
        # already divides for its CheckpointCallback after being bitten by
        # this once).
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=8, deterministic=True, render=False)

    callbacks = [
        RewardComponentCallback(log_interval=1000),
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1), save_path=ckpt_dir, name_prefix="go2_recovery"),
        VecNormSaveCallback(
            vec_env, ckpt_dir, save_freq=50_000,
            curriculum_path=curriculum_path),
        eval_callback,
    ]

    if args.resume:
        model = PPO.load(args.resume, env=vec_env)
        vn = args.vecnorm
        if vn is None:
            step_tok = os.path.basename(args.resume)
            # go2_recovery_500000_steps.zip → nearest vecnorm
            matches = sorted(glob.glob(os.path.join(ckpt_dir, "vecnorm_*.pkl")))
            if matches:
                vn = matches[-1]
        if vn and os.path.exists(vn):
            vec_env = VecNormalize.load(vn, vec_env.venv)
            vec_env.training = True
            vec_env.norm_reward = True
            model.set_env(vec_env)
            print(f"Loaded VecNormalize from {vn}")
        reset_timesteps = False
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            target_kl=0.03,
            verbose=1,
            tensorboard_log=log_dir,
            device="auto",
        )
        reset_timesteps = True

    # Sync eval obs norm from train env.
    eval_env.obs_rms = vec_env.obs_rms

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_timesteps,
        progress_bar=True,
    )

    final = os.path.join(log_dir, "go2_recovery_final.zip")
    model.save(final)
    vec_env.save(os.path.join(log_dir, "vecnorm_final.pkl"))
    level = float(np.mean(vec_env.get_attr("curriculum_level")))
    with open(curriculum_path, "w") as f:
        f.write(str(level))
    print(f"Saved {final} (curriculum_level={level:.3f})")


if __name__ == "__main__":
    main()
