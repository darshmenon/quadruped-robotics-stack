"""
Staged MuJoCo training: flat walk → rough blind → stairs.

Each stage resumes from the previous stage's final checkpoint + VecNormalize.
Inspired by go2-lab-rough-terrain-locomotion staged priors.

Usage:
    python3 training/train_curriculum.py
    python3 training/train_curriculum.py --stages flat rough --timesteps 500000
    python3 training/train_curriculum.py --asymmetric --obs-history 5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault("triton", None)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.go2_mujoco_env import Go2MujocoEnv
from envs.go2_mujoco_stairs_env import Go2MujocoStairsEnv
from envs.go2_mujoco_vision_env import Go2MujocoVisionEnv
from envs.obs_history import ObsHistoryWrapper
from envs.privileged_obs_wrapper import PrivilegedObsWrapper
from policies.asymmetric_mlp import AsymmetricActorCriticPolicy

ROOT = os.path.dirname(__file__)
LOG_ROOT = os.path.join(ROOT, "logs", "mujoco_curriculum")

STAGE_DEFAULT_STEPS = {
    "flat": 1_000_000,
    "rough": 1_000_000,
    "stairs": 500_000,
}


def _make_stage_env(stage: str, cmd, rank, seed, curriculum_level,
                    obs_history: int, asymmetric: bool, gait: bool):
    def _init():
        if stage == "flat":
            env = Go2MujocoEnv(
                cmd=cmd, render_mode=None, randomize_domain=True,
                use_curriculum=True, initial_curriculum_level=curriculum_level,
                gait_conditioned=gait, push_robots=True)
        elif stage == "rough":
            env = Go2MujocoVisionEnv(
                cmd=cmd, render_mode=None, randomize_domain=True,
                use_curriculum=True, use_vision=False,
                initial_curriculum_level=curriculum_level,
                gait_conditioned=gait)
        elif stage == "stairs":
            env = Go2MujocoStairsEnv(
                cmd=cmd, render_mode=None, randomize_domain=True,
                use_curriculum=True, use_vision=False,
                initial_curriculum_level=curriculum_level,
                gait_conditioned=gait)
        else:
            raise ValueError(f"unknown stage {stage}")
        if obs_history > 1:
            env = ObsHistoryWrapper(env, history_len=obs_history)
        if asymmetric:
            env = PrivilegedObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def _train_stage(stage: str, timesteps: int, n_envs: int, cmd, resume: str | None,
                 obs_history: int, asymmetric: bool, gait: bool,
                 curriculum_level: float) -> tuple[str, str]:
    log_dir = os.path.join(LOG_ROOT, stage)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    curriculum_path = os.path.join(log_dir, "curriculum_level.txt")

    print(f"\n=== Stage {stage} | steps={timesteps} | log={log_dir} ===")

    vec_env = DummyVecEnv([
        _make_stage_env(stage, cmd, i, 0, curriculum_level,
                        obs_history, asymmetric, gait)
        for i in range(n_envs)])

    policy_cls = AsymmetricActorCriticPolicy if asymmetric else "MlpPolicy"
    policy_kwargs = dict(net_arch=[512, 256, 128])

    if resume and os.path.exists(resume):
        # Load vecnorm from same dir as checkpoint when possible.
        ckpt_stem = os.path.basename(resume).replace(".zip", "")
        try:
            ckpt_steps = int(ckpt_stem.split("_steps")[0].split("_")[-1])
        except ValueError:
            # best_model.zip / *_final.zip etc. -- no step count in the
            # name, so fall back to matching by save time instead below.
            ckpt_steps = None
        step_candidates = glob.glob(os.path.join(
            os.path.dirname(resume), "vecnorm_*_steps.pkl"))
        norm_path = None
        if step_candidates and ckpt_steps is not None:
            def _steps(p):
                return int(os.path.basename(p).split("_steps")[0].split("_")[-1])
            norm_path = min(step_candidates, key=lambda p: abs(_steps(p) - ckpt_steps))
        elif step_candidates:
            resume_mtime = os.path.getmtime(resume)
            norm_path = min(step_candidates, key=lambda p: abs(os.path.getmtime(p) - resume_mtime))
        else:
            # No per-step vecnorm files (e.g. resuming from a previous
            # stage's go2_..._final / best_model checkpoint, which only
            # ever gets vecnorm_final.pkl) -- these aren't named
            # "*_steps.pkl" so _steps() above would crash on them; there's
            # normally just one, so no ranking needed.
            fallback = glob.glob(os.path.join(LOG_ROOT, "*", "vecnorm_final.pkl"))
            if fallback:
                norm_path = fallback[0]
        if norm_path and os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env.venv)
            vec_env.training = True
            vec_env.norm_reward = True
            print(f"Loaded VecNormalize from {norm_path}")
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                                   clip_obs=10.0, clip_reward=10.0)
            print("WARNING: no VecNormalize stats — fresh normalization")
        model = PPO.load(resume, env=vec_env, tensorboard_log=log_dir)
        model.target_kl = 0.03
        reset_timesteps = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0, clip_reward=10.0)
        model = PPO(
            policy_cls,
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=max(n_envs * 128, 256),
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            target_kl=0.03,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
        )
        reset_timesteps = True

    model.learn(
        total_timesteps=timesteps,
        callback=CheckpointCallback(
            save_freq=50_000, save_path=ckpt_dir, name_prefix=f"go2_{stage}"),
        reset_num_timesteps=reset_timesteps,
        progress_bar=True,
    )

    final = os.path.join(log_dir, f"go2_{stage}_final.zip")
    model.save(final)
    vec_env.save(os.path.join(log_dir, "vecnorm_final.pkl"))
    level = float(np.mean(vec_env.get_attr("curriculum_level")))
    with open(curriculum_path, "w") as f:
        f.write(str(level))
    vec_env.close()
    print(f"Stage {stage} done → {final} (curriculum={level:.3f})")
    return final, os.path.join(log_dir, "vecnorm_final.pkl")


def main():
    parser = argparse.ArgumentParser(description="Staged flat→rough→stairs training")
    parser.add_argument("--stages", nargs="+",
                        default=["flat", "rough", "stairs"],
                        choices=["flat", "rough", "stairs"])
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override per-stage step count")
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    parser.add_argument("--init-from", type=str, default=None,
                        help="optional .zip to seed the first stage")
    parser.add_argument("--asymmetric", action="store_true",
                        help="DreamWaQ-lite asymmetric critic (privileged lin_vel)")
    parser.add_argument("--obs-history", type=int, default=5,
                        help="proprio history frames (default 5 for terrain stages)")
    parser.add_argument("--gait", action="store_true")
    args = parser.parse_args()

    cmd = tuple(args.cmd)
    resume = args.init_from
    curriculum_level = 0.0

    for stage in args.stages:
        steps = args.timesteps or STAGE_DEFAULT_STEPS[stage]
        final, _vn = _train_stage(
            stage, steps, args.n_envs, cmd, resume,
            args.obs_history, args.asymmetric, args.gait, curriculum_level)
        resume = final
        # Carry curriculum forward between stages.
        cp = os.path.join(LOG_ROOT, stage, "curriculum_level.txt")
        if os.path.exists(cp):
            with open(cp) as f:
                curriculum_level = float(f.read().strip())

    print(f"\nCurriculum complete. Last checkpoint: {resume}")


if __name__ == "__main__":
    main()
