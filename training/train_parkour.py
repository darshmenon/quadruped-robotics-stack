"""
Train Go2 parkour / jump-landing policy in MuJoCo.

Curriculum: upward jump → forward → diagonal → over-obstacle hurdles.

Usage:
    python3 training/train_parkour.py
    python3 training/train_parkour.py --timesteps 1000000 --n_envs 8
    python3 training/train_parkour.py --resume training/logs/parkour/checkpoints/go2_parkour_500000_steps.zip
"""

import argparse
import glob
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.modules.setdefault("triton", None)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from envs.go2_mujoco_parkour_env import Go2MujocoParkourEnv
from intelligence.skills.agility_skills import AgilityCommand
from intelligence.skills.jump_curriculum import JumpStage, stage_for_level

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "parkour")
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")
CURRICULUM_PATH = os.path.join(LOG_DIR, "curriculum_level.txt")


class RewardComponentCallback(BaseCallback):
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self._sums: dict = {}
        self._n = 0
        self._interval = log_interval
        self._landing_err = 0.0
        self._landing_n = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            comps = info.get("reward_components")
            if comps:
                for k, v in comps.items():
                    self._sums[k] = self._sums.get(k, 0.0) + float(v)
                self._n += 1
            if "landing_error" in info:
                self._landing_err += float(info["landing_error"])
                self._landing_n += 1
        if self._n >= self._interval:
            for k, v in self._sums.items():
                self.logger.record(f"reward/{k}", v / self._n)
            if self._landing_n:
                self.logger.record("rollout/landing_error", self._landing_err / self._landing_n)
            level = float(np.mean(self.training_env.get_attr("curriculum_level")))
            self.logger.record("rollout/curriculum_level", level)
            self.logger.record("rollout/jump_stage", {
                JumpStage.UPWARD.value: 0,
                JumpStage.FORWARD.value: 1,
                JumpStage.DIAGONAL.value: 2,
                JumpStage.OVER_OBSTACLE.value: 3,
            }[stage_for_level(level).value])
            self._sums = {}
            self._n = 0
            self._landing_err = 0.0
            self._landing_n = 0
        return True


class VecNormSaveCallback(BaseCallback):
    def __init__(self, vec_env, save_path, save_freq, curriculum_path):
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


def make_env(rank, seed=0, curriculum_level=0.0):
    def _init():
        env = Go2MujocoParkourEnv(
            render_mode=None,
            randomize_domain=True,
            use_curriculum=True,
            initial_curriculum_level=curriculum_level,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--curriculum_level", type=float, default=None)
    parser.add_argument("--vecnorm", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)

    if args.curriculum_level is None:
        if os.path.exists(CURRICULUM_PATH):
            with open(CURRICULUM_PATH) as f:
                args.curriculum_level = float(f.read().strip())
            print(f"Loaded curriculum_level={args.curriculum_level:.3f}")
        else:
            args.curriculum_level = 0.0

    stage = stage_for_level(args.curriculum_level)
    print(
        f"Training Go2 parkour | envs={args.n_envs} | steps={args.timesteps} "
        f"| curriculum_level={args.curriculum_level:.3f} | stage={stage.value}"
    )

    vec_env = DummyVecEnv([
        make_env(i, curriculum_level=args.curriculum_level)
        for i in range(args.n_envs)
    ])
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_cmd = AgilityCommand(jump_height=0.15, landing_dx=0.35)
    eval_raw = Monitor(Go2MujocoParkourEnv(
        render_mode=None,
        randomize_domain=False,
        use_curriculum=False,
        initial_command=eval_cmd,
        initial_curriculum_level=min(args.curriculum_level, 0.6),
    ))
    eval_env = VecNormalize(
        DummyVecEnv([lambda: eval_raw]),
        norm_obs=True, norm_reward=False, training=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        # Both save_freq and eval_freq below count n_calls (n_envs timesteps
        # each) -- undivided they land 8x less often than "50_000" implies
        # (SB3 docs warn about exactly this).
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    callbacks = [
        RewardComponentCallback(log_interval=1000),
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1), save_path=CKPT_DIR, name_prefix="go2_parkour"),
        VecNormSaveCallback(vec_env, CKPT_DIR, 50_000, CURRICULUM_PATH),
        eval_callback,
    ]

    if args.resume:
        norm_path = args.vecnorm
        if norm_path is None:
            ckpt_stem = os.path.basename(args.resume).replace(".zip", "")
            try:
                ckpt_steps = int(ckpt_stem.split("_steps")[0].split("_")[-1])
            except ValueError:
                # best_model.zip / *_final.zip etc. -- no step count in the
                # name, so fall back to matching by save time instead below.
                ckpt_steps = None
            candidates = glob.glob(os.path.join(CKPT_DIR, "vecnorm_*_steps.pkl"))
            if candidates and ckpt_steps is not None:
                def _steps(p):
                    return int(os.path.basename(p).split("_steps")[0].split("_")[-1])
                norm_path = min(candidates, key=lambda p: abs(_steps(p) - ckpt_steps))
            elif candidates:
                resume_mtime = os.path.getmtime(args.resume)
                norm_path = min(candidates, key=lambda p: abs(os.path.getmtime(p) - resume_mtime))
        if norm_path and os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env.venv)
            vec_env.norm_reward = True
            print(f"Loaded VecNormalize stats from {norm_path}")
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=LOG_DIR)
        if args.learning_rate is not None:
            model.learning_rate = args.learning_rate
            model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.target_kl = 0.03
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
            ent_coef=0.01,
            max_grad_norm=1.0,
            target_kl=0.03,
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            tensorboard_log=LOG_DIR,
            verbose=1,
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=not bool(args.resume),
    )

    model.save(os.path.join(LOG_DIR, "go2_parkour_final"))
    vec_env.save(os.path.join(LOG_DIR, "vecnorm_final.pkl"))
    final_curriculum = float(np.mean(vec_env.get_attr("curriculum_level")))
    with open(CURRICULUM_PATH, "w") as f:
        f.write(str(final_curriculum))
    print(
        f"Done. curriculum_level={final_curriculum:.3f} "
        f"stage={stage_for_level(final_curriculum).value}  logs={LOG_DIR}"
    )
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
