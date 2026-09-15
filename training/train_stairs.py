"""
Train Go2 stair / ledge climbing in MuJoCo (fixed course + optional height scan).

Usage:
    python3 training/train_stairs.py
    python3 training/train_stairs.py --timesteps 500000 --n_envs 4
    python3 training/train_stairs.py --blind          # proprioception only
    python3 training/train_stairs.py --resume training/logs/stairs/checkpoints/go2_stairs_100000_steps.zip
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.go2_mujoco_stairs_env import Go2MujocoStairsEnv

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "stairs")
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")
CURRICULUM_PATH = os.path.join(LOG_DIR, "curriculum_level.txt")


class RewardComponentCallback(BaseCallback):
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self._sums: dict = {}
        self._n = 0
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
            xs = [float(i.get("x", 0.0)) for i in self.locals["infos"] if "x" in i]
            if xs:
                self.logger.record("rollout/mean_x", float(np.mean(xs)))
            self._sums = {}
            self._n = 0
        return True


class VecNormSaveCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq: int, curriculum_path: str):
        super().__init__()
        self._save_path = save_path
        self._save_freq = save_freq
        self._curriculum_path = curriculum_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self._save_freq < self.training_env.num_envs:
            # Fetched fresh every call via SB3's own accessor, not captured
            # once at construction -- a captured reference stays pointing at
            # whatever VecNormalize existed before any --resume reassignment
            # replaced it, which is never stepped again and so saves frozen
            # default stats (count=1e-4, mean=0, var=1) forever instead of
            # the real, live-accumulating normalization (see train_mujoco.py
            # for the full incident this was ported from).
            vec_env = self.model.get_vec_normalize_env()
            path = os.path.join(
                self._save_path, f"vecnorm_{self.num_timesteps}_steps.pkl")
            vec_env.save(path)
            level = float(np.mean(vec_env.get_attr("curriculum_level")))
            with open(self._curriculum_path, "w") as f:
                f.write(str(level))
        return True


def make_env(cmd, rank, seed=0, curriculum_level=0.0, use_vision=True):
    def _init():
        env = Go2MujocoStairsEnv(
            cmd=cmd,
            render_mode=None,
            randomize_domain=True,
            use_curriculum=True,
            use_vision=use_vision,
            initial_curriculum_level=curriculum_level,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.4, 0.0, 0.0],
                        metavar=("LIN_X", "LIN_Y", "ANG_YAW"))
    parser.add_argument(
        "--blind", action="store_true",
        help="Proprioception only (no height-scan). Required to resume from flat walk SB3.",
    )
    parser.add_argument(
        "--init-from-flat",
        action="store_true",
        help="Seed from training/logs/mujoco/best_model.zip (blind stairs only; obs dim match).",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--curriculum_level", type=float, default=None)
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument(
        "--device", default="auto",
        help="SB3 device: auto | cpu | cuda (force cpu if no NVIDIA driver)",
    )
    args = parser.parse_args()

    use_vision = not args.blind
    if args.init_from_flat and use_vision:
        raise SystemExit("--init-from-flat requires --blind (flat walk is 76-dim; sighted stairs is 94-dim)")

    # Prefer CPU when CUDA is broken / unavailable (common on this box).
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "" if device == "cpu" else os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    os.makedirs(CKPT_DIR, exist_ok=True)

    if args.curriculum_level is None:
        if os.path.exists(CURRICULUM_PATH):
            with open(CURRICULUM_PATH) as f:
                args.curriculum_level = float(f.read().strip())
            print(f"Loaded curriculum_level={args.curriculum_level:.3f}")
        else:
            args.curriculum_level = 0.0

    tag = "blind" if args.blind else "sighted"
    print(
        f"Training Go2 stairs ({tag}) | envs={args.n_envs} | steps={args.timesteps} "
        f"| curriculum_level={args.curriculum_level:.3f}"
    )

    cmd = tuple(args.cmd)
    vec_env = DummyVecEnv([
        make_env(cmd, i, curriculum_level=args.curriculum_level, use_vision=use_vision)
        for i in range(args.n_envs)
    ])
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_raw = Monitor(Go2MujocoStairsEnv(
        cmd=cmd,
        render_mode=None,
        randomize_domain=False,
        use_curriculum=False,
        use_vision=use_vision,
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
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=CKPT_DIR,
            name_prefix="go2_stairs",
        ),
        VecNormSaveCallback(
            CKPT_DIR, save_freq=50_000, curriculum_path=CURRICULUM_PATH),
        eval_callback,
    ]

    if args.resume or args.init_from_flat:
        resume_path = args.resume
        if args.init_from_flat and not resume_path:
            resume_path = os.path.join(
                os.path.dirname(__file__), "logs", "mujoco", "best_model.zip")
            if not os.path.exists(resume_path):
                raise SystemExit(f"--init-from-flat: missing {resume_path}")
            # Seed stairs VecNormalize from flat walk stats when present.
            if args.vecnorm is None:
                flat_vn = os.path.join(
                    os.path.dirname(__file__), "logs", "mujoco", "vecnorm_final.pkl")
                if os.path.exists(flat_vn):
                    args.vecnorm = flat_vn

        norm_path = args.vecnorm
        if norm_path is None and args.resume:
            ckpt_stem = os.path.basename(resume_path).replace(".zip", "")
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
                resume_mtime = os.path.getmtime(resume_path)
                norm_path = min(candidates, key=lambda p: abs(os.path.getmtime(p) - resume_mtime))
        if norm_path and os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env.venv)
            vec_env.training = True
            vec_env.norm_reward = True
            print(f"Loaded VecNormalize stats from {norm_path}")
        model = PPO.load(
            resume_path, env=vec_env, tensorboard_log=LOG_DIR, device=device)
        # Fine-tune stairs a bit gentler when seeding from flat walk.
        lr = args.learning_rate if args.learning_rate is not None else (
            1e-4 if args.init_from_flat else None)
        if lr is not None:
            model.learning_rate = lr
            model.lr_schedule = get_schedule_fn(lr)
        model.target_kl = 0.03
        print(f"Resumed from {resume_path}  device={device}  lr={model.learning_rate}")
        reset_ts = not bool(args.resume)  # init-from-flat starts a new stairs counter
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
            ent_coef=0.008,
            max_grad_norm=1.0,
            target_kl=0.03,
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            tensorboard_log=LOG_DIR,
            verbose=1,
            device=device,
        )
        reset_ts = True

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_ts,
    )

    model.save(os.path.join(LOG_DIR, f"go2_stairs_{tag}_final"))
    vec_env.save(os.path.join(LOG_DIR, "vecnorm_final.pkl"))
    final_curriculum = float(np.mean(vec_env.get_attr("curriculum_level")))
    with open(CURRICULUM_PATH, "w") as f:
        f.write(str(final_curriculum))
    print(f"Done. curriculum_level={final_curriculum:.3f}  logs={LOG_DIR}")
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
