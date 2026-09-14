"""
Train Unitree Go2 walking policy using MuJoCo + Stable-Baselines3 PPO.

Usage:
    python3 training/train_mujoco.py
    python3 training/train_mujoco.py --timesteps 5000000 --cmd 1.0 0.0 0.0
    python3 training/train_mujoco.py --resume training/logs/mujoco/checkpoints/go2_mujoco_500000_steps.zip
"""

import argparse
import glob
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(__file__))

# torch lazily imports triton when SB3 builds the Adam optimizer; a broken
# local triton/CUDA-driver combo segfaults there, so block the import.
sys.modules.setdefault("triton", None)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from envs.go2_mujoco_env import Go2MujocoEnv
from envs.obs_history import ObsHistoryWrapper
from envs.privileged_obs_wrapper import PrivilegedObsWrapper
from policies.asymmetric_mlp import AsymmetricActorCriticPolicy

LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs", "mujoco")
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")
CURRICULUM_PATH = os.path.join(LOG_DIR, "curriculum_level.txt")


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
    """Saves VecNormalize running stats and curriculum_level alongside every
    model checkpoint, so an interrupted run (crash, OOM, preemption) doesn't
    leave curriculum_level.txt stuck at a stale value behind the checkpoint
    it'll actually be resumed from -- previously that file was only written
    once, after model.learn() returned normally."""

    def __init__(self, vec_env: VecNormalize, save_path: str, save_freq: int,
                 curriculum_path: str):
        super().__init__()
        self._vec_env         = vec_env
        self._save_path       = save_path
        self._save_freq       = save_freq
        self._curriculum_path = curriculum_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self._save_freq < self.training_env.num_envs:
            path = os.path.join(self._save_path,
                                f"vecnorm_{self.num_timesteps}_steps.pkl")
            self._vec_env.save(path)
            level = float(np.mean(self._vec_env.get_attr("curriculum_level")))
            with open(self._curriculum_path, "w") as f:
                f.write(str(level))
        return True


class FreezeObsNormCallback(BaseCallback):
    """Stops VecNormalize from updating its running obs/reward stats once
    num_timesteps crosses freeze_at.

    EvalCallback syncs the *live*, still-updating obs_rms from the training
    VecNormalize at every eval call, but a saved vecnorm_*_steps.pkl is a
    discrete snapshot -- replaying the same model checkpoint against a
    snapshot a few thousand steps off from the live-sync instant swung eval
    reward 10x (43 vs 936) in practice, because obs scaling directly shifts
    what the policy sees. Freezing stops the drift so any snapshot taken
    after freeze_at matches what live eval sees, and saved checkpoints
    become trustworthy again.
    """

    def __init__(self, vec_env: VecNormalize, freeze_at: int):
        super().__init__()
        self._vec_env = vec_env
        self._freeze_at = freeze_at
        self._frozen = False

    def _on_step(self) -> bool:
        if not self._frozen and self.num_timesteps >= self._freeze_at:
            self._vec_env.training = False
            self._frozen = True
            print(f"Froze VecNormalize obs/reward stats at num_timesteps={self.num_timesteps}")
        return True


# --------------------------------------------------------------------------- #
# Env factory
# --------------------------------------------------------------------------- #

def make_env(cmd, rank, seed=0, curriculum_level=0.0, gait_conditioned=False,
             obs_history=1, asymmetric=False, push_robots=True, legs_only=False):
    def _init():
        env = Go2MujocoEnv(cmd=cmd, render_mode=None,
                           randomize_domain=True, use_curriculum=True,
                           initial_curriculum_level=curriculum_level,
                           gait_conditioned=gait_conditioned,
                           push_robots=push_robots, legs_only=legs_only)
        if obs_history > 1:
            env = ObsHistoryWrapper(env, history_len=obs_history)
        if asymmetric:
            env = PrivilegedObsWrapper(env)
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
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="override the resumed model's learning rate (fresh runs use 3e-4)")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="override the resumed model's PPO epochs per update (fresh runs use 10)")
    parser.add_argument("--curriculum_level", type=float, default=None,
                        help="starting curriculum level (0-1); defaults to the value saved by "
                             "the previous run, or 0.0 if none was saved")
    parser.add_argument("--vecnorm", type=str, default=None,
                        help="path to a VecNormalize .pkl to load with --resume; "
                             "defaults to the nearest-matching checkpoint by step count")
    parser.add_argument("--gait", action="store_true",
                        help="enable gait-conditioned commands (trotting/bounding/pacing/"
                             "pronking clocks + contact-phase reward); uses a separate "
                             "log dir so existing checkpoints stay valid")
    parser.add_argument("--obs-history", type=int, default=1, metavar="N",
                        help="stack the last N proprio frames into the observation "
                             "(1 = disabled, recommended 5 for blind terrain inference)")
    parser.add_argument("--asymmetric", action="store_true",
                        help="DreamWaQ-lite: critic sees true lin_vel, actor proprio-only")
    parser.add_argument("--legs-only", action="store_true",
                        help="drop the arm+gripper from the action/observation space "
                             "(12 actions, 45-dim obs) so the resulting checkpoint "
                             "matches Go2GazeboEnv exactly and can be loaded straight "
                             "into eval_gazebo.py -- uses a separate log dir")
    parser.add_argument("--no-push", action="store_true",
                        help="disable mid-episode push-robot domain randomization -- "
                             "useful when resuming a checkpoint trained before pushes "
                             "were added, since the abrupt new disturbance + stale "
                             "VecNormalize reward stats can destabilize the resumed policy")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="override training log directory")
    parser.add_argument("--freeze-obs-at", type=int, default=None,
                        help="absolute total_timesteps at which to stop updating "
                             "VecNormalize's running obs/reward stats (see "
                             "FreezeObsNormCallback) -- makes eval and saved "
                             "checkpoints reproducible instead of drifting; leave "
                             "unset for the old always-updating behavior, e.g. early "
                             "in a fresh run when the state distribution is still "
                             "genuinely changing")
    args = parser.parse_args()

    log_dir = args.log_dir
    if log_dir is None:
        suffix_parts = []
        if args.legs_only:
            suffix_parts.append("legsonly")
        if args.gait:
            suffix_parts.append("gait")
        if args.obs_history > 1:
            suffix_parts.append(f"hist{args.obs_history}")
        if args.asymmetric:
            suffix_parts.append("asym")
        log_dir = LOG_DIR if not suffix_parts else (
            os.path.join(os.path.dirname(LOG_DIR), "mujoco_" + "_".join(suffix_parts)))
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    curriculum_path = os.path.join(log_dir, "curriculum_level.txt")

    os.makedirs(ckpt_dir, exist_ok=True)
    cmd = tuple(args.cmd)

    if args.curriculum_level is None:
        if os.path.exists(curriculum_path):
            with open(curriculum_path) as f:
                args.curriculum_level = float(f.read().strip())
            print(f"Loaded curriculum_level={args.curriculum_level:.3f} from {curriculum_path}")
        else:
            args.curriculum_level = 0.0

    print(f"Training Go2 (MuJoCo) | cmd={cmd} | envs={args.n_envs} | steps={args.timesteps} "
          f"| curriculum_level={args.curriculum_level:.3f} "
          f"| gait={args.gait} | obs_history={args.obs_history} "
          f"| asymmetric={args.asymmetric} | legs_only={args.legs_only} | log={log_dir}")

    # ---- training envs with obs + reward normalisation ----
    vec_env = DummyVecEnv([
        make_env(cmd, i, curriculum_level=args.curriculum_level,
                 gait_conditioned=args.gait, obs_history=args.obs_history,
                 asymmetric=args.asymmetric, push_robots=not args.no_push,
                 legs_only=args.legs_only)
        for i in range(args.n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    # ---- eval env: obs normalised (frozen), no reward norm, no domain rand ----
    def _make_eval():
        e = Go2MujocoEnv(cmd=cmd, render_mode=None,
                         randomize_domain=False, use_curriculum=False,
                         gait_conditioned=args.gait, push_robots=not args.no_push,
                         legs_only=args.legs_only)
        if args.obs_history > 1:
            e = ObsHistoryWrapper(e, history_len=args.obs_history)
        if args.asymmetric:
            e = PrivilegedObsWrapper(e)
        return Monitor(e)

    eval_env = VecNormalize(DummyVecEnv([_make_eval]),
                            norm_obs=True, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=50_000,
        # 5 episodes was too few once the robot's failure mode became
        # high-variance instead of consistently frozen -- a single lucky
        # long survival can drag a 5-episode mean up 10x (confirmed: eval
        # reported 2070 at one checkpoint, but a real 40-episode replay of
        # the same checkpoint averaged 7.6 with a 100% fall rate). 20 washes
        # that outlier luck out enough to trust eval_callback's own numbers
        # again without needing an external large-sample replay each time.
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    # EvalCallback always starts best_mean_reward=-inf, with no memory of a
    # previous run's actual best -- on --resume, its first eval was treated
    # as "new best" regardless of how it compared, silently overwriting
    # best_model.zip. This clobbered a real 462.97-reward checkpoint with a
    # transient 181 mid-resume (only recoverable because CheckpointCallback
    # happened to also save that same step as a numbered checkpoint). Seed
    # it from the previous run's own evaluations.npz so only a genuine
    # improvement can overwrite the saved checkpoint.
    prev_evals_path = os.path.join(log_dir, "evaluations.npz")
    if os.path.exists(prev_evals_path):
        prev_evals = np.load(prev_evals_path)
        if len(prev_evals["results"]) > 0:
            eval_callback.best_mean_reward = float(np.mean(prev_evals["results"], axis=1).max())
            print(f"Seeded EvalCallback.best_mean_reward="
                  f"{eval_callback.best_mean_reward:.2f} from {prev_evals_path}")

    callbacks = [
        RewardComponentCallback(log_interval=1000),
        # SB3's CheckpointCallback counts n_calls (one per vectorized step),
        # not total timesteps -- without dividing by n_envs it needs 8x the
        # intended 50k timesteps between saves (SB3 docs warn about exactly
        # this). Combined with n_calls resetting to 0 on every --resume, a
        # crashed session shorter than 50_000 * n_envs steps saved zero model
        # checkpoints despite VecNormSaveCallback (which is num_timesteps-
        # based, see below) saving fine the whole time -- lost ~260k steps
        # of real progress to a mid-run CUDA crash before this was caught.
        CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1), save_path=ckpt_dir,
                           name_prefix="go2_mujoco"),
        VecNormSaveCallback(vec_env, ckpt_dir, save_freq=50_000,
                            curriculum_path=curriculum_path),
        eval_callback,
    ]
    if args.freeze_obs_at is not None:
        callbacks.append(FreezeObsNormCallback(vec_env, args.freeze_obs_at))

    if args.resume:
        norm_path = args.vecnorm
        if norm_path is None:
            # find the vecnorm_<steps>_steps.pkl whose step count is closest
            # to the checkpoint being resumed (exact match is rare, since
            # CheckpointCallback and VecNormSaveCallback save on independent
            # step counters)
            ckpt_stem   = os.path.basename(args.resume).replace(".zip", "")
            try:
                ckpt_steps = int(ckpt_stem.split("_steps")[0].split("_")[-1])
            except ValueError:
                # Resuming from best_model.zip / go2_mujoco_final.zip etc. --
                # no step count in the name to match against, fall through
                # to the "no match found" warning below instead of crashing.
                ckpt_steps = -1
            candidates  = glob.glob(os.path.join(ckpt_dir, "vecnorm_*_steps.pkl"))
            if candidates and ckpt_steps >= 0:
                def _steps(p):
                    return int(os.path.basename(p).split("_steps")[0].split("_")[-1])
                norm_path = min(candidates, key=lambda p: abs(_steps(p) - ckpt_steps))
        if norm_path and os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env.venv)
            vec_env.norm_reward = True
            print(f"Loaded VecNormalize stats from {norm_path}")
        else:
            print("WARNING: no VecNormalize stats found to load — starting with fresh "
                  "obs/reward normalization, which will destabilize early fine-tuning.")
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=log_dir)
        if args.learning_rate is not None:
            model.learning_rate = args.learning_rate
            model.lr_schedule = get_schedule_fn(args.learning_rate)
            print(f"Overrode learning_rate={args.learning_rate}")
        if args.n_epochs is not None:
            model.n_epochs = args.n_epochs
            print(f"Overrode n_epochs={args.n_epochs}")
        # checkpoints saved before this fix have target_kl=None baked in from
        # PPO.load(); reapply it so resumed runs get the same early-stopping
        # protection as fresh ones (see comment on the fresh-init PPO() call).
        model.target_kl = 0.03
        print(f"Resumed from {args.resume}")
    else:
        policy_cls = AsymmetricActorCriticPolicy if args.asymmetric else "MlpPolicy"
        model = PPO(
            policy_cls,
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
            # Without this, a rollout batch runs all 10 epochs regardless of
            # how far the policy has already drifted -- this run's log shows
            # approx_kl climbing 0.017 -> 0.10 and clip_fraction 0.2 -> 0.6
            # over 3M steps with no ceiling, and eval reward collapsed from
            # 458 (peak, ~800k steps) to ~120 by 1.6M and never recovered.
            # target_kl stops epoch iteration early once a batch's KL exceeds
            # this, capping how far a single update can drag the policy.
            target_kl=0.03,
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            tensorboard_log=log_dir,
            verbose=1,
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=not bool(args.resume),
    )

    model.save(os.path.join(log_dir, "go2_mujoco_final"))
    vec_env.save(os.path.join(log_dir, "vecnorm_final.pkl"))

    final_curriculum = float(np.mean(vec_env.get_attr("curriculum_level")))
    with open(curriculum_path, "w") as f:
        f.write(str(final_curriculum))
    print(f"Final curriculum_level={final_curriculum:.3f} saved to {curriculum_path}")

    print("Training done. Model saved to", log_dir)
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
