"""
Train and compare a blind (proprioception-only) vs. sighted (height-scan)
Go2 locomotion policy on the same procedurally randomized rough terrain.

Usage:
    python3 training/train_vision_compare.py
    python3 training/train_vision_compare.py --timesteps 1000000 --n_envs 8
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")  # headless: avoid cv2/Qt plugin clash
sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault("triton", None)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.go2_mujoco_vision_env import Go2MujocoVisionEnv

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "vision_compare")


def make_env(cmd, use_vision, rank, seed=0):
    def _init():
        env = Go2MujocoVisionEnv(cmd=cmd, render_mode=None,
                                 randomize_domain=True, use_curriculum=True,
                                 use_vision=use_vision)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_one(tag, use_vision, cmd, n_envs, timesteps):
    run_dir = os.path.join(LOG_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env(cmd, use_vision, i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    eval_raw = Monitor(Go2MujocoVisionEnv(cmd=cmd, render_mode=None,
                                          randomize_domain=False,
                                          use_curriculum=False,
                                          use_vision=use_vision))
    eval_env = VecNormalize(DummyVecEnv([lambda: eval_raw]),
                            norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=n_envs * 128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        max_grad_norm=1.0,
        policy_kwargs=dict(net_arch=[512, 256, 128]),
        tensorboard_log=run_dir,
        verbose=1,
    )

    # CheckpointCallback/EvalCallback count in vectorized-rollout "calls",
    # i.e. env-steps / n_envs, not raw timesteps -- divide so eval/checkpoint
    # actually fire within a run instead of only after n_envs x freq steps.
    eval_freq = max(timesteps // (n_envs * 4), 1)
    callbacks = [
        CheckpointCallback(save_freq=eval_freq, save_path=run_dir, name_prefix=tag),
        EvalCallback(eval_env, best_model_save_path=run_dir, log_path=run_dir,
                    eval_freq=eval_freq, n_eval_episodes=10, deterministic=True,
                    render=False),
    ]

    print(f"\n=== training {tag} (use_vision={use_vision}) ===")
    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(os.path.join(run_dir, f"{tag}_final"))
    vec_env.save(os.path.join(run_dir, "vecnorm_final.pkl"))
    vec_env.close()
    eval_env.close()

    evaluations = os.path.join(run_dir, "evaluations.npz")
    if os.path.exists(evaluations):
        data = np.load(evaluations)
        return {
            "timesteps": data["timesteps"].tolist(),
            "mean_reward": data["results"].mean(axis=1).tolist(),
        }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="timesteps per policy (blind and sighted each get this many)")
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.4, 0.0, 0.0],
                        metavar=("LIN_X", "LIN_Y", "ANG_YAW"))
    args = parser.parse_args()
    cmd = tuple(args.cmd)

    os.makedirs(LOG_DIR, exist_ok=True)
    results = {}
    for tag, use_vision in (("blind", False), ("sighted", True)):
        results[tag] = train_one(tag, use_vision, cmd, args.n_envs, args.timesteps)

    print("\n=== eval mean-reward curves (rough terrain) ===")
    for tag, curve in results.items():
        if curve is None:
            print(f"{tag}: no evaluations.npz written")
            continue
        final = curve["mean_reward"][-1]
        print(f"{tag}: final eval mean reward = {final:.2f} "
              f"(over {len(curve['mean_reward'])} eval points)")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        for tag, curve in results.items():
            if curve is None:
                continue
            plt.plot(curve["timesteps"], curve["mean_reward"], label=tag)
        plt.xlabel("timesteps")
        plt.ylabel("eval mean reward (rough terrain)")
        plt.title("Blind vs. sighted Go2 locomotion on rough terrain")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(LOG_DIR, "blind_vs_sighted.png")
        plt.savefig(out_path, dpi=150)
        print(f"comparison plot written to {out_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot (raw data still in evaluations.npz)")


if __name__ == "__main__":
    main()
