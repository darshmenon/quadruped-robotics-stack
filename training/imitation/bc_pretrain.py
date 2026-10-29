"""Behavior-clone NMPC (obs, action) data into the PPO actor, then hand off to PPO.

    python3 training/imitation/bc_pretrain.py --data training/imitation/data/nmpc_run*.npz \
        --out training/logs/mujoco_curriculum/bc_init

writes <out>/go2_flat_bc.zip + <out>/vecnorm_final.pkl (obs stats fitted on the demos),
which `train_curriculum.py --stages flat --asymmetric --obs-history 5 --init-from <zip>`
resumes from (it clamps action std to <=0.5 and uses RESUME_LR on resume).

The policy sees the same 5-frame history the env's ObsHistoryWrapper builds
(oldest -> newest, first frame repeated at episode start).
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import train_curriculum as tc  # noqa: E402  (imports mujoco env before stable_baselines3)
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402


def _history(obs, h):
    """(N, 45) -> (N, h*45), repeating frame 0 for the first h-1 rows."""
    pad = np.concatenate([np.repeat(obs[:1], h - 1, axis=0), obs])
    return np.concatenate([pad[i:i + len(obs)] for i in range(h)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs-history", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-speed", type=float, default=0.0,
                    help="drop frames whose hindsight |vx cmd| is below this (m/s); 0 keeps standing frames")
    ap.add_argument("--init-std", type=float, default=0.3)
    args = ap.parse_args()

    X, Y = [], []
    for path in sorted(p for pat in args.data for p in glob.glob(pat)):
        d = np.load(path)
        X.append(_history(d["obs"], args.obs_history))
        Y.append(np.clip(d["act"], -1.0, 1.0))
        print(f"{path}: {len(d['obs'])} frames")
    X, Y = np.concatenate(X), np.concatenate(Y)
    if args.min_speed > 0:
        keep = np.abs(X[:, -45 + 6] / 2.0) >= args.min_speed  # newest frame's vx cmd (scaled by 2)
        X, Y = X[keep], Y[keep]
    print(f"BC set: {len(X)} samples, obs {X.shape[1]}, act {Y.shape[1]}")

    # Same env stack and model as train_curriculum's fresh flat stage.
    vec = DummyVecEnv([tc._make_stage_env("flat", (0.3, 0.0, 0.0), 0, 0, 0.0,
                                          args.obs_history, True, False)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    mean, var = X.mean(0), X.var(0) + 1e-6
    for key, rms in vec.obs_rms.items():
        n = rms.mean.shape[0]  # 'policy' = obs; 'critic' = obs + privileged lin_vel(3)
        rms.mean = np.concatenate([mean, np.zeros(n - len(mean))])
        rms.var = np.concatenate([var, np.ones(n - len(var))])
        rms.count = float(len(X))
    model = PPO(tc.AsymmetricActorCriticPolicy, vec, learning_rate=3e-4, n_steps=2048,
                batch_size=max(8 * 128, 256), target_kl=0.03, ent_coef=0.0, verbose=0,
                policy_kwargs=dict(net_arch=[512, 256, 128], log_std_init=tc.LOG_STD_INIT))
    pol = model.policy
    with torch.no_grad():
        pol.log_std.fill_(math.log(args.init_std))

    Xn = torch.tensor(np.clip((X - mean) / np.sqrt(var), -10, 10), dtype=torch.float32, device=pol.device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=pol.device)
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    n_val = max(1, len(X) // 10)
    perm = torch.randperm(len(X))
    va, tr = perm[:n_val], perm[n_val:]
    for ep in range(1, args.epochs + 1):
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), args.batch):
            idx = p[i:i + args.batch]
            loss = torch.nn.functional.mse_loss(pol.action_net(pol.mlp_extractor.policy_net(Xn[idx])), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == 1:
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(pol.action_net(pol.mlp_extractor.policy_net(Xn[va])), Yt[va])
            print(f"epoch {ep:3d}  train {loss.item():.4f}  val {v.item():.4f}  (var of targets {Yt.var().item():.4f})")

    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, "go2_flat_bc.zip")
    model.save(zip_path)
    vec.save(os.path.join(args.out, "vecnorm_final.pkl"))
    print("saved", zip_path)


if __name__ == "__main__":
    main()
