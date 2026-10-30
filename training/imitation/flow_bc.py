speaking_modes_table_vs_net_vs_humanized"""Flow-matching behavior cloning from NMPC demos (collect_nmpc.py), with a MuJoCo rollout check.

Learns a velocity field v(a_t, t | obs) that transports Gaussian noise to the demo action
(conditional optimal-transport path: a_t = (1-t) z + t a, target v = a - z). Unlike MSE BC it
doesn't average across modes of the demonstrations (e.g. which diagonal pair leads the trot).

    python3 training/imitation/flow_bc.py --data training/imitation/data/nmpc_run*.npz \
        --out training/logs/flow_bc --epochs 200 --eval-episodes 3

Acting: integrate noise -> action with --steps Euler steps (default 8); `--det` starts from
zero noise so rollouts are repeatable. Obs layout is the legs-only 45-d frame with a 5-frame
history (225), normalised with the demo mean/std saved next to the weights.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import play_policy as pp  # noqa: E402  (imports the MuJoCo env before torch-heavy libs)

ACT_DIM = 12


class FlowPolicy(nn.Module):
    def __init__(self, obs_dim, hidden=512, act_dim=ACT_DIM):
        super().__init__()
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 256), nn.SiLU(),
            nn.Linear(256, act_dim))

    def velocity(self, obs, a_t, t):
        return self.net(torch.cat([obs, a_t, t], dim=-1))

    @torch.no_grad()
    def act(self, obs, steps=8, det=True):
        z = torch.zeros(obs.shape[0], self.act_dim) if det else torch.randn(obs.shape[0], self.act_dim)
        a = z
        for i in range(steps):
            t = torch.full((obs.shape[0], 1), i / steps)
            a = a + self.velocity(obs, a, t) / steps
        return a.clamp(-1.0, 1.0)


def _history(obs, h):
    pad = np.concatenate([np.repeat(obs[:1], h - 1, axis=0), obs])
    return np.concatenate([pad[i:i + len(obs)] for i in range(h)], axis=1)


def rollout(policy, mean, std, steps, episodes, max_steps, cmd, det):
    env = pp._make_env("flat", (cmd, 0.0, 0.0), use_vision=False, obs_history=5,
                       asymmetric=False, legs_only=True)
    core = env
    while hasattr(core, "env"):
        core = core.env
    out = []
    for _ in range(episodes):
        obs, _ = env.reset()
        x0, y0 = pp._base_xy(core)
        fell, n = False, 0
        for n in range(1, max_steps + 1):
            o = torch.tensor(np.clip((obs - mean) / std, -10, 10), dtype=torch.float32)[None]
            a = policy.act(o, steps=steps, det=det)[0].numpy()
            obs, _, term, trunc, _ = env.step(a)
            if term:
                fell = True
                break
            if trunc:
                break
        x1, y1 = pp._base_xy(core)
        out.append((n, float(np.hypot(x1 - x0, y1 - y0)), fell))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=8, help="Euler steps when acting")
    ap.add_argument("--det", action="store_true", help="zero initial noise (repeatable rollouts)")
    ap.add_argument("--eval-episodes", type=int, default=0)
    ap.add_argument("--eval-cmd", type=float, default=0.3)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    X, Y = [], []
    for path in sorted(p for pat in args.data for p in glob.glob(pat)):
        d = np.load(path)
        X.append(_history(d["obs"], 5))
        Y.append(np.clip(d["act"], -1.0, 1.0))
    X, Y = np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)
    mean, std = X.mean(0), X.std(0) + 1e-3
    Xn = torch.tensor(np.clip((X - mean) / std, -10, 10))
    Yt = torch.tensor(Y)
    print(f"{len(X)} samples, obs {X.shape[1]}, act {Y.shape[1]}")

    policy = FlowPolicy(X.shape[1])
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n_val = max(1, len(X) // 10)
    perm = torch.randperm(len(X))
    va, tr = perm[:n_val], perm[n_val:]

    def fm_loss(idx):
        a, o = Yt[idx], Xn[idx]
        z, t = torch.randn_like(a), torch.rand(len(idx), 1)
        return nn.functional.mse_loss(policy.velocity(o, (1 - t) * z + t * a, t), a - z)

    for ep in range(1, args.epochs + 1):
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), args.batch):
            loss = fm_loss(p[i:i + args.batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep == 1 or ep % max(1, args.epochs // 5) == 0:
            with torch.no_grad():
                v = fm_loss(va).item()
                err = nn.functional.mse_loss(policy.act(Xn[va], args.steps, det=False), Yt[va]).item()
            print(f"epoch {ep:4d}  train {loss.item():.4f}  val_flow {v:.4f}  val_action_mse {err:.4f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save(dict(model=policy.state_dict(), mean=mean, std=std, obs_dim=X.shape[1]),
               os.path.join(args.out, "flow_bc.pt"))
    print("saved", os.path.join(args.out, "flow_bc.pt"))

    if args.eval_episodes:
        res = rollout(policy, mean, std, args.steps, args.eval_episodes, args.max_steps, args.eval_cmd, args.det)
        for i, (n, dist, fell) in enumerate(res, 1):
            print(f"  eval ep {i}: steps={n} dist={dist:.2f}m fell={fell}")
        print(f"fall rate {100 * np.mean([r[2] for r in res]):.0f}%  mean dist {np.mean([r[1] for r in res]):.2f} m")


if __name__ == "__main__":
    main()
