"""Roll a checkpoint out, record an mp4 and break the reward down per term.

Answers "what is the policy being paid for, and what does it do right before it
falls": per-term mean reward/step, each term's share of the total, and the last
steps before every termination. Reward terms come from info["reward_components"].

    MUJOCO_GL=egl python3 training/analyze_rollout.py --model <zip> --scene flat \
        --cmd 0.2 --episodes 3 --video /tmp/walk.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import play_policy as pp  # noqa: E402  (before stable_baselines3, see eval_checkpoints.py)
import mujoco  # noqa: E402
from eval_checkpoints import LAYOUTS, _build, _find_vecnorm  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402


def _load(model_path, scene, cmd, wz=0.0):
    vn = _find_vecnorm(model_path)
    last = None
    for layout in LAYOUTS:
        try:
            env, core = _build(scene, cmd, layout, vn, wz)
            return env, core, PPO.load(model_path, env=env)
        except (AssertionError, ValueError) as e:
            last = e
    raise SystemExit(f"no obs layout fits {model_path}: {last}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scene", default="flat", choices=["flat", "stairs", "rough"])
    ap.add_argument("--cmd", type=float, default=0.2)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--video", default=None)
    ap.add_argument("--tail", type=int, default=8, help="steps before termination to print")
    args = ap.parse_args()

    env, core, model = _load(args.model, args.scene, args.cmd)
    writer = renderer = None
    if args.video:
        renderer = mujoco.Renderer(core.model, height=480, width=640)
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))
    every = max(1, int(1.0 / (pp.SIM_DT * pp.CTRL_DECIMATION)) // 25)

    totals, n_steps = {}, 0
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        hist = []
        for step in range(1, args.max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            comp = infos[0].get("reward_components", {})
            lin = core.data.sensor("lin_vel").data
            hist.append((step, float(reward[0]), dict(comp), float(core.data.qpos[2]),
                         float(lin[0]), float(np.abs(action).max())))
            for k, v in comp.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n_steps += 1
            if writer and step % every == 0:
                frame = pp._render_frame(renderer, core.data)
                frame = pp._draw_hud(frame, (args.cmd, 0, 0), lin, action[0], float(reward[0]), ep, step, 0)
                writer.write(frame)
            if done[0]:
                break
        fell = not infos[0].get("TimeLimit.truncated", False) and step < args.max_steps
        print(f"\nep {ep}: {step} steps, {'FELL' if fell else 'survived'}, "
              f"mean_vx={np.mean([h[4] for h in hist]):.2f} (cmd {args.cmd})")
        for s, r, comp, z, vx, amax in hist[-args.tail:]:
            top = sorted(comp.items(), key=lambda kv: -abs(kv[1]))[:4]
            print(f"  step {s:4d} r={r:+.2f} z={z:.2f} vx={vx:+.2f} max|a|={amax:.2f}  "
                  + " ".join(f"{k}={v:+.2f}" for k, v in top))

    print(f"\nreward terms, mean per step over {n_steps} steps (sorted by magnitude):")
    tot_abs = sum(abs(v) for v in totals.values()) or 1.0
    for k, v in sorted(totals.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<22}{v / n_steps:+9.4f}   {100 * abs(v) / tot_abs:5.1f}% of |total|")
    if writer:
        writer.release()
        renderer.close()
        print(f"\nvideo: {args.video}")


if __name__ == "__main__":
    main()
