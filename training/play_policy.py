"""
Play a trained Go2 MuJoCo policy in the headless OpenCV viewer.

Loads a PPO .zip checkpoint and optional VecNormalize .pkl stats, then runs
the policy in real-time with the same camera + HUD as headless_control.py.

Usage:
  python3 training/play_policy.py --model training/logs/mujoco/best_model.zip
  python3 training/play_policy.py --model best_model.zip --vecnorm vecnorm_final.pkl
  python3 training/play_policy.py --model best_model.zip --cmd 0.8 0 0 --record out.mp4

Controls:
  R    — reset episode
  ESC  — quit
"""

import argparse
import os
import sys
import time

# torch lazily imports triton when SB3 builds the Adam optimizer; a broken
# local triton/CUDA-driver combo segfaults there, so block the import.
sys.modules.setdefault("triton", None)

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.go2_mujoco_env import Go2MujocoEnv, SIM_DT, CTRL_DECIMATION


def _render_frame(renderer, data):
    body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = 2.0
    cam.elevation = -20.0
    if body_id >= 0:
        cam.lookat[:] = data.xpos[body_id]
    renderer.update_scene(data, camera=cam)
    return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def _draw_hud(frame, cmd, lin_vel, action, reward, episode, step, fps):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"cmd  vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}",
        f"vel  vx={lin_vel[0]:+.2f}  vy={lin_vel[1]:+.2f}  vz={lin_vel[2]:+.2f}",
        f"reward={reward:+.3f}  |act|={np.linalg.norm(action):.2f}  ep={episode}  step={step}  fps={fps:.0f}",
        "R reset | ESC quit",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, 22 + i * 22), font, 0.55, (0, 0, 0), 2)
        cv2.putText(frame, txt, (10, 22 + i * 22), font, 0.55, (200, 220, 255), 1)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True,
                        help="path to .zip checkpoint")
    parser.add_argument("--vecnorm", default=None,
                        help="VecNormalize .pkl (auto-detected from model dir if omitted)")
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                        metavar=("LIN_X", "LIN_Y", "ANG_YAW"))
    parser.add_argument("--record",  default=None,
                        help="output video path, e.g. out.mp4")
    parser.add_argument("--fps-render", type=int, default=30)
    args = parser.parse_args()

    cmd = tuple(args.cmd)
    raw = Go2MujocoEnv(cmd=cmd, render_mode=None,
                       randomize_domain=False, use_curriculum=False)
    env = DummyVecEnv([lambda: Monitor(raw)])

    # auto-detect vecnorm stats next to the checkpoint
    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        for candidate in [
            os.path.join(os.path.dirname(args.model), "vecnorm_final.pkl"),
            os.path.join(os.path.dirname(args.model), "..", "vecnorm_final.pkl"),
        ]:
            if os.path.exists(candidate):
                vecnorm_path = os.path.normpath(candidate)
                break

    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training    = False
        env.norm_reward = False
        print(f"VecNormalize stats: {vecnorm_path}")
    else:
        print("No VecNormalize stats found — running without obs normalisation")

    model = PPO.load(args.model, env=env)
    print(f"Model: {args.model}")
    print(f"cmd={cmd}")

    renderer = mujoco.Renderer(raw.model, height=480, width=640)
    cv2.namedWindow("Go2 Policy", cv2.WINDOW_AUTOSIZE)

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, args.fps_render, (640, 480))
        print(f"Recording → {args.record}")

    SIM_HZ       = int(1.0 / (SIM_DT * CTRL_DECIMATION))
    RENDER_EVERY = max(1, SIM_HZ // args.fps_render)

    obs = env.reset()
    episode = 1; step = 0; ep_reward = 0.0
    fps_display = 0.0; frame_count = 0; t0 = time.perf_counter()

    print("\nRunning policy — R to reset, ESC to quit\n")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        ep_reward += float(reward[0])
        step += 1

        if step % RENDER_EVERY == 0:
            now = time.perf_counter()
            frame_count += 1
            if now - t0 >= 1.0:
                fps_display = frame_count / (now - t0)
                frame_count = 0; t0 = now

            lin_vel = raw.data.sensor("lin_vel").data
            frame   = _render_frame(renderer, raw.data)
            frame   = _draw_hud(frame, np.array(cmd), lin_vel,
                                 action[0], float(reward[0]),
                                 episode, step, fps_display)
            if writer:
                writer.write(frame)
            cv2.imshow("Go2 Policy", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord('r'):
                obs = env.reset()
                step = 0; ep_reward = 0.0; episode += 1

        if done[0]:
            print(f"  ep {episode}  steps={step}  total_reward={ep_reward:.1f}")
            obs = env.reset()
            step = 0; ep_reward = 0.0; episode += 1

    cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"Saved {args.record}")
    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
