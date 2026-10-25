"""
Play a trained Go2 MuJoCo policy in the headless OpenCV viewer.

Loads a PPO .zip checkpoint and optional VecNormalize .pkl stats, then runs
the policy in real-time with the same camera + HUD as headless_control.py.

Usage:
  python3 training/play_policy.py --model training/logs/mujoco/best_model.zip
  python3 training/play_policy.py --model best_model.zip --vecnorm vecnorm_final.pkl
  python3 training/play_policy.py --model best_model.zip --cmd 0.8 0 0 --record out.mp4
  python3 training/play_policy.py --model training/logs/stairs/best_model.zip --scene stairs

  # Headless smoke test (no window; prints mean reward / distance / fall rate)
  python3 training/play_policy.py --model training/logs/mujoco/best_model.zip \
    --no-display --episodes 3 --cmd 0.5 0 0

Controls (GUI mode):
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
from envs.go2_mujoco_stairs_env import Go2MujocoStairsEnv
from envs.go2_mujoco_vision_env import Go2MujocoVisionEnv
from envs.obs_history import ObsHistoryWrapper
from envs.privileged_obs_wrapper import PrivilegedObsWrapper


def _make_env(scene: str, cmd, use_vision: bool, gait_conditioned=False,
              gait_name="trotting", obs_history=1, asymmetric=False,
              legs_only=False):
    kwargs = dict(
        cmd=cmd, render_mode=None, randomize_domain=False,
        use_curriculum=False, gait_conditioned=gait_conditioned,
        gait_name=gait_name,
    )
    if scene == "stairs":
        env = Go2MujocoStairsEnv(**kwargs, use_vision=use_vision, legs_only=legs_only)
    elif scene == "rough":
        env = Go2MujocoVisionEnv(**kwargs, use_vision=use_vision)
    else:
        env = Go2MujocoEnv(**kwargs, legs_only=legs_only)
    if obs_history > 1:
        env = ObsHistoryWrapper(env, history_len=obs_history)
    if asymmetric:
        env = PrivilegedObsWrapper(env)
    return env


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


def _draw_hud(frame, cmd, lin_vel, action, reward, episode, step, fps, gait=None):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"cmd  vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}",
        f"vel  vx={lin_vel[0]:+.2f}  vy={lin_vel[1]:+.2f}  vz={lin_vel[2]:+.2f}",
        f"reward={reward:+.3f}  |act|={np.linalg.norm(action):.2f}  ep={episode}  step={step}  fps={fps:.0f}",
    ]
    if gait is not None:
        lines.append(f"gait={gait}")
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        y += 20
    return frame


def _base_xy(core):
    try:
        return float(core.data.qpos[0]), float(core.data.qpos[1])
    except Exception:
        return 0.0, 0.0


def _run_eval(env, model, core, episodes: int, max_steps: int):
    """Headless smoke test: no OpenCV window."""
    results = []
    for ep in range(1, episodes + 1):
        obs = env.reset()
        x0, y0 = _base_xy(core)
        ep_reward = 0.0
        steps = 0
        fell = False
        x1, y1 = x0, y0
        for steps in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            # DummyVecEnv auto-resets on done, so read the pose only while the
            # episode is still running or the final position is the reset pose.
            if not done[0]:
                x1, y1 = _base_xy(core)
            ep_reward += float(reward[0])
            if done[0]:
                info = infos[0] if infos else {}
                # Short episode before timeout ⇒ likely fall / early terminate
                if steps < max_steps * 0.9 and not info.get("TimeLimit.truncated", False):
                    fell = True
                break
        dist = float(np.hypot(x1 - x0, y1 - y0))
        mean_vx = dist / max(steps * SIM_DT * CTRL_DECIMATION, 1e-6)
        results.append(
            dict(ep=ep, steps=steps, reward=ep_reward, dist=dist, mean_vx=mean_vx, fell=fell)
        )
        print(
            f"  ep {ep}: steps={steps}  reward={ep_reward:.1f}  "
            f"dist={dist:.2f}m  mean_vx≈{mean_vx:.2f}  fell={fell}"
        )

    n = len(results)
    mean_r = sum(r["reward"] for r in results) / n
    mean_d = sum(r["dist"] for r in results) / n
    fall_rate = sum(1 for r in results if r["fell"]) / n
    mean_vx = sum(r["mean_vx"] for r in results) / n
    ok = mean_vx > 0.05 and fall_rate < 1.0
    print(
        f"\nSUMMARY  episodes={n}  mean_reward={mean_r:.1f}  "
        f"mean_dist={mean_d:.2f}m  mean_vx≈{mean_vx:.2f}  fall_rate={fall_rate:.0%}"
    )
    print("RESULT:", "PASS (policy moving forward)" if ok else "WEAK/FAIL (little motion or always falling)")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True,
                        help="path to SB3 PPO .zip checkpoint (not Isaac .pt)")
    parser.add_argument("--vecnorm", default=None,
                        help="VecNormalize .pkl (auto-detected from model dir if omitted)")
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                        metavar=("LIN_X", "LIN_Y", "ANG_YAW"))
    parser.add_argument("--record",  default=None,
                        help="output video path, e.g. out.mp4")
    parser.add_argument("--fps-render", type=int, default=30)
    parser.add_argument(
        "--scene", choices=("flat", "stairs", "rough"), default="flat",
        help="MuJoCo world: flat ground, stairs/ledges course, or rough heightfield",
    )
    parser.add_argument(
        "--blind", action="store_true",
        help="For stairs/rough: proprioception only (no height-scan). Match training.",
    )
    parser.add_argument("--gait", action="store_true",
                        help="match a gait-conditioned checkpoint")
    parser.add_argument("--gait-name", default="trotting",
                        choices=["trotting", "bounding", "pacing", "pronking"])
    parser.add_argument("--obs-history", type=int, default=1, metavar="N",
                        help="must match the history length used at train time")
    parser.add_argument("--legs-only", action="store_true",
                        help="12-dim legs-only policy (curriculum checkpoints); "
                             "the rough scene is always legs-only")
    parser.add_argument("--asymmetric", action="store_true",
                        help="wrap with PrivilegedObsWrapper (DreamWaQ-lite checkpoints)")
    parser.add_argument(
        "--no-display", action="store_true",
        help="Headless: no OpenCV window (use with --episodes and/or --record)",
    )
    parser.add_argument(
        "--episodes", type=int, default=0,
        help="If >0, run N eval episodes then exit (prints PASS/FAIL summary)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="Max steps per eval episode (default 1000 ≈ 20s)",
    )
    args = parser.parse_args()

    if args.model.endswith(".pt"):
        print(
            "ERROR: this script loads Stable-Baselines3 .zip checkpoints only.\n"
            "Isaac/rsl_rl .pt files under training/pretrained/ need their upstream "
            "Isaac Gym/Lab play scripts — see docs/PRETRAINED.md.",
            file=sys.stderr,
        )
        return 2

    cmd = tuple(args.cmd)
    use_vision = not args.blind
    raw = _make_env(args.scene, cmd, use_vision=use_vision,
                    gait_conditioned=args.gait, gait_name=args.gait_name,
                    obs_history=args.obs_history, asymmetric=args.asymmetric,
                    legs_only=args.legs_only)
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
    print(f"scene={args.scene}  vision={use_vision and args.scene != 'flat'}  "
          f"gait={args.gait}  obs_history={args.obs_history}  "
          f"asymmetric={args.asymmetric}  cmd={cmd}")

    # Unwrap to the MuJoCo env for sensors / renderer.
    core = raw
    while hasattr(core, "env"):
        core = core.env

    if args.episodes > 0:
        return _run_eval(env, model, core, args.episodes, args.max_steps)

    renderer = mujoco.Renderer(core.model, height=480, width=640)
    if not args.no_display:
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

    mode = "record-only" if args.no_display else "GUI"
    print(f"\nRunning policy ({mode}) — R to reset, ESC to quit\n")

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

            lin_vel = core.data.sensor("lin_vel").data
            gait = getattr(core, "gait_cmd", None) if args.gait else None
            frame   = _render_frame(renderer, core.data)
            frame   = _draw_hud(frame, np.array(cmd), lin_vel,
                                 action[0], float(reward[0]),
                                 episode, step, fps_display, gait=gait)
            if writer:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Go2 Policy", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord('r'):
                    obs = env.reset()
                    step = 0; ep_reward = 0.0; episode += 1
            elif args.no_display and step > args.max_steps:
                break

        if done[0]:
            print(f"  ep {episode}  steps={step}  total_reward={ep_reward:.1f}")
            obs = env.reset()
            step = 0; ep_reward = 0.0; episode += 1
            if args.no_display and writer is None:
                break

    if not args.no_display:
        cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"Saved {args.record}")
    renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
