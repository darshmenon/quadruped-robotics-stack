#!/usr/bin/env python3
"""Open MuJoCo's interactive viewer, run an RL policy in it in real time, and screen-record that window.

    # walking
    GO2_TROT_PRIOR=1 GO2_WIDE_CMD=1 python3 scripts/record_go2_mujoco_gui.py --seconds 14 --out docs/images/go2_mujoco_window.gif

    # climbing four 5 cm steps with the height-scan climber, up to 6 live attempts (each recorded to <out>_<k>.mp4)
    GO2_NO_STAND_SUBSIDY=1 GO2_TROT_PRIOR=1 GO2_WIDE_CMD=1 GO2_LIFT_SCALE=2.0 GO2_STAIRS_VISION=1 \
    GO2_STAIRS_SCENE=training/envs/go2_steps_h05.xml python3 scripts/record_go2_mujoco_gui.py --climb \
        --model training/pretrained/go2_climb_scan/go2_stairs_h05.zip --attempts 6 --out docs/images/go2_mujoco_climb.gif

The window is the same kind of live view as the Gazebo GUI: a tracking camera follows the robot while the policy runs
with vx plus a yaw P-controller on heading. ffmpeg captures only the viewer window (x11grab, X11 session), starting when
the robot starts moving. With --climb an attempt ends when the robot is on the top landing, falls or stalls; the GIF is
made from the first successful attempt (or the furthest one if none succeed) and the summary says which. Needs $DISPLAY.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.modules.setdefault("tensorflow", None)  # SB3 -> tensorboard imports all of TensorFlow (minutes under load); unused here


def find_window(title_part="MuJoCo", tries=40):
    for _ in range(tries):
        tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True).stdout
        best, wid = 0, None
        for line in tree.splitlines():
            if title_part in line and line.strip().startswith("0x"):
                m = re.search(r"(\d+)x(\d+)\+", line)
                if m and int(m.group(1)) * int(m.group(2)) > best:
                    best, wid = int(m.group(1)) * int(m.group(2)), line.split()[0]
        if wid:
            return wid
        time.sleep(0.5)
    return None


def raise_window(wid):
    """Ask the window manager to bring the window to the front (it may decline: focus-stealing prevention)."""
    try:
        from Xlib import X, display, protocol
        d = display.Display(os.environ.get("DISPLAY", ":0"))
        win = d.create_resource_object("window", int(wid, 16))
        ev = protocol.event.ClientMessage(window=win, client_type=d.intern_atom("_NET_ACTIVE_WINDOW"),
                                          data=(32, [1, X.CurrentTime, 0, 0, 0]))
        d.screen().root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        d.flush()
    except Exception as e:
        print(f"could not raise the window: {e}", flush=True)


def add_markers(scn, mujoco, np, x_from=-2, x_to=22):
    """Decorations drawn only in the viewer (no physics): lane dashes and coloured pillars, so the tracking camera shows travel."""
    def add(gtype, size, pos, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.array(size, dtype=np.float64), np.array(pos, dtype=np.float64),
                            np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        scn.ngeom += 1
    for x in range(x_from, x_to):
        for y in (-0.7, 0.7):
            add(mujoco.mjtGeom.mjGEOM_BOX, [0.25, 0.025, 0.002], [x + 0.5, y, 0.002], [0.95, 0.95, 0.95, 1])
    cols = [[0.85, 0.25, 0.2, 1], [0.2, 0.45, 0.85, 1], [0.9, 0.75, 0.2, 1], [0.25, 0.7, 0.35, 1]]
    for k, x in enumerate(range(2, x_to, 3)):  # cylinder size = (radius, half-height)
        add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.12, 0.5, 0.0], [x, 2.0 if k % 2 else -2.0, 0.5], cols[k % 4])


def build_env(args, ar):
    """Eval env for walking; the training-style env (randomisation, curriculum) for the height-scan climber."""
    if args.climb and os.environ.get("GO2_STAIRS_VISION", "0") == "1":
        import train_curriculum as tc
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        env = DummyVecEnv([tc._make_stage_env("stairs", (args.vx, 0.0, 0.0), 0, int.from_bytes(os.urandom(2), "little"), 0.02, 5, True, True)])
        env = VecNormalize.load(ar._find_vecnorm(args.model), env)
        env.training, env.norm_reward = False, False
        return env, env.venv.envs[0].unwrapped, PPO.load(args.model, env=env)
    return ar._load(args.model, "stairs" if args.climb else "flat", args.vx, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(REPO / "training/pretrained/go2_walk_lat2/go2_flat_walk.zip"))
    ap.add_argument("--out", required=True, help="output GIF; MP4s are written next to it")
    ap.add_argument("--seconds", type=float, default=14.0, help="seconds to record once the robot moves (walking), or the per-attempt limit (climbing)")
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--steer-gain", type=float, default=2.0)
    ap.add_argument("--wz-max", type=float, default=0.5)
    ap.add_argument("--climb", action="store_true", help="stairs scene; attempts end on the top landing, a fall or a stall")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--gif-fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=None)
    ap.add_argument("--distance", type=float, default=None)
    args = ap.parse_args()
    if not os.environ.get("DISPLAY"):
        sys.exit("no $DISPLAY: this records a window on screen")
    az, el, dist = (args.azimuth, args.elevation, args.distance)
    az = az if az is not None else (60.0 if args.climb else 135.0)
    el = el if el is not None else (-14.0 if args.climb else -28.0)
    dist = dist if dist is not None else (1.9 if args.climb else 2.4)

    import analyze_rollout as ar  # imports play_policy (mujoco env) before SB3
    import mujoco
    import mujoco.viewer
    import numpy as np

    env, core, ppo = build_env(args, ar)
    ctrl_dt = 0.02

    def yaw(q):
        w, x, y, z = q[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    base = mujoco.mj_name2id(core.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    first_arm = mujoco.mj_name2id(core.model, mujoco.mjtObj.mjOBJ_BODY, "arm_base")
    if first_arm >= 0:  # hide the (unused) arm, visual only
        core.model.geom_rgba[core.model.geom_bodyid >= first_arm, 3] = 0.0
    top_z = float(getattr(core, "_top_z", 0.0))
    results = []  # (success, forward_m, fell, mp4)
    out = Path(args.out)

    with mujoco.viewer.launch_passive(core.model, core.data, show_left_ui=False, show_right_ui=False) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = base
        v.cam.distance, v.cam.elevation, v.cam.azimuth = dist, el, az
        if not args.climb:
            with v.lock():
                add_markers(v.user_scn, mujoco, np)
        wid = find_window("MuJoCo")
        print(f"viewer window {wid}", flush=True)
        if wid:
            raise_window(wid)
        for k in range(1, args.attempts + 1):
            if not v.is_running():
                break
            obs = env.reset()
            q0 = core.data.qpos.copy()
            start_q = q0
            y0 = yaw(q0)
            mp4 = str(out.with_name(f"{out.stem}_{k}.mp4")) if args.attempts > 1 else str(out.with_suffix(".mp4"))
            grab, t_move, steps, fell, success, landed_at, final_q = None, None, 0, False, False, None, None
            while v.is_running():
                t0 = time.time()
                q = core.data.qpos
                err = math.atan2(math.sin(y0 - yaw(q)), math.cos(y0 - yaw(q)))
                core.cmd[:] = [args.vx, 0.0, max(-args.wz_max, min(args.wz_max, args.steer_gain * err))]
                a, _ = ppo.predict(obs, deterministic=True)
                last_q = core.data.qpos.copy()  # the vec env auto-resets on done, so keep the pose from before the step
                obs, _, done, _ = env.step(a)
                steps += 1
                if done[0]:
                    final_q = last_q
                    fell = True  # fall, or the stairs env's stall rule (parked in front of a step)
                    break
                v.sync()
                moved = math.hypot(core.data.qpos[0] - q0[0], core.data.qpos[1] - q0[1])
                if grab is None and wid and moved > 0.1:  # the clip starts when the robot starts moving
                    t_move = time.time()
                    grab = subprocess.Popen(
                        ["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", str(args.fps), "-window_id", wid,
                         "-i", os.environ["DISPLAY"], "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                         "-preset", "veryfast", "-pix_fmt", "yuv420p", mp4], stdin=subprocess.PIPE)
                    print(f"[attempt {k}] recording -> {mp4}", flush=True)
                if args.climb and top_z > 0:
                    if core._terrain_height_under_base() >= 0.9 * top_z and landed_at is None:
                        landed_at = time.time()  # on the top landing: keep filming a few seconds, then call it a success
                    if landed_at is not None and time.time() - landed_at > 3.0:
                        success = True
                        break
                if t_move is not None and time.time() - t_move >= args.seconds:
                    break
                time.sleep(max(0.0, ctrl_dt - (time.time() - t0)))  # real-time pacing so the capture plays at true speed
            fq = final_q if final_q is not None else core.data.qpos
            q0 = start_q
            fwd = float(fq[0] - q0[0])
            dyaw = math.degrees(math.atan2(math.sin(yaw(fq) - y0), math.cos(yaw(fq) - y0)))
            if grab is not None:
                try:
                    grab.stdin.write(b"q")
                    grab.stdin.flush()
                    grab.wait(timeout=30)
                except Exception:
                    grab.kill()
            ok = success if args.climb else (not fell)
            print(f"[attempt {k}/{args.attempts}] forward {fwd:.2f} m, heading change {dyaw:+.0f} deg, "
                  f"{'climbed to the top landing' if success else ('ended early: fall/stall' if fell else 'time limit')}, {steps} steps", flush=True)
            results.append((ok, fwd, fell, mp4 if os.path.exists(mp4) else None))
            if ok and args.climb:
                break

    good = [r for r in results if r[0] and r[3]]
    pick = good[0] if good else (max((r for r in results if r[3]), key=lambda r: r[1], default=None))
    wins = sum(1 for r in results if r[0])
    print(f"summary: {wins} of {len(results)} attempt(s) succeeded", flush=True)
    if pick is None:
        print("no video was recorded", flush=True)
        return 1
    vf = f"fps={args.gif_fps},scale={args.width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", pick[3], "-vf", vf, str(out)])
    print(f"gif: {out} from {pick[3]} ({'successful attempt' if pick[0] else 'best attempt, none succeeded'}, {os.path.getsize(out) / 1e6:.2f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
