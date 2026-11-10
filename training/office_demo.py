#!/usr/bin/env python3
"""Live MuJoCo window: Go2 (official Unitree meshes) walks a scripted tour of the office.

  python3 training/office_demo.py                      # just watch
  python3 training/office_demo.py --record out.mp4     # also screen-record the cropped viewer window + make a GIF

Uses the lat2 walker (legs-only, gait, asymmetric, 5-frame history). Walker follows forward and in-place turns only.
"""
import argparse, os, re, subprocess, sys, time
os.environ.setdefault("GO2_TROT_PRIOR", "1"); os.environ.setdefault("GO2_WIDE_CMD", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
_scene = "plain" if "--plain" in sys.argv or "--steps" in sys.argv else "office"
os.environ["GO2_SCENE_XML"] = os.path.join(HERE, "envs", f"go2_{_scene}_scene.xml")
if "--steps" in sys.argv:   # 3 cm height-scan climber on the Unitree-mesh steps (needs these set before the env import)
    os.environ.update(GO2_STAIRS_VISION="1", GO2_NO_STAND_SUBSIDY="1", GO2_LIFT_SCALE="2.0",
                      GO2_STAIRS_SCENE=os.path.join(HERE, "envs", "go2_steps_h03_unitree.xml"))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "envs"))
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from play_policy import _make_env

def yaw_of(d):
    w, x, y, z = d.sensor("orientation").data
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def raise_window(wid):
    """Raise the viewer to the top and mark it always-on-top so nothing covers it while x11grab records."""
    try:
        from Xlib import X, display, protocol
        d = display.Display(); root = d.screen().root; w = d.create_resource_object("window", int(wid, 16))
        def msg(atom, data):
            root.send_event(protocol.event.ClientMessage(window=w, client_type=d.intern_atom(atom), data=(32, data)),
                            event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        msg("_NET_WM_STATE", [1, d.intern_atom("_NET_WM_STATE_ABOVE"), 0, 1, 0])   # add ABOVE
        msg("_NET_ACTIVE_WINDOW", [2, X.CurrentTime, 0, 0, 0])
        w.configure(stack_mode=X.Above)
        d.flush(); d.close()
    except Exception as e:
        print("raise_window failed:", e)

def window_geometry(title="MuJoCo", tries=60):
    for _ in range(tries):
        tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True).stdout
        wid = re.search(rf'(0x[0-9a-f]+) "{title}[^"]*":', tree)
        out = subprocess.run(["xwininfo", "-id", wid.group(1)], capture_output=True, text=True).stdout if wid else ""
        m = {k: re.search(rf"{k}:\s+(-?\d+)", out) for k in ("Absolute upper-left X", "Absolute upper-left Y", "Width", "Height")}
        if all(m.values()):
            x, y, w, h = (int(m[k].group(1)) for k in ("Absolute upper-left X", "Absolute upper-left Y", "Width", "Height"))
            return x, y, w - w % 2, h - h % 2, wid.group(1)
        time.sleep(0.25)
    return None

# (kind, value): walk <seconds> at vx | turn to relative heading <deg>
TOUR = [("walk", 11), ("turn", 90), ("walk", 3), ("turn", 90), ("walk", 9), ("turn", 90), ("walk", 3), ("turn", 90), ("walk", 6), ("walk", 0.1)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--record", default=None, help="mp4 path; a .gif is written next to it")
    ap.add_argument("--plain", action="store_true", help="plain flat ground instead of the office")
    ap.add_argument("--spin", action="store_true", help="only rotate in place (full 360 deg) instead of the tour")
    ap.add_argument("--cam", choices=("iso", "side", "follow"), default="iso", help="camera: iso (default), side (profile) or follow (behind)")
    ap.add_argument("--walk", action="store_true", help="only walk forward (no turns)")
    ap.add_argument("--steps", action="store_true", help="climb the four 3 cm steps (go2_climb_scan, height scan)")
    ap.add_argument("--vx", type=float, default=0.35)
    ap.add_argument("--wz", type=float, default=0.5)
    ap.add_argument("--loops", type=int, default=1)
    a = ap.parse_args()
    tour = [("walk", 1.0)] + [("turn", 90)] * 4 + [("walk", 0.1)] if a.spin else ([("walk", 18)] if a.walk or a.steps else TOUR)

    if a.model is None:
        a.model = os.path.join(HERE, "pretrained/go2_climb_scan/go2_stairs_h03.zip" if a.steps else "pretrained/go2_walk_lat2/go2_flat_walk.zip")
    raw = _make_env("stairs" if a.steps else "flat", (a.vx, 0, 0), use_vision=a.steps, gait_conditioned=True, obs_history=5, asymmetric=True, legs_only=True)
    env = DummyVecEnv([lambda: raw])
    vn = os.path.join(os.path.dirname(a.model), "vecnorm_final.pkl")
    env = VecNormalize.load(vn, env); env.training = False; env.norm_reward = False
    model = PPO.load(a.model, env=env)
    core = raw
    while hasattr(core, "env"): core = core.env
    obs = env.reset()

    def start_rec():
        g = window_geometry()
        if not g:
            print("could not find viewer window"); return None
        x, y, w, h, wid = g
        raise_window(wid); time.sleep(0.6)
        g = window_geometry(); x, y, w, h = g[:4]; start_rec.wid = wid
        print(f"recording window {w}x{h}+{x}+{y} -> {a.record}")
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", "30", "-video_size", f"{w}x{h}",
                              "-i", f"{os.environ.get('DISPLAY', ':0')}+{x},{y}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", a.record],
                             stdin=subprocess.PIPE)
        time.sleep(0.5)
        return p

    ff = None; nstep = 0
    with mujoco.viewer.launch_passive(core.model, core.data, show_left_ui=False, show_right_ui=False) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = mujoco.mj_name2id(core.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        v.cam.distance, v.cam.azimuth, v.cam.elevation = {"iso": (3.2, -125, -22), "side": (2.6, 90, -8), "follow": (2.4, 0, -14)}[a.cam]
        time.sleep(1.5)
        dt = core.model.opt.timestep * 4
        falls = 0
        for _ in range(a.loops):
            for kind, val in tour:
                if not v.is_running(): break
                if kind == "walk":
                    core.cmd = np.array([a.vx, 0, 0], np.float32); n = int(val / dt)
                    target = None
                else:
                    target = wrap(yaw_of(core.data) + np.deg2rad(val)); n = int(12 / dt)
                for _ in range(n):
                    t0 = time.perf_counter()
                    if target is not None:
                        err = wrap(target - yaw_of(core.data))
                        if abs(err) < 0.06: break
                        core.cmd = np.array([0, 0, float(np.clip(2.0 * err, -a.wz, a.wz))], np.float32)
                    act, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _ = env.step(act)
                    if done[0]: falls += 1
                    v.sync(); nstep += 1
                    if ff is not None and nstep % 150 == 0:
                        raise_window(start_rec.wid)
                    if a.record and ff is None and nstep == 15:
                        ff = start_rec()
                        if ff is None: return 1
                    time.sleep(max(0, dt - (time.perf_counter() - t0)))
                    if not v.is_running(): break
        time.sleep(0.5)
        print(f"tour finished, resets={falls}")
    if ff:
        ff.stdin.write(b"q"); ff.stdin.flush(); ff.wait(timeout=20)
        gif = os.path.splitext(a.record)[0] + ".gif"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.record, "-vf", "fps=8,scale=400:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64:stats_mode=diff[p];[s1][p]paletteuse=dither=none:diff_mode=rectangle", gif], check=True)
        print("wrote", a.record, gif)

if __name__ == "__main__":
    sys.exit(main() or 0)
