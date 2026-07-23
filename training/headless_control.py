"""
Headless MuJoCo viewer + IK-based gait controller for Unitree Go2.

No RL policy required. Generates explicit foot trajectories (linear stance,
sine-arch swing) and maps them to joint angles via 2D inverse kinematics.
Gait (walk/trot/canter/bound/pronk) is selected automatically from speed
using the intelligence.gait.GaitScheduler.
scipy.signal low-pass filters velocity commands for smooth speed changes.

Controls (terminal):
  W / S  —  forward / backward speed
  A / D  —  strafe left / right
  Q / E  —  yaw left / right
  Space  —  stop (zero command)
  R      —  reset simulation
  ESC    —  quit

Usage:
  python3 training/headless_control.py
  python3 training/headless_control.py --record out.mp4
"""

import argparse
import os
import sys
import threading
import time

import cv2
import mujoco
import numpy as np
from scipy.signal import butter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.go2_mujoco_env import SCENE_XML, SIM_DT, CTRL_DECIMATION
from intelligence.gait.gait_scheduler import GaitScheduler, Gait, GAITS

# --------------------------------------------------------------------------- #
# Actuator order: [FL_hip, FR_hip, RL_hip, RR_hip,
#                  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#                  FL_calf, FR_calf, RL_calf, RR_calf]
# Leg indices: 0=FL, 1=FR, 2=RL, 3=RR
# --------------------------------------------------------------------------- #

HIP_DEFAULT   = np.array([ 0.1, -0.1,  0.1, -0.1], dtype=np.float64)
THIGH_DEFAULT = np.array([ 0.8,  0.8,  1.0,  1.0], dtype=np.float64)
CALF_DEFAULT  = np.array([-1.5, -1.5, -1.5, -1.5], dtype=np.float64)

SETTLE_STEPS = 400  # sim steps at 200 Hz to let robot settle after spawn

L1 = 0.213   # thigh link length (m)
L2 = 0.213   # calf link length (m)

# Foot depth below hip at nominal standing height — matches the leg extension
# implied by THIGH_DEFAULT/CALF_DEFAULT via forward kinematics, so gait IK
# doesn't snap the legs on the stand->walk transition.
PZ_NOM = 0.306   # m, downward from hip to foot contact point

STRIDE_LEN = 0.14   # m stride at maximum speed

VX_MAX = 1.2
VX_MIN = -0.6
VY_MAX = 0.5
WZ_MAX = 0.8
CMD_STEP = 0.08

# Per-gait foot clearance and stride scaling
_STEP_HEIGHT = {
    Gait.STAND:  0.00,
    Gait.WALK:   0.05,
    Gait.TROT:   0.06,
    Gait.CANTER: 0.07,
    Gait.BOUND:  0.08,
    Gait.PRONK:  0.10,
}
_STRIDE_SCALE = {
    Gait.STAND:  0.0,
    Gait.WALK:   0.7,
    Gait.TROT:   1.0,
    Gait.CANTER: 1.2,
    Gait.BOUND:  1.4,
    Gait.PRONK:  0.8,
}


# --------------------------------------------------------------------------- #
# 2-D inverse kinematics (sagittal plane, backward-bending knee)
# --------------------------------------------------------------------------- #

def _leg_ik(px: float, pz: float) -> tuple:
    r = np.sqrt(px * px + pz * pz)
    r = np.clip(r, abs(L1 - L2) + 0.005, L1 + L2 - 0.005)
    cos_k = (L1 * L1 + L2 * L2 - r * r) / (2.0 * L1 * L2)
    phi_k = np.arccos(np.clip(cos_k, -1.0, 1.0))
    calf  = phi_k - np.pi
    sin_k = np.sin(phi_k)
    sin_hip = np.clip(L2 * sin_k / r, -1.0, 1.0)
    a_hip = np.arcsin(sin_hip)
    thigh = np.arctan2(px, pz) + a_hip
    return float(thigh), float(calf)


# --------------------------------------------------------------------------- #
# Foot trajectory
# --------------------------------------------------------------------------- #

def _foot_target(phi: float, stride: float,
                 duty_factor: float, step_height: float) -> tuple:
    """
    phi in [0, 2π).  duty_factor splits stance (first portion) from swing.
    Stance: foot slides backward (body advances).
    Swing:  foot lifts via sine arch and swings forward.
    """
    boundary = duty_factor * 2.0 * np.pi
    if phi < boundary:
        t  = phi / boundary
        px = stride * (0.5 - t)
        pz = PZ_NOM
    else:
        t  = (phi - boundary) / (2.0 * np.pi - boundary)
        px = stride * (-0.5 + t)
        pz = PZ_NOM - step_height * np.sin(t * np.pi)
    return px, pz


# --------------------------------------------------------------------------- #
# Gait controller
# --------------------------------------------------------------------------- #

def gait_ctrl(leg_phases: np.ndarray, cmd: np.ndarray, gait_params) -> np.ndarray:
    """
    Compute 12 joint targets from IK for the given gait.

    leg_phases: per-leg accumulated phase [FL, FR, RL, RR] in [0, 2π),
                maintained by the caller and advanced at gait_params.frequency.
                Keeping phase in the caller ensures continuity across gait switches.
    cmd: [vx, vy, wz] (filtered)
    """
    vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
    speed = float(np.hypot(vx, wz * 0.3))

    if speed < 0.02 or gait_params.name == "stand":
        return np.concatenate([HIP_DEFAULT, THIGH_DEFAULT, CALF_DEFAULT])

    gait_enum  = Gait(gait_params.name)
    stride     = STRIDE_LEN * _STRIDE_SCALE[gait_enum] * np.clip(vx / VX_MAX, -1.0, 1.0)
    step_h     = _STEP_HEIGHT[gait_enum]
    duty       = gait_params.duty_factor

    hips   = np.empty(4)
    thighs = np.empty(4)
    calves = np.empty(4)

    for i in range(4):
        phi = leg_phases[i]

        yaw_sign   = 1.0 if (i % 2 == 1) else -1.0
        leg_stride = stride + yaw_sign * 0.03 * wz

        px, pz = _foot_target(phi, leg_stride, duty, step_h)
        thigh_i, calf_i = _leg_ik(px, pz)

        lat_sign  = 1.0 if (i % 2 == 0) else -1.0
        hips[i]   = HIP_DEFAULT[i] + lat_sign * 0.12 * vy / VY_MAX
        thighs[i] = thigh_i
        calves[i] = calf_i

    return np.concatenate([hips, thighs, calves])


# --------------------------------------------------------------------------- #
# Command low-pass filter
# --------------------------------------------------------------------------- #

def _butter_lp(cutoff, fs, order=2):
    nyq = 0.5 * fs
    return butter(order, cutoff / nyq, btype="low")


class CommandFilter:
    def __init__(self, ctrl_hz: float = 50.0, cutoff: float = 1.5):
        b, a = _butter_lp(cutoff, ctrl_hz)
        self._b = b   # len 3 for 2nd-order
        self._a = a   # len 3: [1, a1, a2]
        self._x = np.zeros((3, len(b)))   # input history per channel
        self._y = np.zeros((3, len(a)))   # output history per channel

    def update(self, raw: np.ndarray) -> np.ndarray:
        out = np.empty(3)
        for i in range(3):
            self._x[i] = np.roll(self._x[i], 1)
            self._x[i, 0] = raw[i]
            # full difference equation: y = sum(b*x) - sum(a[1:]*y_prev)
            y = (np.dot(self._b, self._x[i])
                 - np.dot(self._a[1:], self._y[i, :-1]))
            self._y[i] = np.roll(self._y[i], 1)
            self._y[i, 0] = y
            out[i] = y
        return out


# --------------------------------------------------------------------------- #
# Keyboard input (non-blocking terminal)
# --------------------------------------------------------------------------- #

def _make_key_reader():
    import termios, tty, select
    try:
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (termios.error, Exception):
        # stdin is not a TTY (e.g. subprocess / IDE) — return no-op reader
        return lambda: None, lambda: None

    def read():
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore():
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return read, restore


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

def _make_renderer(model, h=480, w=640):
    return mujoco.Renderer(model, height=h, width=w)


def _render_frame(renderer, data):
    body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = 2.0
    cam.elevation = -20.0
    if body_id >= 0:
        cam.lookat[:] = data.xpos[body_id]
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _draw_hud(frame, cmd, filtered_cmd, body_z, step, fps, gait_name):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"cmd  vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}",
        f"filt vx={filtered_cmd[0]:+.2f}  vy={filtered_cmd[1]:+.2f}  wz={filtered_cmd[2]:+.2f}",
        f"body z={body_z:.3f} m    gait={gait_name}    step={step}    fps={fps:.0f}",
        "W/S fwd | A/D strafe | Q/E yaw | Space stop | R reset | ESC quit",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, 22 + i * 22), font, 0.55, (0, 0, 0), 2)
        cv2.putText(frame, txt, (10, 22 + i * 22), font, 0.55, (200, 255, 200), 1)
    return frame


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record",     type=str,   default=None,
                        help="Output video path, e.g. out.mp4")
    parser.add_argument("--fps-render", type=int,   default=30)
    parser.add_argument("--steps",      type=int,   default=0,
                        help="Auto-quit after N sim steps (0 = run until ESC)")
    parser.add_argument("--cmd",        type=float, nargs=3, default=[0.5, 0.0, 0.0],
                        metavar=("VX", "VY", "WZ"),
                        help="Initial velocity command (default: 0.5 0 0)")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip cv2.imshow (record-only mode)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    renderer  = _make_renderer(model)
    scheduler = GaitScheduler()

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, args.fps_render, (640, 480))
        print(f"Recording → {args.record}")

    if not args.no_display:
        cv2.namedWindow("Go2 Headless Control", cv2.WINDOW_AUTOSIZE)

    raw_cmd  = np.array(args.cmd, dtype=np.float64)
    cmd_filt = CommandFilter(ctrl_hz=50, cutoff=1.5)
    filtered = np.zeros(3)

    do_reset = threading.Event()
    do_quit  = threading.Event()

    _stand_ctrl = np.concatenate([HIP_DEFAULT, THIGH_DEFAULT, CALF_DEFAULT])

    def _reset():
        mujoco.mj_resetData(model, data)
        data.qpos[2]   = 0.42
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[7:]  = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                           0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
        data.ctrl[:]   = _stand_ctrl
        mujoco.mj_forward(model, data)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)

    _reset()

    read_key, restore_term = _make_key_reader()

    def key_thread():
        try:
            while not do_quit.is_set():
                k = read_key()
                if k is None:
                    time.sleep(0.005)
                    continue
                if   k in ("w", "W"): raw_cmd[0] = min(raw_cmd[0] + CMD_STEP, VX_MAX)
                elif k in ("s", "S"): raw_cmd[0] = max(raw_cmd[0] - CMD_STEP, VX_MIN)
                elif k in ("a", "A"): raw_cmd[1] = min(raw_cmd[1] + CMD_STEP,  VY_MAX)
                elif k in ("d", "D"): raw_cmd[1] = max(raw_cmd[1] - CMD_STEP, -VY_MAX)
                elif k in ("q", "Q"): raw_cmd[2] = min(raw_cmd[2] + CMD_STEP,  WZ_MAX)
                elif k in ("e", "E"): raw_cmd[2] = max(raw_cmd[2] - CMD_STEP, -WZ_MAX)
                elif k == " ":        raw_cmd[:] = 0.0
                elif k in ("r", "R"): raw_cmd[:] = 0.0; do_reset.set()
                elif k == "\x1b":     do_quit.set()
        finally:
            restore_term()

    kt = threading.Thread(target=key_thread, daemon=True)
    kt.start()

    print("\nGo2 Headless Control — IK gait controller (no RL)")
    print("W/S fwd | A/D strafe | Q/E yaw | Space stop | R reset | ESC quit\n")

    SIM_HZ       = int(1.0 / (SIM_DT * CTRL_DECIMATION))
    RENDER_EVERY = max(1, SIM_HZ // args.fps_render)

    step = 0; sim_t = 0.0
    fps_display = 0.0; frame_count = 0; t0_fps = time.perf_counter()

    # shared cycle phase in [0, 2π), continuous across gait switches; per-leg
    # phase_offsets from the active gait are added back in each tick below
    cycle_phase = 0.0

    while not do_quit.is_set():
        if do_reset.is_set():
            _reset()
            sim_t = 0.0; step = 0
            cmd_filt   = CommandFilter(ctrl_hz=50, cutoff=1.5)
            scheduler.__init__()
            cycle_phase = 0.0
            do_reset.clear()

        filtered[:] = cmd_filt.update(raw_cmd)

        speed      = float(np.hypot(filtered[0], filtered[2] * 0.3))
        gait_p      = scheduler.get_gait_params(speed)
        cycle_phase = (cycle_phase + 2.0 * np.pi * gait_p.frequency * SIM_DT * CTRL_DECIMATION) % (2.0 * np.pi)
        leg_phases  = (cycle_phase + np.array(gait_p.phase_offsets) * 2.0 * np.pi) % (2.0 * np.pi)
        ctrl        = gait_ctrl(leg_phases, filtered, gait_p)

        data.ctrl[:] = np.clip(ctrl,
                               model.actuator_ctrlrange[:, 0],
                               model.actuator_ctrlrange[:, 1])

        for _ in range(CTRL_DECIMATION):
            mujoco.mj_step(model, data)

        sim_t += SIM_DT * CTRL_DECIMATION
        step  += 1

        if step % RENDER_EVERY == 0:
            now = time.perf_counter()
            frame_count += 1
            if now - t0_fps >= 1.0:
                fps_display = frame_count / (now - t0_fps)
                frame_count = 0; t0_fps = now

            frame = _render_frame(renderer, data)
            frame = _draw_hud(frame, raw_cmd, filtered,
                              float(data.qpos[2]), step, fps_display,
                              gait_p.name)

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Go2 Headless Control", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    do_quit.set()

        if args.steps and step >= args.steps:
            do_quit.set()

        if data.qpos[2] < 0.12:
            print("  [fallen — auto-reset]")
            _reset(); sim_t = 0.0
            cmd_filt   = CommandFilter(ctrl_hz=50, cutoff=1.5)
            scheduler.__init__()
            cycle_phase = 0.0

    do_quit.set()
    cv2.destroyAllWindows()
    if writer:
        writer.release(); print(f"Saved {args.record}")
    renderer.close()
    print("Done.")


if __name__ == "__main__":
    main()
