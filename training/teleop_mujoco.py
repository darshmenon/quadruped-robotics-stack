"""
Keyboard teleop for Go2 in MuJoCo using a trained SB3 policy.

Controls:
  W/S  - forward / backward
  A/D  - strafe left / right
  Q/E  - yaw left / yaw right
  R    - reset episode
  ESC  - quit

Usage:
    python3 training/teleop_mujoco.py --model training/logs/mujoco/best_model.zip
    python3 training/teleop_mujoco.py  # random policy (for testing)
"""

import argparse
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mujoco
import mujoco.viewer

from envs.go2_mujoco_env import Go2MujocoEnv, DEFAULT_QPOS, ACT_SCALE, CTRL_DECIMATION

try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def _get_key_reader():
    """Returns a non-blocking key reader for linux terminals."""
    import termios, tty, select

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def read():
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore():
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return read, restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="SB3 .zip model path")
    args = parser.parse_args()

    model = None
    if args.model and HAS_SB3:
        model = PPO.load(args.model)
        print(f"Loaded policy from {args.model}")
    else:
        print("No model — using random actions. Pass --model <path> to use a trained policy.")

    env = Go2MujocoEnv(cmd=(0.0, 0.0, 0.0), render_mode=None)
    mj_model = env.model
    mj_data = env.data

    cmd = np.zeros(3, dtype=np.float32)  # lin_x, lin_y, ang_yaw
    cmd_lock = threading.Lock()
    do_reset = threading.Event()
    do_quit = threading.Event()

    CMD_STEP = 0.1

    def key_thread():
        read_key, restore = _get_key_reader()
        try:
            while not do_quit.is_set():
                k = read_key()
                if k is None:
                    continue
                with cmd_lock:
                    if k in ('w', 'W'):
                        cmd[0] = min(cmd[0] + CMD_STEP, 1.5)
                    elif k in ('s', 'S'):
                        cmd[0] = max(cmd[0] - CMD_STEP, -1.5)
                    elif k in ('a', 'A'):
                        cmd[1] = min(cmd[1] + CMD_STEP, 1.0)
                    elif k in ('d', 'D'):
                        cmd[1] = max(cmd[1] - CMD_STEP, -1.0)
                    elif k in ('q', 'Q'):
                        cmd[2] = min(cmd[2] + CMD_STEP, 1.0)
                    elif k in ('e', 'E'):
                        cmd[2] = max(cmd[2] - CMD_STEP, -1.0)
                    elif k in ('r', 'R'):
                        do_reset.set()
                        cmd[:] = 0.0
                    elif k == '\x1b':
                        do_quit.set()
        finally:
            restore()

    print("\nTeleop controls: W/S=fwd/back  A/D=strafe  Q/E=yaw  R=reset  ESC=quit\n")
    kt = threading.Thread(target=key_thread, daemon=True)
    kt.start()

    obs, _ = env.reset(seed=0)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and not do_quit.is_set():
            if do_reset.is_set():
                obs, _ = env.reset()
                do_reset.clear()

            with cmd_lock:
                env.cmd[:] = cmd
                print(f"\rcmd: lin_x={cmd[0]:+.2f}  lin_y={cmd[1]:+.2f}  ang={cmd[2]:+.2f}  ", end="")

            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, _, terminated, truncated, _ = env.step(action)
            viewer.sync()

            if terminated or truncated:
                obs, _ = env.reset()

    do_quit.set()
    env.close()
    print("\nQuitting.")


if __name__ == "__main__":
    main()
