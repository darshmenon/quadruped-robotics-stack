"""
Keyboard teleop for Go2 in MuJoCo using a trained SB3 policy.

Works with every checkpoint layout the repo has produced (plain 49-d, legs-only, 5-frame
history, asymmetric critic, gait-conditioned): the layout is probed against the checkpoint and
the VecNormalize stats next to it are applied, same as training/eval_checkpoints.py.

Controls (terminal must have focus; the MuJoCo viewer opens alongside):
  W/S    - forward / backward
  A/D    - strafe left / right
  Q/E    - yaw left / yaw right
  SPACE  - zero the command
  R      - reset episode
  ESC    - quit

Usage:
    python3 training/teleop_mujoco.py --model training/logs/mujoco/best_model.zip
    python3 training/teleop_mujoco.py --model training/logs/mujoco_curriculum/best_walk_trotprior/go2_flat_walk.zip --trot-prior
    python3 training/teleop_mujoco.py  # random policy (for testing)

The trot-prior walker was only trained on forward commands (vx ~0.25-0.45), so the prompt
shows "(untrained)" when the command leaves that range: expect it to ignore turn/strafe/stop.
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torch lazily imports triton when SB3 builds the Adam optimizer; a broken
# local triton/CUDA-driver combo segfaults there, so block the import.
sys.modules.setdefault("triton", None)

import numpy as np
import mujoco.viewer

import play_policy as pp  # noqa: E402  (before stable_baselines3: import order segfaults mujoco)
from envs.go2_mujoco_env import Go2MujocoEnv  # noqa: E402

try:
    from eval_checkpoints import LAYOUTS, _build, _find_vecnorm  # noqa: E402
    from stable_baselines3 import PPO  # noqa: E402
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

CMD_STEP = 0.05
TRAINED_VX = (0.2, 0.5)  # forward range the trot-prior walker saw in training


def _get_key_reader():
    """Returns a non-blocking key reader for linux terminals."""
    import select
    import termios
    import tty

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


def _load_policy(path):
    """Probe the known obs layouts; returns (vec_env, core_env, model)."""
    vn = _find_vecnorm(path)
    last = None
    for layout in LAYOUTS:
        try:
            env, core = _build("flat", 0.0, layout, vn)
            return env, core, PPO.load(path, env=env)
        except (AssertionError, ValueError) as e:  # obs layout mismatch
            last = e
    raise SystemExit(f"no obs layout fits {path}: {last}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None, help="SB3 .zip model path")
    parser.add_argument("--trot-prior", action="store_true",
                        help="apply the clocked trot prior (same as GO2_TROT_PRIOR=1; needed for trot-prior checkpoints)")
    args = parser.parse_args()
    if args.trot_prior:
        os.environ["GO2_TROT_PRIOR"] = "1"  # read by Go2MujocoEnv at construction

    if args.model and HAS_SB3:
        env, core, model = _load_policy(args.model)
        print(f"Loaded policy from {args.model}")
        obs = env.reset()
    else:
        print("No model — using random actions. Pass --model <path> to use a trained policy.")
        core = Go2MujocoEnv(cmd=(0.0, 0.0, 0.0), render_mode=None)
        env, model = core, None
        obs, _ = core.reset(seed=0)

    cmd = np.zeros(3, dtype=np.float32)  # lin_x, lin_y, ang_yaw
    cmd_lock = threading.Lock()
    do_reset = threading.Event()
    do_quit = threading.Event()

    def key_thread():
        read_key, restore = _get_key_reader()
        try:
            while not do_quit.is_set():
                k = read_key()
                if k is None:
                    time.sleep(0.01)
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
                    elif k == ' ':
                        cmd[:] = 0.0
                    elif k in ('r', 'R'):
                        do_reset.set()
                        cmd[:] = 0.0
                    elif k == '\x1b':
                        do_quit.set()
        finally:
            restore()

    print("\nTeleop controls: W/S=fwd/back  A/D=strafe  Q/E=yaw  SPACE=zero  R=reset  ESC=quit\n")
    threading.Thread(target=key_thread, daemon=True).start()

    dt = pp.SIM_DT * pp.CTRL_DECIMATION
    with mujoco.viewer.launch_passive(core.model, core.data) as viewer:
        while viewer.is_running() and not do_quit.is_set():
            t0 = time.time()
            if do_reset.is_set():
                obs = env.reset() if model is not None else core.reset()[0]
                do_reset.clear()

            with cmd_lock:
                core.cmd[:] = cmd
                untrained = "" if TRAINED_VX[0] <= cmd[0] <= TRAINED_VX[1] and not cmd[1] and not cmd[2] \
                    else "  (untrained)"
                print(f"\rcmd: lin_x={cmd[0]:+.2f}  lin_y={cmd[1]:+.2f}  ang={cmd[2]:+.2f}{untrained}      ", end="")

            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = env.step(action)  # DummyVecEnv auto-resets on termination
            else:
                obs, _, term, trunc, _ = core.step(core.action_space.sample())
                if term or trunc:
                    obs, _ = core.reset()
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - t0)))  # real-time pacing

    do_quit.set()
    print("\nQuitting.")


if __name__ == "__main__":
    main()
