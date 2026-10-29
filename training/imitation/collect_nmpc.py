"""Log Quad-SDK NMPC walking as (obs, action) pairs in the MuJoCo Go2 legs-only layout.

Run next to the NMPC stack (use a fresh ROS_DOMAIN_ID / GZ_PARTITION):

    python3 training/imitation/collect_nmpc.py --seconds 180 --out training/imitation/data/nmpc_run1.npz

Per 50 Hz tick it writes one 45-d frame, exactly Go2MujocoEnv._get_obs(legs_only):
    ang_vel*0.25 (3) | gravity in body (3) | cmd*[2,2,.25] (3) | dof_pos (12) | dof_vel*0.05 (12) | prev_action (12)
and the 12-d action (NMPC joint target - stand pose) / ACT_SCALE, in the env's
actuator order [hips FL,FR,RL,RR | thighs | calves]; frames are NOT clipped here
(the clipped fraction is reported so you can see if NMPC strides exceed +-0.25 rad).

Joint reorder: Quad-SDK legs are [FL, BL, FR, BR] x [abd, hip, knee]; the env's qpos is
[FL, FR, RL, RR] x [hip, thigh, calf]. dof_pos is taken relative to the Quad-SDK stand
pose (0, 0.8, -1.5), which equals the env's DEFAULT_QPOS offset convention.

cmd is relabelled in hindsight: NMPC walks to a goal, so the commanded twist is the
body-frame velocity actually achieved, averaged over --cmd-window seconds ahead.
"""
import argparse
import threading
import time

import numpy as np
import rclpy
from quad_msgs.msg import LegCommandArray, RobotState
from rclpy.node import Node

ACT_SCALE = 0.25
STAND = np.array([0.0, 0.8, -1.5])
QSDK_LEG_OF = {"FL": 0, "FR": 2, "RL": 1, "RR": 3}  # env leg -> Quad-SDK leg index


def _quat_rotate_inv(q, v):
    x, y, z, w = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r.T @ v


def _reorder(qsdk12):
    """Quad-SDK [leg][abd,hip,knee] flat 12 -> env qpos order [FL h,t,c, FR, RL, RR]."""
    return np.concatenate([qsdk12[3 * QSDK_LEG_OF[leg]: 3 * QSDK_LEG_OF[leg] + 3]
                           for leg in ("FL", "FR", "RL", "RR")])


def _to_actuator_order(qpos_order12):
    """[leg][h,t,c] -> [hips(4), thighs(4), calves(4)] (the env's action layout)."""
    a = qpos_order12.reshape(4, 3)
    return np.concatenate([a[:, 0], a[:, 1], a[:, 2]])


class Collector(Node):
    def __init__(self):
        super().__init__("bc_collector")
        self.lock = threading.Lock()
        self.state = None
        self.cmd = None
        self.create_subscription(RobotState, "/robot_1/state/ground_truth", self._on_state, 5)
        self.create_subscription(LegCommandArray, "/robot_1/control/joint_command", self._on_cmd, 5)

    def _on_state(self, m):
        with self.lock:
            self.state = m

    def _on_cmd(self, m):
        with self.lock:
            self.cmd = np.array([mc.pos_setpoint for lc in m.leg_commands for mc in lc.motor_commands])

    def snapshot(self):
        with self.lock:
            return self.state, None if self.cmd is None else self.cmd.copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmd-window", type=float, default=1.0)
    args = ap.parse_args()

    rclpy.init()
    node = Collector()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    dt, frames, acts, body_vel, z, t_log = 0.02, [], [], [], [], []
    prev_act = np.zeros(12)
    t0 = next_t = time.time()
    while time.time() - t0 < args.seconds:
        next_t += dt
        time.sleep(max(0.0, next_t - time.time()))
        st, cmd = node.snapshot()
        if st is None or cmd is None or len(st.joints.position) < 12 or len(cmd) < 12:
            continue
        q = _reorder(np.array(st.joints.position[:12]))
        qd = _reorder(np.array(st.joints.velocity[:12]))
        dof_pos = (q - np.tile(STAND, 4)).astype(np.float32)
        o = st.body.pose.orientation
        quat = (o.x, o.y, o.z, o.w)
        tw = st.body.twist
        lin_b = _quat_rotate_inv(quat, np.array([tw.linear.x, tw.linear.y, tw.linear.z]))
        ang_b = _quat_rotate_inv(quat, np.array([tw.angular.x, tw.angular.y, tw.angular.z]))
        grav = _quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        act = _to_actuator_order((_reorder(cmd[:12]) - np.tile(STAND, 4))) / ACT_SCALE
        # frame stored with cmd placeholder (filled in hindsight below)
        frames.append(np.concatenate([ang_b * 0.25, grav, np.zeros(3), dof_pos, qd * 0.05, prev_act]).astype(np.float32))
        acts.append(act.astype(np.float32))
        body_vel.append([lin_b[0], lin_b[1], ang_b[2]])
        z.append(st.body.pose.position.z)
        t_log.append(time.time() - t0)
        prev_act = np.clip(act, -1, 1)
        if len(frames) % 500 == 0:
            print(f"{len(frames)} frames, x-vel {lin_b[0]:+.2f} z {z[-1]:.2f}", flush=True)

    frames, acts, body_vel = np.array(frames), np.array(acts), np.array(body_vel)
    w = max(1, int(args.cmd_window / dt))
    for i in range(len(frames)):  # hindsight command: mean achieved body twist over the next window
        frames[i, 6:9] = body_vel[i:i + w].mean(axis=0) * np.array([2.0, 2.0, 0.25])
    np.savez_compressed(args.out, obs=frames, act=acts, z=np.array(z), t=np.array(t_log))
    clipped = float(np.mean(np.abs(acts) > 1.0))
    print(f"saved {len(frames)} frames -> {args.out}; |action|>1 (clipped) fraction: {clipped:.1%}; "
          f"mean body vx {body_vel[:, 0].mean():+.2f}; z range {min(z):.2f}-{max(z):.2f}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
