#!/usr/bin/env python3
"""Drive the native-Gazebo Go2 from /cmd_vel with the trained MuJoCo PPO policy.

Drop-in replacement for scripts/cmd_vel_go2_gz.py (analytic gait): same /cmd_vel in, same
/go2/cmd/<joint> Float64 targets out, so launch/slam3d_go2.launch.py / Nav2 / the frontier
explorer don't change. Needs the policy trained with --asymmetric --obs-history 5 --gait and
GO2_TROT_PRIOR=1 (the prior is applied here exactly as Go2MujocoEnv.step does):

    GO2_TROT_PRIOR=1 python3 scripts/rl_cmd_vel_go2_gz.py \
        --model training/logs/mujoco_curriculum/best_walk_trotprior/go2_flat_walk.zip

Rebuilds Go2MujocoEnv._get_obs (legs-only 45-d + 9 gait feats, 5-frame history) from
/joint_states + /imu/data and normalises with the sibling vecnorm_final.pkl.

Limits: the walker was only trained on forward commands (vx ~0.25-0.45, vy/wz ~0 at curriculum
level 0.02), so /cmd_vel angular.z / linear.y are passed through but not followed; |vx| below
--stand-thresh holds the stand pose.
"""
import argparse
import os
import pickle
import sys
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from envs.go2_mujoco_env import (  # noqa: E402  (before stable_baselines3: import order segfaults mujoco)
    ACT_SCALE, DEFAULT_QPOS, GAIT_CONTACT_KAPPA, GAIT_FOOTSWING_RANGE, GAIT_FREQ_RANGE,
    GAIT_PRESETS, TROT_CALF_LIFT, TROT_THIGH_SWING)
import train_curriculum as tc  # noqa: E402,F401  (registers AsymmetricActorCriticPolicy)
from stable_baselines3 import PPO  # noqa: E402

CTRL_DT = 0.02
HIST = 5
# qpos / /joint_states order (leg by leg) and the policy's action order (hips, thighs, calves).
JOINTS = [f"{leg}_{j}_joint" for leg in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
ACT_ORDER = [f"{leg}_{j}_joint" for j in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
ACT_DEFAULT = np.array([DEFAULT_QPOS[JOINTS.index(n)] for n in ACT_ORDER], dtype=np.float32)


class RLCmdVel(Node):
    def __init__(self, model_path, trot_prior, stand_thresh, debug=False, sim_time=True, gyro_scale=1.0):
        # Tick on /clock: Gazebo runs below real time in heavy worlds (~0.7x measured), and a wall-clock
        # 50 Hz loop then feeds the policy stale sensor frames and runs its gait clock too fast.
        super().__init__("rl_cmd_vel_go2_gz",
                         parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, sim_time)])
        self.debug, self.n_js, self.n_imu, self.n_tick, self.last_a = debug, 0, 0, 0, np.zeros(12, np.float32)
        self.trot_prior, self.stand_thresh, self.gyro_scale = trot_prior, stand_thresh, gyro_scale
        d = os.path.dirname(os.path.abspath(model_path))
        vn = next(p for p in (os.path.join(d, "vecnorm_final.pkl"),
                              os.path.join(d, "..", "vecnorm_final.pkl")) if os.path.exists(p))
        with open(vn, "rb") as f:
            rms = pickle.load(f).obs_rms["policy"]
        self.mean, self.var = rms.mean.astype(np.float32), rms.var.astype(np.float32)
        self.pol = PPO.load(model_path, device="cpu").policy
        self.pol.set_training_mode(False)

        phase, offset, bound = GAIT_PRESETS["trotting"]
        self.gait_cmd = np.array([0.5 * sum(GAIT_FREQ_RANGE), phase, offset, bound,
                                  0.5 * sum(GAIT_FOOTSWING_RANGE)], dtype=np.float32)
        self.gait_index = 0.0
        self.cmd = np.zeros(3, np.float32)
        self.prev_action = np.zeros(12, np.float32)
        self.q, self.qd = DEFAULT_QPOS[:12].copy(), np.zeros(12, np.float32)
        self.ang_vel, self.quat = np.zeros(3, np.float32), np.array([1.0, 0, 0, 0], np.float32)
        self.hist = deque(maxlen=HIST)
        self.have_js = self.have_imu = False

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self._js, qos)
        self.create_subscription(Imu, "/imu/data", self._imu, qos)
        self.create_subscription(Twist, "/cmd_vel", self._cmd, 10)
        self.pubs = {n: self.create_publisher(Float64, f"/go2/cmd/{n}", 10) for n in JOINTS}
        self.create_timer(CTRL_DT, self._tick)

    def _js(self, m):
        idx = {n: i for i, n in enumerate(m.name)}
        if all(n in idx for n in JOINTS):
            self.q = np.array([m.position[idx[n]] for n in JOINTS], np.float32)
            self.qd = np.array([m.velocity[idx[n]] for n in JOINTS], np.float32)
            self.have_js = True
            self.n_js += 1

    def _imu(self, m):
        o, a = m.orientation, m.angular_velocity
        self.quat = np.array([o.w, o.x, o.y, o.z], np.float32)
        self.ang_vel = np.array([a.x, a.y, a.z], np.float32) * np.array([self.gyro_scale, self.gyro_scale, 1.0], np.float32)
        self.have_imu = True
        self.n_imu += 1

    def _cmd(self, m):
        self.cmd[:] = [m.linear.x, m.linear.y, m.angular.z]

    def _gait(self):
        """Same clock/prior maths as Go2MujocoEnv._step_gait / _trot_prior_offset."""
        freq, phase, offset, bound, _ = self.gait_cmd
        self.gait_index = (self.gait_index + CTRL_DT * float(freq)) % 1.0
        gi = self.gait_index
        fp = np.array([gi + phase + offset + bound, gi + offset, gi + bound, gi + phase]) % 1.0
        clocks, swing = np.empty(4, np.float32), np.zeros(4, np.float32)
        for i, p in enumerate(fp):
            warped = p if p < 0.5 else 0.5 + (p - 0.5)
            clocks[i] = np.sin(2 * np.pi * warped)
            if warped >= 0.5:
                swing[i] = np.sin(2 * np.pi * (warped - 0.5))
        return clocks, swing

    def _obs(self, clocks):
        w, x, y, z = self.quat
        gravity = np.array([2 * (-z * x - w * y), -2 * (z * y - w * x), 1 - 2 * (w * w + z * z)], np.float32)
        freq_n = (self.gait_cmd[0] - GAIT_FREQ_RANGE[0]) / (GAIT_FREQ_RANGE[1] - GAIT_FREQ_RANGE[0] + 1e-6)
        feat = np.array([freq_n, self.gait_cmd[1], self.gait_cmd[2], self.gait_cmd[3],
                         self.gait_cmd[4] / GAIT_FOOTSWING_RANGE[1]], np.float32)
        return np.concatenate([self.ang_vel * 0.25, gravity, self.cmd * np.array([2.0, 2.0, 0.25], np.float32),
                               self.q - DEFAULT_QPOS[:12], self.qd * 0.05, self.prev_action, feat, clocks])

    def _publish(self, target_act_order):
        for n, v in zip(ACT_ORDER, target_act_order):
            self.pubs[n].publish(Float64(data=float(v)))

    def _tick(self):
        if not (self.have_js and self.have_imu):
            return
        if abs(self.cmd[0]) < self.stand_thresh:  # untrained region: hold the stand pose
            self.hist.clear()
            self.prev_action[:] = 0.0
            self._publish(ACT_DEFAULT)
            return
        clocks, swing = self._gait()
        o = self._obs(clocks)
        self.n_tick += 1
        if self.debug and self.n_tick % 50 == 0:
            print(f"[dbg] js {self.n_js} imu {self.n_imu} ticks {self.n_tick} | grav {np.round(o[3:6], 2)} "
                  f"|angvel| {np.linalg.norm(self.ang_vel):.2f} | max|dof_pos| {np.abs(o[9:21]).max():.2f} "
                  f"max|dof_vel*0.05| {np.abs(o[21:33]).max():.2f} | mean|a| {np.abs(self.last_a).mean():.2f} "
                  f"cmd {np.round(self.cmd, 2)}", flush=True)
            self.n_js = self.n_imu = 0
        if not self.hist:
            self.hist.extend([o] * HIST)
        else:
            self.hist.append(o)
        x = np.clip((np.concatenate(self.hist) - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)
        import torch
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32)[None]
            a = self.pol.action_net(self.pol.mlp_extractor.policy_net(t))[0].numpy()
        a = np.clip(a, -1.0, 1.0)
        target = ACT_DEFAULT + a * ACT_SCALE
        if self.trot_prior:
            target[4:8] += TROT_THIGH_SWING * (swing - 0.5 * float(swing.max()))
            target[8:12] += -TROT_CALF_LIFT * swing
        self.prev_action = self.last_a = a.astype(np.float32)
        self._publish(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--no-trot-prior", action="store_true")
    ap.add_argument("--stand-thresh", type=float, default=0.05)
    ap.add_argument("--gyro-scale", type=float, default=1.0,
                    help="scale the IMU roll/pitch rates (not yaw) before they enter the obs: Gazebo wobbles ~2-2.4x MuJoCo in roll/pitch while yaw matches")
    ap.add_argument("--wall-time", action="store_true", help="tick on wall-clock instead of /clock")
    ap.add_argument("--debug", action="store_true", help="print obs/rate diagnostics once per second")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    prior = os.environ.get("GO2_TROT_PRIOR", "0") == "1" and not args.no_trot_prior
    rclpy.spin(RLCmdVel(args.model, prior, args.stand_thresh, args.debug, not args.wall_time, args.gyro_scale))


if __name__ == "__main__":
    main()
