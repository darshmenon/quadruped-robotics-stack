#!/usr/bin/env python3
"""Leg odometry for the Go2 from /joint_states (+ /imu/data), the legged equivalent of wheel odometry.

Per foot, forward kinematics gives its position p in the body frame and the Jacobian J(q) its velocity J*qdot.
A foot on the ground does not move in the world, so the body velocity is  v_body = -(J qdot) - omega x p.
Stance feet are the ones lowest in the gravity-aligned frame (soft weights, tolerance --stance-tol); their estimates
are averaged. The pose is the integral of R(imu orientation) * v_body.

Output: nav_msgs/Odometry on /leg_odom (frame leg_odom -> base), stamped with the /joint_states stamp. The twist
(in the base frame) carries a covariance that grows with fewer stance feet and with foot slip evidence (spread of
the per-foot estimates); --sigma-v (default 0.5 m/s) is the floor: against ground truth the raw velocity is mostly noise (RMS error 0.66 m/s at a true speed of
~0.16 m/s, scale ~0.5-0.7), so 0.1 was far too confident. Fusing it with RKO-LIO gave no gain in the sim: use it as a weak source.
Geometry from urdf/go2_unitree (hip +-0.1934/+-0.0465, thigh offset +-0.0955, thigh/calf links 0.213 m).

  live:    python3 scripts/leg_odom_go2.py --ros-args -p use_sim_time:=true
  offline: python3 scripts/leg_odom_go2.py --bag <bag dir> --tum out.tum        (needs ROS 2 sourced for rosbag2_py)
"""
import argparse, math, sys
import numpy as np

HIP = {"FL": (0.1934, 0.0465), "FR": (0.1934, -0.0465), "RL": (-0.1934, 0.0465), "RR": (-0.1934, -0.0465)}
SIDE = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}
L_THIGH_OFF, L_LEG = 0.0955, 0.213
LEGS = ["FL", "FR", "RL", "RR"]
JOINTS = {l: [f"{l}_hip_joint", f"{l}_thigh_joint", f"{l}_calf_joint"] for l in LEGS}


def rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def foot_pos_exact(leg, q):
    """Foot position in the body frame for q = (hip, thigh, calf): the chain hip -> thigh -> calf -> foot."""
    hx, hy = HIP[leg]
    calf_to_foot = np.array([0.0, 0.0, -L_LEG])
    thigh_to_calf = np.array([0.0, 0.0, -L_LEG])
    in_thigh = thigh_to_calf + ry(q[2]) @ calf_to_foot
    in_hip = np.array([0.0, SIDE[leg] * L_THIGH_OFF, 0.0]) + ry(q[1]) @ in_thigh
    return np.array([hx, hy, 0.0]) + rx(q[0]) @ in_hip


def jacobian(leg, q, eps=1e-6):
    J = np.zeros((3, 3))
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        J[:, k] = (foot_pos_exact(leg, q + d) - foot_pos_exact(leg, q - d)) / (2 * eps)
    return J


def quat_to_R(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


class LegOdom:
    def __init__(self, stance_tol=0.002, sigma_v=0.5, scale=1.0):
        self.stance_tol, self.sigma_v, self.scale = stance_tol, sigma_v, scale
        self.pos = np.zeros(3); self.t_prev = None; self.R = np.eye(3); self.omega = np.zeros(3)
        self.v_body = np.zeros(3); self.cov_v = np.eye(3) * sigma_v ** 2; self.n_stance = 0

    def set_imu(self, R, omega): self.R, self.omega = R, np.asarray(omega, float)

    def update(self, t, qpos, qvel):
        """qpos/qvel: dict leg -> (3,) arrays. Returns body-frame velocity or None for the first sample."""
        pts, vels = {}, {}
        for l in LEGS:
            pts[l] = foot_pos_exact(l, qpos[l])
            vels[l] = -(jacobian(l, qpos[l]) @ qvel[l]) - np.cross(self.omega, pts[l])
        zw = {l: (self.R @ pts[l])[2] for l in LEGS}                       # foot height in the gravity-aligned frame
        zmin = min(zw.values())
        w = {l: math.exp(-max(zw[l] - zmin, 0.0) / self.stance_tol) for l in LEGS}
        ws = sum(w.values())
        v = sum(w[l] * vels[l] for l in LEGS) / ws
        v = v * self.scale
        v[2] = 0.0                      # the base height is held by the controller: a nonzero vz is foot compression, not motion
        self.n_stance = sum(1 for l in LEGS if zw[l] - zmin < self.stance_tol)
        spread = math.sqrt(sum(w[l] * float(np.sum((vels[l] - v) ** 2)) for l in LEGS) / ws / 3.0)   # slip evidence
        sig = math.sqrt(self.sigma_v ** 2 + spread ** 2) * (1.0 if self.n_stance >= 2 else 4.0)     # one stance foot: far less trustworthy
        self.v_body, self.cov_v = v, np.eye(3) * sig ** 2
        if self.t_prev is not None:
            dt = t - self.t_prev
            if 0.0 < dt < 0.1:
                self.pos = self.pos + self.R @ v * dt
        self.t_prev = t
        return v


def _yaw(R): return math.atan2(R[1, 0], R[0, 0])


def run_bag(bag, tum, a):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
    lo = LegOdom(a.stance_tol, a.sigma_v, a.scale); n = 0; out = open(tum, "w")
    imu_seen = False
    while r.has_next():
        name, data, _ = r.read_next()
        if name == "/imu/data":
            m = deserialize_message(data, types[name]); o, g = m.orientation, m.angular_velocity
            lo.set_imu(quat_to_R(o.x, o.y, o.z, o.w), (g.x, g.y, g.z)); imu_seen = True
        elif name == "/joint_states" and imu_seen:
            m = deserialize_message(data, types[name]); idx = {nm: i for i, nm in enumerate(m.name)}
            qp = {l: np.array([m.position[idx[j]] for j in JOINTS[l]]) for l in LEGS}
            qv = {l: np.array([m.velocity[idx[j]] for j in JOINTS[l]]) for l in LEGS}
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            lo.update(t, qp, qv)
            if lo.t_prev is not None and n % a.decimate == 0:
                q = _quat_from_R(lo.R)
                out.write(f"{t:.6f} {lo.pos[0]:.5f} {lo.pos[1]:.5f} {lo.pos[2]:.5f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
            n += 1
    out.close(); print(f"{n} joint_states -> {tum}")


def _quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2; q = [0.0] * 4
    q[i] = 0.25 * s; q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s; q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def run_ros(a):
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Imu, JointState

    class N(Node):
        def __init__(s):
            super().__init__("leg_odom_go2")
            s.lo = LegOdom(s.declare_parameter("stance_tol", a.stance_tol).value, s.declare_parameter("sigma_v", a.sigma_v).value, s.declare_parameter("scale", a.scale).value)
            s.frame = s.declare_parameter("odom_frame", "leg_odom").value
            s.base = s.declare_parameter("base_frame", "base").value
            s.pub = s.create_publisher(Odometry, "/leg_odom", 20); s.imu = False
            s.create_subscription(Imu, "/imu/data", s.on_imu, 50); s.create_subscription(JointState, "/joint_states", s.on_js, 50)
            s.get_logger().info("leg_odom_go2 up: /joint_states + /imu/data -> /leg_odom")

        def on_imu(s, m):
            o, g = m.orientation, m.angular_velocity
            s.lo.set_imu(quat_to_R(o.x, o.y, o.z, o.w), (g.x, g.y, g.z)); s.imu = True

        def on_js(s, m):
            if not s.imu or len(m.velocity) < 12: return
            idx = {nm: i for i, nm in enumerate(m.name)}
            try:
                qp = {l: np.array([m.position[idx[j]] for j in JOINTS[l]]) for l in LEGS}
                qv = {l: np.array([m.velocity[idx[j]] for j in JOINTS[l]]) for l in LEGS}
            except KeyError: return
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if s.lo.update(t, qp, qv) is None: return
            o = Odometry(); o.header.stamp = m.header.stamp; o.header.frame_id = s.frame; o.child_frame_id = s.base
            o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, s.lo.pos)
            q = _quat_from_R(s.lo.R); o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z, o.pose.pose.orientation.w = q
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = map(float, s.lo.v_body)
            o.twist.twist.angular.x, o.twist.twist.angular.y, o.twist.twist.angular.z = map(float, s.lo.omega)
            c = [0.0] * 36
            for i in range(3): c[7 * i] = float(s.lo.cov_v[i, i]); c[7 * (i + 3)] = 0.01 ** 2   # gyro: tight
            o.twist.covariance = c
            pc = [0.0] * 36                                   # position grows with path; leave a loose, honest prior
            for i in range(3): pc[7 * i] = 1.0
            o.pose.covariance = pc
            s.pub.publish(o)

    rclpy.init(); n = N()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally: n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag"); ap.add_argument("--tum"); ap.add_argument("--decimate", type=int, default=4)
    ap.add_argument("--stance-tol", type=float, default=0.002); ap.add_argument("--sigma-v", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on the estimated velocity (calibrate against ground truth)")
    a, _ = ap.parse_known_args()
    if a.bag:
        if not a.tum: sys.exit("--tum is required with --bag")
        run_bag(a.bag, a.tum, a)
    else:
        run_ros(a)
