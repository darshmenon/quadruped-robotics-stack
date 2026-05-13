"""
Gymnasium environment for Unitree Go2 training via Gazebo + ROS2.

Architecture:
  gz sim (headless) <-> ros_gz_bridge <-> this env via rclpy

Requires Gazebo to be launched separately:
  ros2 launch training/launch/gazebo_rl.launch.py headless:=true
"""

import time
import subprocess
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState, Imu
    from std_msgs.msg import Float64MultiArray
    from std_srvs.srv import Empty
    from geometry_msgs.msg import Twist
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

OBS_DIM = 45
ACT_DIM = 12
ACT_SCALE = 0.25
EPISODE_LEN_S = 20.0
CTRL_DT = 0.02  # 50 Hz

# go2 joint names (champ URDF mapping)
CHAMP_JOINTS = [
    "lf_hip_joint", "lf_upper_leg_joint", "lf_lower_leg_joint",
    "rf_hip_joint", "rf_upper_leg_joint", "rf_lower_leg_joint",
    "lh_hip_joint", "lh_upper_leg_joint", "lh_lower_leg_joint",
    "rh_hip_joint", "rh_upper_leg_joint", "rh_lower_leg_joint",
]

DEFAULT_QPOS = np.array([
    0.1,  0.8, -1.5,   # LF: hip, thigh, calf
   -0.1,  0.8, -1.5,   # RF
    0.1,  1.0, -1.5,   # LH
   -0.1,  1.0, -1.5,   # RH
], dtype=np.float32)


class _Go2RosNode(Node):
    def __init__(self):
        super().__init__("go2_rl_env")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.joint_pos = np.zeros(12, dtype=np.float32)
        self.joint_vel = np.zeros(12, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.orientation = np.array([1., 0., 0., 0.], dtype=np.float32)  # w,x,y,z
        self._lock = threading.Lock()
        self._joint_order = None

        self.sub_js = self.create_subscription(
            JointState, "/joint_states", self._js_cb, qos)
        self.sub_imu = self.create_subscription(
            Imu, "/imu/data", self._imu_cb, qos)

        self.pub_cmd = self.create_publisher(
            Float64MultiArray,
            "/joint_group_effort_controller/commands",
            qos_profile=10,
        )

        self.reset_client = self.create_client(Empty, "/reset_simulation")

    def _js_cb(self, msg):
        if self._joint_order is None:
            self._joint_order = {n: i for i, n in enumerate(msg.name)}
        with self._lock:
            for i, name in enumerate(CHAMP_JOINTS):
                idx = self._joint_order.get(name)
                if idx is not None:
                    self.joint_pos[i] = msg.position[idx]
                    self.joint_vel[i] = msg.velocity[idx]

    def _imu_cb(self, msg):
        with self._lock:
            q = msg.orientation
            self.orientation[:] = [q.w, q.x, q.y, q.z]
            av = msg.angular_velocity
            self.ang_vel[:] = [av.x, av.y, av.z]

    def send_action(self, target_pos):
        # PD control: effort = kp*(target - pos) + kd*(0 - vel)
        with self._lock:
            err = target_pos - self.joint_pos
            effort = 20.0 * err - 0.5 * self.joint_vel
        msg = Float64MultiArray()
        msg.data = effort.tolist()
        self.pub_cmd.publish(msg)

    def reset_sim(self):
        if self.reset_client.wait_for_service(timeout_sec=2.0):
            req = Empty.Request()
            self.reset_client.call_async(req)
        time.sleep(0.5)

    def get_obs_data(self):
        with self._lock:
            return (
                self.ang_vel.copy(),
                self.orientation.copy(),
                self.joint_pos.copy(),
                self.joint_vel.copy(),
            )


class Go2GazeboEnv(gym.Env):
    """Gymnasium env that wraps a running Gazebo simulation via ROS2."""

    def __init__(self, cmd=(0.5, 0.0, 0.0), auto_launch=True):
        super().__init__()
        if not HAS_ROS:
            raise RuntimeError("rclpy not available. Source ROS2 setup.bash first.")

        self.cmd = np.array(cmd, dtype=np.float32)
        self._proc = None

        if auto_launch:
            self._launch_gazebo()

        if not rclpy.ok():
            rclpy.init()
        self._node = _Go2RosNode()
        self._executor = rclpy.executors.MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # wait for first joint state
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self._node._joint_order is not None:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Timed out waiting for /joint_states. Is Gazebo running?")

        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)

        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._max_steps = int(EPISODE_LEN_S / CTRL_DT)

    def _launch_gazebo(self):
        import os
        pkg_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        launch_file = os.path.join(pkg_dir, "training", "launch", "gazebo_rl.launch.py")
        cmd = [
            "bash", "-c",
            f"source /opt/ros/humble/setup.bash && "
            f"source {pkg_dir}/ros2/install/setup.bash && "
            f"ros2 launch {launch_file} headless:=true"
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5.0)  # wait for Gazebo to come up

    def _build_obs(self):
        ang_vel, quat, jpos, jvel = self._node.get_obs_data()
        w, x, y, z = quat
        gravity = np.array([
            2 * (-z * x + w * y),
            -2 * (z * y + w * x),
            1 - 2 * (w * w + z * z),
        ], dtype=np.float32)
        cmd_scaled = self.cmd * np.array([2.0, 2.0, 0.25], dtype=np.float32)
        dof_pos = (jpos - DEFAULT_QPOS) * 1.0
        dof_vel = jvel * 0.05
        return np.concatenate([
            ang_vel * 0.25, gravity, cmd_scaled,
            dof_pos, dof_vel, self._prev_action,
        ]).astype(np.float32)

    def _is_terminated(self):
        _, quat, _, _ = self._node.get_obs_data()
        w, x, y, z = quat
        gravity_z = 1 - 2 * (w * w + z * z)
        return gravity_z > 0.5   # ~60 deg tilt = fallen

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._node.reset_sim()
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        time.sleep(0.2)
        return self._build_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target = DEFAULT_QPOS + action * ACT_SCALE
        self._node.send_action(target)
        time.sleep(CTRL_DT)

        self._prev_action = action.copy()
        self._step_count += 1

        obs = self._build_obs()
        ang_vel, quat, _, _ = self._node.get_obs_data()

        r_lin = float(np.exp(-((ang_vel[0] - self.cmd[0])**2) / 0.25))
        r_ang = float(np.exp(-((ang_vel[2] - self.cmd[2])**2) / 0.25))
        reward = r_lin + 0.5 * r_ang

        terminated = self._is_terminated()
        truncated = self._step_count >= self._max_steps
        return obs, reward, terminated, truncated, {}

    def close(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        self._executor.shutdown(wait=False)
        if rclpy.ok():
            rclpy.shutdown()
