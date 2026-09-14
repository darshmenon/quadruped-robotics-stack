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
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64
    from std_srvs.srv import Empty
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

OBS_DIM = 45
ACT_DIM = 12
ACT_SCALE = 0.25
EPISODE_LEN_S = 20.0
CTRL_DT = 0.02  # 50 Hz

# go2 joint names (matches go2 URDF and ros2_control config). Leg-by-leg
# order -- used for observations (jpos in _build_obs), matching qpos's
# leg-by-leg order in go2_mujoco_env.py's DEFAULT_QPOS.
CHAMP_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

DEFAULT_QPOS = np.array([
    0.1,  0.8, -1.5,   # LF: hip, thigh, calf
   -0.1,  0.8, -1.5,   # RF
    0.1,  1.0, -1.5,   # LH
   -0.1,  1.0, -1.5,   # RH
], dtype=np.float32)

# Action order: grouped by joint type across all 4 legs, matching
# go2_mujoco_env.py's ACT_DEFAULT / <actuator> declaration order exactly --
# NOT leg-by-leg like CHAMP_JOINTS/DEFAULT_QPOS above (those are for
# observations). A legs_only-trained policy's 12-dim action vector is
# indexed this way; previously step() applied it through CHAMP_JOINTS
# (leg-by-leg) instead, so 10 of 12 actions landed on the wrong joint --
# e.g. index 1 ("FR_hip" in the trained policy's own frame) drove
# FL_thigh_joint here instead. Confirmed by direct comparison against
# go2_mujoco_env.py's ACT_DEFAULT ordering comment.
ACT_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
ACT_DEFAULT_QPOS = np.array([
    0.1, -0.1,  0.1, -0.1,
    0.8,  0.8,  1.0,  1.0,
   -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)


class _Go2RosNode(Node):
    def __init__(self):
        super().__init__("go2_rl_env")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.joint_pos = np.zeros(12, dtype=np.float32)
        self.joint_vel = np.zeros(12, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.orientation = np.array([1., 0., 0., 0.], dtype=np.float32)  # w,x,y,z
        self.lin_vel = np.zeros(3, dtype=np.float32)
        self._lock = threading.Lock()
        self._joint_order = None

        self.sub_js = self.create_subscription(
            JointState, "/joint_states", self._js_cb, qos)
        self.sub_imu = self.create_subscription(
            Imu, "/imu/data", self._imu_cb, qos)
        # base-frame linear velocity, published by scripts/gz_pose_to_odom.py
        # (finite-differenced from ground-truth Gazebo pose); step()'s reward
        # previously had nothing to compare cmd[0]/cmd[1] (linear velocity)
        # against and used ang_vel[0] instead, which rewards roll rate, not
        # forward progress.
        self.sub_odom = self.create_subscription(
            Odometry, "/odom", self._odom_cb, qos)

        # per-joint position publishers (Float64 → gz JointPositionController)
        self._joint_pubs = {
            name: self.create_publisher(Float64, f"/go2/cmd/{name}", 10)
            for name in CHAMP_JOINTS
        }

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

    def _odom_cb(self, msg):
        with self._lock:
            lv = msg.twist.twist.linear
            self.lin_vel[:] = [lv.x, lv.y, lv.z]

    def send_action(self, target_pos, joint_order=CHAMP_JOINTS):
        for i, name in enumerate(joint_order):
            msg = Float64()
            msg.data = float(target_pos[i])
            self._joint_pubs[name].publish(msg)

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

    def get_lin_vel(self):
        with self._lock:
            return self.lin_vel.copy()


class Go2GazeboEnv(gym.Env):
    """Gymnasium env that wraps a running Gazebo simulation via ROS2."""

    def __init__(self, cmd=(0.5, 0.0, 0.0), auto_launch=True,
                 ros_domain_id="177", gz_partition="go2rltrain",
                 qpos_reference=None):
        super().__init__()
        if not HAS_ROS:
            raise RuntimeError("rclpy not available. Source ROS2 setup.bash first.")

        self.cmd = np.array(cmd, dtype=np.float32)
        self._proc = None
        # dof_pos in _build_obs is (jpos - reference) -- must match whatever
        # pose a given policy was trained relative to. Defaults to this env's
        # own standing pose (DEFAULT_QPOS); pass e.g. the recovery env's
        # ACT_CURL (curled) reference when evaluating a fall-recovery policy,
        # otherwise dof_pos is silently computed against the wrong baseline.
        self._qpos_reference = (
            DEFAULT_QPOS if qpos_reference is None
            else np.asarray(qpos_reference, dtype=np.float32)
        )

        # Isolate from other concurrent ROS2/Gazebo sessions on this machine
        # (e.g. launch/slam3d_go2.launch.py on domain 157) -- both the
        # subprocess-launched sim below and this process's own rclpy node
        # need to agree on the same domain/partition to see each other's
        # topics/services.
        import os
        os.environ["ROS_DOMAIN_ID"] = str(ros_domain_id)
        os.environ["GZ_PARTITION"] = str(gz_partition)

        if auto_launch:
            self._launch_gazebo(ros_domain_id, gz_partition)

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

    def _launch_gazebo(self, ros_domain_id, gz_partition):
        import os
        pkg_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        launch_file = os.path.join(pkg_dir, "training", "launch", "gazebo_rl.launch.py")
        cmd = [
            "bash", "-c",
            f"source /opt/ros/humble/setup.bash && "
            f"source {pkg_dir}/ros2/install/setup.bash && "
            f"ros2 launch {launch_file} headless:=true "
            f"ros_domain_id:={ros_domain_id} gz_partition:={gz_partition}"
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5.0)  # wait for Gazebo to come up

    def _build_obs(self):
        ang_vel, quat, jpos, jvel = self._node.get_obs_data()
        w, x, y, z = quat
        # Matches go2_mujoco_env.py's _gravity_vec() exactly, cross terms
        # included -- that formula has negated x/y signs relative to the
        # true projected gravity vector (verified against MuJoCo's own xmat;
        # introduced by commit 4345778, "fix gravity projection", which was
        # itself the bug). Every MuJoCo checkpoint ever trained (including
        # legs_only, the one meant to load here) learned balance behavior
        # against that flipped convention, so this env must reproduce the
        # same flip rather than the physically correct one, or a legs_only
        # checkpoint sees a mirrored tilt reading the moment it's deployed
        # here -- which is the whole point of legs_only matching this env.
        gravity = np.array([
            2 * (-z * x - w * y),
            -2 * (z * y - w * x),
            1 - 2 * (w * w + z * z),
        ], dtype=np.float32)
        cmd_scaled = self.cmd * np.array([2.0, 2.0, 0.25], dtype=np.float32)
        dof_pos = (jpos - self._qpos_reference) * 1.0
        dof_vel = jvel * 0.05
        return np.concatenate([
            ang_vel * 0.25, gravity, cmd_scaled,
            dof_pos, dof_vel, self._prev_action,
        ]).astype(np.float32)

    def _is_terminated(self):
        _, quat, _, _ = self._node.get_obs_data()
        w, x, y, z = quat
        # -cos(tilt); see go2_mujoco_env.py's _is_terminated for the identity
        # derivation. -0.5 is ~60 deg -- the old `> 0.5` threshold here only
        # tripped past 120 deg, the same stale-threshold bug already fixed
        # for the MuJoCo backend but never ported to this one.
        gravity_z = 1 - 2 * (w * w + z * z)
        return gravity_z > -0.5

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._node.reset_sim()
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        time.sleep(0.2)
        return self._build_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # ACT_DEFAULT_QPOS/ACT_JOINT_ORDER, not DEFAULT_QPOS/CHAMP_JOINTS --
        # the action vector is in actuator order (see ACT_JOINT_ORDER
        # comment above), a different ordering than the leg-by-leg one used
        # for observations.
        target = ACT_DEFAULT_QPOS + action * ACT_SCALE
        self._node.send_action(target, ACT_JOINT_ORDER)
        time.sleep(CTRL_DT)

        self._prev_action = action.copy()
        self._step_count += 1

        obs = self._build_obs()
        ang_vel, quat, _, _ = self._node.get_obs_data()
        lin_vel = self._node.get_lin_vel()

        # was comparing ang_vel[0] (roll rate) against cmd[0] (linear speed
        # target) -- rewarded matching roll rate to a forward-speed number,
        # never actual forward progress, which is the root cause of the
        # README's documented "forward walking trips the fall-detector
        # instead of translating" issue: nothing in the reward ever asked
        # for translation.
        r_lin = float(np.exp(-((lin_vel[0] - self.cmd[0])**2 + (lin_vel[1] - self.cmd[1])**2) / 0.25))
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
