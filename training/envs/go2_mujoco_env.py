"""MuJoCo-based Gymnasium environment for Unitree Go2 locomotion training."""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

SCENE_XML = os.path.join(os.path.dirname(__file__), "go2_scene.xml")

# Default joint angles matching go2_config.py (order: FL/FR/RL/RR hip,thigh,calf)
DEFAULT_QPOS = np.array([
    0.1,   # FL_hip
    0.8,   # FL_thigh
    -1.5,  # FL_calf
    -0.1,  # FR_hip
    0.8,   # FR_thigh
    -1.5,  # FR_calf
    0.1,   # RL_hip
    1.0,   # RL_thigh
    -1.5,  # RL_calf
    -0.1,  # RR_hip
    1.0,   # RR_thigh
    -1.5,  # RR_calf
], dtype=np.float32)

# Order matches actuator definition in go2_scene.xml:
# FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf...
ACT_DEFAULT = np.array([
    0.1, -0.1,  0.1, -0.1,   # hips
    0.8,  0.8,  1.0,  1.0,   # thighs
   -1.5, -1.5, -1.5, -1.5,  # calves
], dtype=np.float32)

OBS_DIM = 45   # 3 ang_vel + 3 gravity + 3 cmd + 12 dof_pos + 12 dof_vel + 12 prev_action
ACT_DIM = 12
ACT_SCALE = 0.25

# PD gains (from go2_config)
KP = 20.0
KD = 0.5

EPISODE_LEN_S = 20.0
SIM_DT = 0.005
CTRL_DECIMATION = 4   # policy runs at 50 Hz, sim at 200 Hz


class Go2MujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cmd=(0.5, 0.0, 0.0), render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        self.cmd = np.array(cmd, dtype=np.float32)  # lin_x, lin_y, ang_yaw
        self.render_mode = render_mode
        self._renderer = None
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._max_steps = int(EPISODE_LEN_S / (SIM_DT * CTRL_DECIMATION))

        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        # joint index map: qpos[7:] order from scene XML
        self._joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        # actuator order from scene XML: hips, thighs, calves per-leg-group
        self._act_default = ACT_DEFAULT.copy()

    # ------------------------------------------------------------------ #
    def _get_obs(self):
        d = self.data
        # base angular velocity (body frame)
        ang_vel = d.sensor("ang_vel").data.astype(np.float32) * 0.25

        # gravity vector in body frame
        quat = d.sensor("orientation").data.astype(np.float32)  # [w, x, y, z]
        w, x, y, z = quat
        gravity = np.array([
            2 * (-z * x - w * y),
            -2 * (z * y - w * x),
            1 - 2 * (w * w + z * z),
        ], dtype=np.float32)

        # command (scaled)
        cmd_scaled = self.cmd * np.array([2.0, 2.0, 0.25], dtype=np.float32)

        # joint positions & velocities (qpos[7:], qvel[6:] for freejoint robot)
        dof_pos = (d.qpos[7:].astype(np.float32) - DEFAULT_QPOS) * 1.0
        dof_vel = d.qvel[6:].astype(np.float32) * 0.05

        return np.concatenate([ang_vel, gravity, cmd_scaled, dof_pos, dof_vel, self._prev_action])

    def _compute_reward(self):
        d = self.data
        lin_vel = d.sensor("lin_vel").data.astype(np.float32)
        ang_vel = d.sensor("ang_vel").data.astype(np.float32)

        # tracking linear velocity
        r_lin = np.exp(-((lin_vel[0] - self.cmd[0])**2 + (lin_vel[1] - self.cmd[1])**2) / 0.25)
        # tracking angular velocity
        r_ang = np.exp(-((ang_vel[2] - self.cmd[2])**2) / 0.25)
        # penalise vertical velocity
        r_z = -2.0 * lin_vel[2]**2
        # penalise torques (from ctrl)
        r_torque = -0.0002 * float(np.sum(self.data.actuator_force**2))
        # penalise action rate
        r_act = -0.01 * float(np.sum((self._prev_action)**2))

        return r_lin + 0.5 * r_ang + r_z + r_torque + r_act

    def _is_terminated(self):
        z = self.data.qpos[2]
        # base too low (fallen) or too high
        if z < 0.15 or z > 0.8:
            return True
        # excessive tilt
        quat = self.data.sensor("orientation").data
        w, x, y, z_q = quat
        gravity_z = 1 - 2 * (w * w + z_q * z_q)
        if gravity_z > 0.5:   # more than ~60 deg tilt
            return True
        return False

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # set initial pose
        self.data.qpos[2] = 0.42
        self.data.qpos[3:7] = [1, 0, 0, 0]  # w,x,y,z identity quat
        self.data.qpos[7:] = DEFAULT_QPOS

        # small random perturbation for curriculum robustness
        if seed is not None or True:
            noise = (self.np_random.random(12) - 0.5) * 0.1
            self.data.qpos[7:] += noise

        self.data.ctrl[:] = self._act_default
        mujoco.mj_forward(self.model, self.data)

        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target = self._act_default + action * ACT_SCALE
        self.data.ctrl[:] = target

        for _ in range(CTRL_DECIMATION):
            mujoco.mj_step(self.model, self.data)

        self._prev_action = action.copy()
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self._max_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = mujoco.viewer.launch_passive(self.model, self.data)
            self._renderer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            if hasattr(self._renderer, "close"):
                self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = Go2MujocoEnv(render_mode=None)
    obs, _ = env.reset(seed=0)
    print("obs shape:", obs.shape)
    for _ in range(200):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
    print("env smoke-test passed")
