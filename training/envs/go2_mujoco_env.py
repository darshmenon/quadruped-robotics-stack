"""MuJoCo-based Gymnasium environment for Unitree Go2 locomotion training."""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

SCENE_XML = os.path.join(os.path.dirname(__file__), "go2_scene.xml")

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

# Actuator order: [FL_hip, FR_hip, RL_hip, RR_hip,
#                  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#                  FL_calf, FR_calf, RL_calf, RR_calf]
ACT_DEFAULT = np.array([
    0.1, -0.1,  0.1, -0.1,
    0.8,  0.8,  1.0,  1.0,
   -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)

OBS_DIM = 49   # 3 ang_vel + 3 gravity + 3 cmd + 12 dof_pos + 12 dof_vel + 12 prev_action + 4 contacts
ACT_DIM = 12
ACT_SCALE = 0.25

EPISODE_LEN_S = 20.0
SIM_DT = 0.005
CTRL_DECIMATION = 4   # policy at 50 Hz, sim at 200 Hz

TARGET_HEIGHT = 0.27  # nominal base height above ground while standing


class Go2MujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cmd=(0.5, 0.0, 0.0), render_mode=None,
                 randomize_domain=True, use_curriculum=True):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        self.cmd = np.array(cmd, dtype=np.float32)
        self.render_mode = render_mode
        self.randomize_domain = randomize_domain
        self.use_curriculum = use_curriculum
        self._renderer = None
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._max_steps = int(EPISODE_LEN_S / (SIM_DT * CTRL_DECIMATION))
        self._last_episode_steps = self._max_steps

        self.curriculum_level = 0.0

        # cache original model params for domain randomization
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._base_mass = float(self.model.body_mass[self._base_body_id])
        self._base_floor_friction = self.model.geom_friction[self._floor_geom_id].copy()
        self._base_gainprm = self.model.actuator_gainprm[:, 0].copy()
        self._base_biasprm1 = self.model.actuator_biasprm[:, 1].copy()

        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self._act_default = ACT_DEFAULT.copy()

    # ------------------------------------------------------------------ #

    def _gravity_vec(self) -> np.ndarray:
        w, x, y, z = self.data.sensor("orientation").data.astype(np.float32)
        return np.array([
            2 * (-z * x - w * y),
            -2 * (z * y - w * x),
            1 - 2 * (w * w + z * z),
        ], dtype=np.float32)

    def _get_contacts(self) -> np.ndarray:
        raw = np.array(
            [self.data.sensor(n).data[0]
             for n in ("FL_contact", "FR_contact", "RL_contact", "RR_contact")],
            dtype=np.float32)
        return np.clip(raw / 50.0, 0.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        d = self.data
        ang_vel   = d.sensor("ang_vel").data.astype(np.float32) * 0.25
        gravity   = self._gravity_vec()
        cmd_scaled = self.cmd * np.array([2.0, 2.0, 0.25], dtype=np.float32)
        dof_pos   = (d.qpos[7:].astype(np.float32) - DEFAULT_QPOS)
        dof_vel   = d.qvel[6:].astype(np.float32) * 0.05
        contacts  = self._get_contacts()
        return np.concatenate(
            [ang_vel, gravity, cmd_scaled, dof_pos, dof_vel, self._prev_action, contacts])

    def _compute_reward(self, action: np.ndarray):
        d = self.data
        lin_vel  = d.sensor("lin_vel").data.astype(np.float32)
        ang_vel  = d.sensor("ang_vel").data.astype(np.float32)
        gravity  = self._gravity_vec()

        r_lin    = 2.0 * float(np.exp(
            -((lin_vel[0] - self.cmd[0])**2 + (lin_vel[1] - self.cmd[1])**2) / 0.1))
        r_ang    = 0.5 * float(np.exp(-((ang_vel[2] - self.cmd[2])**2) / 0.25))
        r_z      = -2.0  * float(lin_vel[2]**2)
        r_height = -1.0  * (float(d.qpos[2]) - TARGET_HEIGHT)**2
        r_orient = -0.5  * float(gravity[0]**2 + gravity[1]**2)
        r_torque = -2e-4 * float(np.sum(d.actuator_force**2))
        r_smooth = -5e-3 * float(np.sum((action - self._prev_action)**2))

        contacts  = self._get_contacts()
        r_contact = 0.15 * min(float(np.sum(contacts > 0.3)) / 2.0, 1.0)

        # Explicit stall penalty: standing still while a real command is
        # active must never out-earn walking, no matter how forgiving the
        # tracking kernel above is (previously the policy converged to
        # standing still — see README "Known issue").
        cmd_speed = float(np.hypot(self.cmd[0], self.cmd[1]))
        actual_speed = float(np.hypot(lin_vel[0], lin_vel[1]))
        r_stall = -0.6 if (cmd_speed > 0.15 and actual_speed < 0.3 * cmd_speed) else 0.0

        components = dict(
            lin=r_lin, ang=r_ang, vz=r_z, height=r_height,
            orient=r_orient, torque=r_torque, smooth=r_smooth, contact=r_contact,
            stall=r_stall,
        )
        return float(sum(components.values())), components

    def _sample_cmd(self) -> np.ndarray:
        max_vx = 0.3 + 0.9 * self.curriculum_level
        vx = float(self.np_random.uniform(-0.1, max_vx))
        vy = float(self.np_random.uniform(-0.2, 0.2)) * self.curriculum_level
        wz = float(self.np_random.uniform(-0.5, 0.5)) * self.curriculum_level
        return np.array([vx, vy, wz], dtype=np.float32)

    def _apply_domain_rand(self) -> None:
        if not self.randomize_domain:
            return
        rng = self.np_random
        self.model.body_mass[self._base_body_id] = (
            self._base_mass * float(rng.uniform(0.85, 1.15)))
        self.model.geom_friction[self._floor_geom_id] = (
            self._base_floor_friction * float(rng.uniform(0.7, 1.3)))
        kp_scale = float(rng.uniform(0.85, 1.15))
        self.model.actuator_gainprm[:, 0] = self._base_gainprm * kp_scale
        self.model.actuator_biasprm[:, 1] = self._base_biasprm1 * kp_scale

    def _is_terminated(self) -> bool:
        z = float(self.data.qpos[2])
        if z < 0.15 or z > 0.8:
            return True
        w, x, y, z_q = self.data.sensor("orientation").data
        return bool(1 - 2 * (w * w + z_q * z_q) > 0.5)

    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.use_curriculum:
            success = self._last_episode_steps >= 0.75 * self._max_steps
            self.curriculum_level = float(np.clip(
                self.curriculum_level + (0.005 if success else -0.002), 0.0, 1.0))
            self.cmd = self._sample_cmd()

        mujoco.mj_resetData(self.model, self.data)
        self._apply_domain_rand()

        self.data.qpos[2]   = 0.42
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[7:]  = DEFAULT_QPOS + (self.np_random.random(12) - 0.5) * 0.1
        self.data.ctrl[:]   = self._act_default
        mujoco.mj_forward(self.model, self.data)

        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._last_episode_steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.data.ctrl[:] = self._act_default + action * ACT_SCALE
        for _ in range(CTRL_DECIMATION):
            mujoco.mj_step(self.model, self.data)

        reward, components = self._compute_reward(action)
        self._prev_action = action.copy()
        self._step_count += 1
        self._last_episode_steps = self._step_count

        obs = self._get_obs()
        terminated = self._is_terminated()
        truncated  = self._step_count >= self._max_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {"reward_components": components}

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
    env = Go2MujocoEnv(render_mode=None, randomize_domain=False, use_curriculum=False)
    obs, _ = env.reset(seed=0)
    print("obs shape:", obs.shape)
    assert obs.shape == (OBS_DIM,), f"expected {OBS_DIM}, got {obs.shape[0]}"
    for _ in range(200):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
    print("env smoke-test passed")
