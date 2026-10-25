"""Go2 MuJoCo env on the fixed stairs / ledge / gap course.

Loads `go2_stairs_scene.xml` (easy→mid→hard solid stairs, platforms, gaps,
hollow open-riser steps along +X). Flat spawn at the origin.

Optional 18-point height-scan observation (same layout as the rough-terrain
vision env) for blind vs. sighted stair climbing, following Blind Stair
Climbing / StairMaster-style proprioception + local height perception.
"""

from __future__ import annotations

import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

try:
    from envs import go2_mujoco_env as base_module
    from envs.go2_mujoco_env import Go2MujocoEnv, OBS_DIM as BASE_OBS_DIM
    from envs.go2_mujoco_vision_env import SCAN_DIM, SCAN_GRID
except ImportError:
    import go2_mujoco_env as base_module
    from go2_mujoco_env import Go2MujocoEnv, OBS_DIM as BASE_OBS_DIM
    from go2_mujoco_vision_env import SCAN_DIM, SCAN_GRID

STAIRS_SCENE_XML = os.path.join(os.path.dirname(__file__), "go2_stairs_scene.xml")

# Curriculum gates: encourage reaching farther along +X before speeding up.
PROGRESS_WEIGHT = 1.2
MAX_CURRICULUM_LEVEL = 0.85


class Go2MujocoStairsEnv(Go2MujocoEnv):
    """Fixed stair/ledge course with optional height-scan vision."""

    def __init__(
        self,
        cmd=(0.4, 0.0, 0.0),
        render_mode=None,
        randomize_domain=True,
        use_curriculum=True,
        use_vision=True,
        initial_curriculum_level=0.0,
        disable_arm_reach=True,
        gait_conditioned=False,
        gait_name="trotting",
        legs_only=False,
    ):
        self.use_vision = use_vision
        self.disable_arm_reach = disable_arm_reach
        self._scan_geomid = np.zeros(1, dtype=np.int32)
        self._prev_x = 0.0

        original_scene = base_module.SCENE_XML
        base_module.SCENE_XML = STAIRS_SCENE_XML
        try:
            super().__init__(
                cmd=cmd,
                render_mode=render_mode,
                randomize_domain=randomize_domain,
                use_curriculum=use_curriculum,
                initial_curriculum_level=initial_curriculum_level,
                gait_conditioned=gait_conditioned,
                gait_name=gait_name,
                legs_only=legs_only,
            )
        finally:
            base_module.SCENE_XML = original_scene

        # Cap curriculum the same way as the flat env (avoid runaway difficulty).
        self.curriculum_level = float(
            np.clip(self.curriculum_level, 0.0, MAX_CURRICULUM_LEVEL)
        )

        self._terrain_geomgroup = np.zeros(6, dtype=np.uint8)
        self._terrain_geomgroup[1] = 1  # floor + stair/ledge geoms

        base_dim = int(self.observation_space.shape[0])
        obs_dim = base_dim + (SCAN_DIM if use_vision else 0)
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # height rays (group 1 = floor + stairs/ledges)
    # ------------------------------------------------------------------ #

    def _cast_terrain_ray(self, x: float, y: float, from_z: float) -> float:
        origin = np.array([x, y, from_z])
        dist = mujoco.mj_ray(
            self.model,
            self.data,
            origin,
            np.array([0.0, 0.0, -1.0]),
            self._terrain_geomgroup,
            1,
            -1,
            self._scan_geomid,
        )
        return from_z - dist if dist >= 0 else 0.0

    def _height_scan(self) -> np.ndarray:
        base_pos = self.data.xpos[self._base_body_id]
        ray_start_z = base_pos[2] + 3.0
        return np.array(
            [
                self._cast_terrain_ray(base_pos[0] + dx, base_pos[1] + dy, ray_start_z)
                - base_pos[2]
                for dx, dy in SCAN_GRID
            ],
            dtype=np.float32,
        )

    def _terrain_height_under_base(self) -> float:
        base_pos = self.data.xpos[self._base_body_id]
        return self._cast_terrain_ray(base_pos[0], base_pos[1], base_pos[2] + 3.0)

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        if not self.use_vision:
            return base_obs
        return np.concatenate([base_obs, self._height_scan()])

    # ------------------------------------------------------------------ #
    # command / reward / termination
    # ------------------------------------------------------------------ #

    def _sample_cmd(self) -> np.ndarray:
        """Bias forward along +X (toward the stair course); speed scales with curriculum."""
        max_vx = 0.25 + 0.75 * self.curriculum_level
        vx = float(self.np_random.uniform(0.15, max_vx))
        vy = float(self.np_random.uniform(-0.1, 0.1)) * self.curriculum_level
        wz = float(self.np_random.uniform(-0.3, 0.3)) * self.curriculum_level
        return np.array([vx, vy, wz], dtype=np.float32)

    def _compute_reward(self, action: np.ndarray):
        reward, components = super()._compute_reward(action)

        # Height relative to local stair/ledge surface, not world z=0.
        terrain_z = self._terrain_height_under_base()
        base_z = float(self.data.qpos[2])
        old_h = components["height"]
        new_h = -1.0 * ((base_z - terrain_z) - base_module.TARGET_HEIGHT) ** 2
        components["height"] = new_h
        reward += new_h - old_h

        # Forward progress along the course (encourages climbing, not spinning in place).
        x = float(self.data.qpos[0])
        dx = x - self._prev_x
        self._prev_x = x
        r_progress = PROGRESS_WEIGHT * float(np.clip(dx, -0.05, 0.08))
        components["progress"] = r_progress
        reward += r_progress

        if self.disable_arm_reach:
            for key in ("reach", "reach_dense", "reach_bonus"):
                reward -= components.get(key, 0.0)
                components[key] = 0.0

        # Soften pitch penalty a bit — stairs need nose-up / nose-down attitudes.
        old_orient = components["orient"]
        gravity = self._gravity_vec()
        new_orient = -0.35 * float(gravity[0] ** 2 + gravity[1] ** 2)
        components["orient"] = new_orient
        reward += new_orient - old_orient

        return reward, components

    def _is_terminated(self) -> bool:
        terrain_z = self._terrain_height_under_base()
        height_above = float(self.data.qpos[2]) - terrain_z
        # Allow slightly higher peaks on ledges / stair landings.
        if height_above < 0.12 or height_above > 1.0:
            return True
        w, x, y, z_q = self.data.sensor("orientation").data
        return bool(1 - 2 * (w * w + z_q * z_q) > 0.5)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.curriculum_level = float(
            np.clip(self.curriculum_level, 0.0, MAX_CURRICULUM_LEVEL)
        )
        self._prev_x = float(self.data.qpos[0])
        info["x"] = self._prev_x
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["x"] = float(self.data.qpos[0])
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    for vision in (False, True):
        env = Go2MujocoStairsEnv(
            render_mode=None,
            randomize_domain=False,
            use_curriculum=True,
            use_vision=vision,
        )
        obs, info = env.reset(seed=0)
        expected = BASE_OBS_DIM + (SCAN_DIM if vision else 0)
        assert obs.shape == (expected,), f"expected {expected}, got {obs.shape[0]}"
        for _ in range(100):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                obs, info = env.reset()
        print(
            f"use_vision={vision}: obs={obs.shape} x={info.get('x', 0):.2f} smoke-test ok"
        )
