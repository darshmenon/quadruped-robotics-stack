"""Go2 MuJoCo env on procedurally randomized rough/multi-terrain, with an
optional height-scan observation simulating a downward-facing depth camera.

Used to compare a "blind" policy (proprioception only, same observation
layout as Go2MujocoEnv but now walking on uneven ground) against a
"sighted" policy that also sees a local terrain height map, following the
same blind-vs-perceptive setup used in ANYmal/legged_gym locomotion
research.
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

try:
    from envs import go2_mujoco_env as base_module
    from envs.go2_mujoco_env import Go2MujocoEnv, OBS_DIM as BASE_OBS_DIM
except ImportError:
    import go2_mujoco_env as base_module
    from go2_mujoco_env import Go2MujocoEnv, OBS_DIM as BASE_OBS_DIM

ROUGH_SCENE_XML = os.path.join(os.path.dirname(__file__), "go2_rough_scene.xml")

# height-scan grid, in the base's local xy-plane (metres), matching the
# footprint a downward depth camera mounted on the chest would cover
SCAN_GRID = [
    (dx, dy)
    for dx in (-0.3, -0.15, 0.0, 0.15, 0.3, 0.45)
    for dy in (-0.2, 0.0, 0.2)
]
SCAN_DIM = len(SCAN_GRID)  # 18 points

N_OBSTACLES = 8
SPAWN_CLEARANCE = 1.0  # metres kept obstacle-free around the spawn point


class Go2MujocoVisionEnv(Go2MujocoEnv):
    """Rough-terrain Go2 env with a switchable height-scan observation.

    use_vision=False -> blind baseline: same observation layout as
        Go2MujocoEnv, just walking on uneven/multi-terrain ground.
    use_vision=True  -> sighted policy: same observation plus an 18-value
        local terrain height scan relative to the base.
    """

    def __init__(self, cmd=(0.5, 0.0, 0.0), render_mode=None,
                 randomize_domain=True, use_curriculum=True, use_vision=True,
                 terrain_difficulty=1.0):
        self.use_vision = use_vision
        self.terrain_difficulty = float(np.clip(terrain_difficulty, 0.0, 1.0))
        self._scan_geomid = np.zeros(1, dtype=np.int32)

        # Go2MujocoEnv.__init__ loads the module-level SCENE_XML constant by
        # name, so point it at the rough-terrain scene for the duration of
        # the base __init__ call, then restore it.
        original_scene = base_module.SCENE_XML
        base_module.SCENE_XML = ROUGH_SCENE_XML
        try:
            super().__init__(cmd=cmd, render_mode=render_mode,
                             randomize_domain=randomize_domain,
                             use_curriculum=use_curriculum)
        finally:
            base_module.SCENE_XML = original_scene

        self._hfield_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        self._nrow = int(self.model.hfield_nrow[self._hfield_id])
        self._ncol = int(self.model.hfield_ncol[self._hfield_id])
        self._hfield_x_radius = float(self.model.hfield_size[self._hfield_id, 0])
        self._hfield_y_radius = float(self.model.hfield_size[self._hfield_id, 1])
        self._hfield_elevation_z = float(self.model.hfield_size[self._hfield_id, 2])
        self._terrain_grid = np.zeros((self._nrow, self._ncol), dtype=np.float32)

        # restrict terrain/obstacle ray-casts to geom group 1 (floor +
        # obstacles, see go2_rough_scene.xml) so they can't self-intersect
        # the robot's own geoms, which sit in the default group 0
        self._terrain_geomgroup = np.zeros(6, dtype=np.uint8)
        self._terrain_geomgroup[1] = 1

        self._obstacle_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{i}")
            for i in range(N_OBSTACLES)
        ]

        obs_dim = BASE_OBS_DIM + (SCAN_DIM if use_vision else 0)
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # terrain generation
    # ------------------------------------------------------------------ #

    def _randomize_terrain(self) -> None:
        """Write a fresh multi-terrain height map: flat spawn patch, then a
        mix of random bumps, ramps, and rough noise scaled by curriculum
        level and terrain_difficulty."""
        rng = self.np_random
        level = self.curriculum_level * self.terrain_difficulty
        n = self._nrow * self._ncol
        grid = np.zeros((self._nrow, self._ncol), dtype=np.float32)

        # smooth low-frequency component (rolling ground / ramps)
        xs = np.linspace(-1.0, 1.0, self._ncol)
        ys = np.linspace(-1.0, 1.0, self._nrow)
        gx, gy = np.meshgrid(xs, ys)
        n_waves = int(rng.integers(1, 4))
        for _ in range(n_waves):
            fx, fy = rng.uniform(0.5, 2.5, size=2)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.1, 0.4) * level
            grid += amp * np.sin(fx * np.pi * gx + fy * np.pi * gy + phase)

        # high-frequency rough noise (gravel-like texture)
        grid += rng.uniform(-0.15, 0.15, size=grid.shape).astype(np.float32) * level

        # keep a flat spawn patch at the origin so the robot always starts
        # standing on level ground regardless of terrain_difficulty
        cx, cy = self._ncol // 2, self._nrow // 2
        pad = max(2, int(0.15 * min(self._ncol, self._nrow)))
        grid[cy - pad:cy + pad, cx - pad:cx + pad] = 0.0

        grid -= grid.min()
        if grid.max() > 1e-6:
            grid /= grid.max()

        # re-flatten the spawn patch to the exact zero datum *after*
        # normalization too, so it always sits at world height == geom
        # pos.z regardless of the surrounding terrain's min/max range —
        # otherwise the patch stays flat but can drift to any height
        # within [0, elevation_z], burying the robot's hardcoded spawn qpos
        grid[cy - pad:cy + pad, cx - pad:cx + pad] = 0.0

        self._terrain_grid = grid
        start = self._hfield_id * n
        self.model.hfield_data[start:start + n] = grid.reshape(-1)

        self._randomize_obstacles()

    def _terrain_height_at(self, x: float, y: float) -> float:
        """Height-field surface height (world z) below a given (x, y), read
        from the same grid just written to model.hfield_data — used to seat
        obstacles flush on the ground instead of ray-casting against a model
        that hasn't been mj_forward'd yet."""
        col = int(np.clip(
            (x / self._hfield_x_radius + 1.0) / 2.0 * (self._ncol - 1),
            0, self._ncol - 1))
        row = int(np.clip(
            (y / self._hfield_y_radius + 1.0) / 2.0 * (self._nrow - 1),
            0, self._nrow - 1))
        return float(self._terrain_grid[row, col]) * self._hfield_elevation_z

    def _randomize_obstacles(self) -> None:
        """Scatter discrete box obstacles (crates/steps) on top of the
        terrain, sized and counted by curriculum level so early training
        sees a clear course and later training sees a cluttered one."""
        rng = self.np_random
        level = self.curriculum_level * self.terrain_difficulty
        n_active = int(round(N_OBSTACLES * np.clip(level * 1.25, 0.0, 1.0)))

        for i, gid in enumerate(self._obstacle_geom_ids):
            if i >= n_active:
                self.model.geom_pos[gid] = [0.0, 0.0, -5.0]
                self.model.geom_size[gid] = [0.05, 0.05, 0.05]
                self.model.geom_quat[gid] = [1.0, 0.0, 0.0, 0.0]
                continue

            for _ in range(10):
                x = float(rng.uniform(-self._hfield_x_radius + 0.5,
                                       self._hfield_x_radius - 0.5))
                y = float(rng.uniform(-self._hfield_y_radius + 0.5,
                                       self._hfield_y_radius - 0.5))
                if np.hypot(x, y) >= SPAWN_CLEARANCE:
                    break

            size_scale = 0.4 + 0.6 * level
            half_x = float(rng.uniform(0.1, 0.35)) * size_scale
            half_y = float(rng.uniform(0.1, 0.35)) * size_scale
            half_z = float(rng.uniform(0.03, 0.12)) * size_scale
            yaw = float(rng.uniform(0.0, 2 * np.pi))

            terrain_z = self._terrain_height_at(x, y)
            self.model.geom_size[gid] = [half_x, half_y, half_z]
            self.model.geom_pos[gid] = [x, y, terrain_z + half_z]
            self.model.geom_quat[gid] = [
                np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]

    # ------------------------------------------------------------------ #
    # height-scan (simulated depth-camera height map)
    # ------------------------------------------------------------------ #

    def _cast_terrain_ray(self, x: float, y: float, from_z: float) -> float:
        origin = np.array([x, y, from_z])
        dist = mujoco.mj_ray(
            self.model, self.data, origin, np.array([0.0, 0.0, -1.0]),
            self._terrain_geomgroup, 1, -1, self._scan_geomid)
        return from_z - dist if dist >= 0 else 0.0

    def _height_scan(self) -> np.ndarray:
        base_pos = self.data.xpos[self._base_body_id]
        ray_start_z = base_pos[2] + 3.0
        heights = np.array([
            self._cast_terrain_ray(base_pos[0] + dx, base_pos[1] + dy, ray_start_z) - base_pos[2]
            for dx, dy in SCAN_GRID
        ], dtype=np.float32)
        return heights

    def _terrain_height_under_base(self) -> float:
        base_pos = self.data.xpos[self._base_body_id]
        return self._cast_terrain_ray(base_pos[0], base_pos[1], base_pos[2] + 3.0)

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        if not self.use_vision:
            return base_obs
        return np.concatenate([base_obs, self._height_scan()])

    # ------------------------------------------------------------------ #
    # reward / termination: measure base height relative to local terrain
    # instead of absolute world height, since the floor is no longer flat
    # ------------------------------------------------------------------ #

    def _compute_reward(self, action: np.ndarray):
        reward, components = super()._compute_reward(action)
        terrain_z = self._terrain_height_under_base()
        base_z = float(self.data.qpos[2])
        old_height_term = components["height"]
        new_height_term = -1.0 * ((base_z - terrain_z) - base_module.TARGET_HEIGHT) ** 2
        components["height"] = new_height_term
        reward += (new_height_term - old_height_term)
        return reward, components

    def _is_terminated(self) -> bool:
        terrain_z = self._terrain_height_under_base()
        height_above_terrain = float(self.data.qpos[2]) - terrain_z
        if height_above_terrain < 0.12 or height_above_terrain > 0.8:
            return True
        w, x, y, z_q = self.data.sensor("orientation").data
        return bool(1 - 2 * (w * w + z_q * z_q) > 0.5)

    def reset(self, *, seed=None, options=None):
        # seed self.np_random via the base gym.Env machinery *before*
        # generating terrain, so terrain layout is reproducible under a
        # given seed; Go2MujocoEnv.reset() below re-calls this with
        # seed=None, which Gymnasium treats as a no-op that keeps the RNG
        # we just seeded.
        gym.Env.reset(self, seed=seed)
        self._randomize_terrain()
        return super().reset(seed=None, options=options)


if __name__ == "__main__":
    for vision in (False, True):
        env = Go2MujocoVisionEnv(render_mode=None, randomize_domain=False,
                                 use_curriculum=True, use_vision=vision)
        obs, _ = env.reset(seed=0)
        expected = BASE_OBS_DIM + (SCAN_DIM if vision else 0)
        assert obs.shape == (expected,), f"expected {expected}, got {obs.shape[0]}"
        for _ in range(100):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()
        print(f"use_vision={vision}: obs shape {obs.shape} smoke-test passed")
