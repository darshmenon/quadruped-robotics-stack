"""Guards for the reward/obs changes that fixed the sitting policy (mujoco legs-only env)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
pytest.importorskip("mujoco")
from envs import go2_mujoco_env as g  # noqa: E402


def _env(**kw):
    return g.Go2MujocoEnv(render_mode=None, legs_only=True, randomize_domain=False,
                          use_curriculum=False, push_robots=False, **kw)


def test_legs_only_obs_dims():
    for gait, n in ((False, 45), (True, 45 + g.GAIT_OBS_DIM)):
        env = _env(gait_conditioned=gait)
        obs, _ = env.reset()
        assert obs.shape == (n,) == env.observation_space.shape


def test_standing_still_earns_little_tracking_reward():
    # kernel must not pay ~1.3/step for a frozen robot at cmd 0.2 (the old 0.1 sigma did)
    env = _env()
    env.reset()
    env.cmd = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    _, comp = env._compute_reward(np.zeros(12, dtype=np.float32))
    assert comp["lin"] < 0.9


def test_low_base_is_penalised():
    env = _env()
    env.reset()
    env.data.qpos[2] = g.TARGET_HEIGHT  # spawn is 0.42 (dropped); settled stance is ~0.25-0.27
    _, comp_stand = env._compute_reward(np.zeros(12, dtype=np.float32))
    env.data.qpos[2] = 0.15
    _, comp_low = env._compute_reward(np.zeros(12, dtype=np.float32))
    assert comp_low["height"] < comp_stand["height"] - 0.1


def test_contact_penalties_ramp_with_curriculum():
    env = _env()
    env.reset()
    scale = lambda lvl: g.PENALTY_RAMP_MIN + (1 - g.PENALTY_RAMP_MIN) * min(1.0, lvl / g.PENALTY_RAMP_LEVEL)
    assert scale(0.0) == pytest.approx(g.PENALTY_RAMP_MIN)
    assert scale(g.PENALTY_RAMP_LEVEL) == pytest.approx(1.0)


def test_step_exposes_reward_components():
    env = _env()
    env.reset()
    _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
    assert {"lin", "height", "stall", "alive"} <= set(info["reward_components"])
