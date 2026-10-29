"""Pure-numpy checks for the NMPC -> MuJoCo layout conversion used by imitation/collect_nmpc.py."""
import numpy as np

QSDK_LEG_OF = {"FL": 0, "FR": 2, "RL": 1, "RR": 3}  # mirrors collect_nmpc.py (rclpy not importable in CI)


def _reorder(q):
    return np.concatenate([q[3 * QSDK_LEG_OF[l]: 3 * QSDK_LEG_OF[l] + 3] for l in ("FL", "FR", "RL", "RR")])


def _to_actuator_order(q):
    a = q.reshape(4, 3)
    return np.concatenate([a[:, 0], a[:, 1], a[:, 2]])


def test_leg_reorder_fl_bl_fr_br_to_fl_fr_rl_rr():
    q = np.arange(12.0)  # leg i (FL,BL,FR,BR) joints = 3i..3i+2
    out = _reorder(q)
    assert list(out[:3]) == [0, 1, 2]      # FL
    assert list(out[3:6]) == [6, 7, 8]     # FR
    assert list(out[6:9]) == [3, 4, 5]     # RL (= BL)
    assert list(out[9:]) == [9, 10, 11]    # RR (= BR)


def test_actuator_order_is_hips_thighs_calves():
    q = np.arange(12.0)
    a = _to_actuator_order(q)
    assert list(a[:4]) == [0, 3, 6, 9]
    assert list(a[4:8]) == [1, 4, 7, 10]
    assert list(a[8:]) == [2, 5, 8, 11]


def test_history_stack_repeats_first_frame():
    h = 3
    obs = np.arange(5.0).reshape(5, 1)
    pad = np.concatenate([np.repeat(obs[:1], h - 1, axis=0), obs])
    out = np.concatenate([pad[i:i + len(obs)] for i in range(h)], axis=1)
    assert list(out[0]) == [0, 0, 0] and list(out[3]) == [1, 2, 3]
