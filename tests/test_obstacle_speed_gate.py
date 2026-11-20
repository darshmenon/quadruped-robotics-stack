import importlib.util
import math
import pathlib

spec = importlib.util.spec_from_file_location(
    "gate", pathlib.Path(__file__).parents[1] / "scripts" / "obstacle_speed_gate_go2.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

ROBOT = (0.0, 0.0, 0.0)          # at the origin, facing +x
FWD = (0.4, 0.0)


def obs(x, y, vx=0.0, vy=0.0, id=1):
    return {"id": id, "x": x, "y": y, "vx": vx, "vy": vy, "width": 0.0, "length": 0.0}


def test_no_obstacles_full_speed():
    assert gate.gate_scale(ROBOT, FWD, []) == (1.0, None)


def test_far_obstacle_ignored():
    assert gate.gate_scale(ROBOT, FWD, [obs(5.0, 0.0, -0.3)])[0] == 1.0


def test_moving_obstacle_ahead_ramps_down():
    far = gate.gate_scale(ROBOT, FWD, [obs(1.8, 0.0, 0.3)])[0]    # moving away: only the distance ramp applies
    near = gate.gate_scale(ROBOT, FWD, [obs(1.0, 0.0, 0.3)])[0]
    assert 0.0 < near < far < 1.0


def test_inside_stop_distance_stops():
    assert gate.gate_scale(ROBOT, FWD, [obs(0.5, 0.0, 0.2)]) == (0.0, 1)


def test_obstacle_behind_ignored():
    assert gate.gate_scale(ROBOT, FWD, [obs(-1.0, 0.0, 0.3)])[0] == 1.0


def test_parked_obstacle_left_to_planner():
    assert gate.gate_scale(ROBOT, FWD, [obs(1.2, 0.0, 0.0)])[0] == 1.0


def test_closing_obstacle_slows_more_than_receding():
    closing = gate.gate_scale(ROBOT, FWD, [obs(1.9, 0.0, -0.8)])[0]
    receding = gate.gate_scale(ROBOT, FWD, [obs(1.9, 0.0, 0.8)])[0]
    assert closing < receding


def test_pure_rotation_not_gated():
    assert gate.gate_scale(ROBOT, (0.0, 0.0), [obs(0.3, 0.0, 0.5)])[0] == 1.0


def test_heading_is_map_frame():
    # robot faces +y: an obstacle on +y is ahead, one on +x is to the side
    r = (0.0, 0.0, math.pi / 2)
    assert gate.gate_scale(r, FWD, [obs(0.0, 0.5, 0.2)])[0] == 0.0
    assert gate.gate_scale(r, FWD, [obs(1.0, 0.0, 0.2)])[0] == 1.0
