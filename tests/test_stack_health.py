import importlib.util
import pathlib
import sys

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("diagnostic_msgs")
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "stack_health", pathlib.Path(__file__).parents[1] / "scripts" / "stack_health_go2.py")
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


def test_all_ok():
    h = health.summarize([(True, "topic /odom", "10 Hz"), (True, "fall check", "z=0.3")])
    assert h["ok"] and h["failed"] == []


def test_failures_listed_sorted():
    h = health.summarize([(False, "topic /points", "no messages"), (True, "topic /odom", "ok"),
                          (False, "fall check", "tilt 80")])
    assert not h["ok"] and h["failed"] == ["fall check", "topic /points"]
    assert h["checks"]["topic /odom"]["ok"]
