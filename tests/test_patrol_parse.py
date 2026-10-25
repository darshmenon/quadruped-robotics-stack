import importlib.util
import pathlib
import sys
import types

# patrol_go2 imports ROS at module level; stub it so the parser is testable anywhere.
for name in ("rclpy", "rclpy.action", "rclpy.node", "geometry_msgs", "geometry_msgs.msg",
             "nav2_msgs", "nav2_msgs.action"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["rclpy.action"].ActionClient = object
sys.modules["rclpy.node"].Node = object
sys.modules["geometry_msgs.msg"].PoseStamped = object
sys.modules["nav2_msgs.action"].NavigateToPose = object

spec = importlib.util.spec_from_file_location(
    "patrol_go2", pathlib.Path(__file__).parents[1] / "scripts" / "patrol_go2.py")
patrol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patrol)


def test_parse_waypoints():
    assert patrol.parse_waypoints(["1,0", "-2.5,3"]) == [(1.0, 0.0), (-2.5, 3.0)]
