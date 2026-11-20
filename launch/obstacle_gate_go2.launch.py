"""Obstacle speed gate and stack health node for the Go2.

The gate reads /cmd_vel_in and writes /cmd_vel, so point the velocity source (Nav2, frontier explorer, teleop)
at /cmd_vel_in, e.g.  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_in

    ros2 launch launch/obstacle_gate_go2.launch.py slow_dist:=2.0 stop_dist:=0.7
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def generate_launch_description():
    args = {"slow_dist": "2.0", "stop_dist": "0.7", "use_sim_time": "true", "skip_obstacles": "false"}
    decl = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
    ros = lambda k: ["-p", [f"{k}:=", LaunchConfiguration(k)]]
    gate = ExecuteProcess(cmd=["python3", str(SCRIPTS / "obstacle_speed_gate_go2.py"), "--ros-args",
                               *ros("use_sim_time"), *ros("slow_dist"), *ros("stop_dist")], output="screen")
    health = ExecuteProcess(cmd=["python3", str(SCRIPTS / "stack_health_go2.py"), "--ros-args",
                                 *ros("use_sim_time"), *ros("skip_obstacles")], output="screen")
    return LaunchDescription(decl + [gate, health])
