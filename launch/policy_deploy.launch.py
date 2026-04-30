"""
Launch file: Deploy trained RL policy to MuJoCo simulation.

Usage:
    ros2 launch launch/policy_deploy.launch.py checkpoint:=/path/to/policy.pt
    ros2 launch launch/policy_deploy.launch.py checkpoint:=/path/to/policy.pt task:=go2
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_SCRIPT = os.path.join(REPO_ROOT, 'training', 'deploy', 'deploy_mujoco', 'deploy_mujoco.py')


def _launch_setup(context, *args, **kwargs):
    task = LaunchConfiguration('task').perform(context)
    checkpoint = LaunchConfiguration('checkpoint').perform(context)

    cmd = ['python3', DEPLOY_SCRIPT]
    if checkpoint:
        cmd += ['--checkpoint', checkpoint]
    if task:
        cmd += ['--task', task]

    policy_process = ExecuteProcess(
        cmd=cmd,
        output='screen',
        cwd=os.path.join(REPO_ROOT, 'training'),
    )
    return [policy_process]


def generate_launch_description():
    task_arg = DeclareLaunchArgument(
        'task',
        default_value='go2',
        description='Task name: go2, h1, g1',
    )

    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint',
        default_value='',
        description='Path to trained policy checkpoint (.pt file)',
    )

    return LaunchDescription([
        task_arg,
        checkpoint_arg,
        OpaqueFunction(function=_launch_setup),
    ])
