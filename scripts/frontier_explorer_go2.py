#!/usr/bin/env python3
"""Lightweight frontier exploration driver for the Go2 in Gazebo.

There's no Nav2 costmap/planner wired up for the Go2 yet (no footprint or
controller tuning exists for a legged base), so this doesn't do Nav2-style
frontier exploration with a global planner. Instead it directly scores
frontier cells on the /map occupancy grid (wavefront from the robot's cell,
frontier = reachable free cell adjacent to unknown) and hands the best
cluster's centroid off to whichever locomotion backend is driving:

- control_mode:=cmd_vel (default, CHAMP backend): a simple point-and-go
  Twist controller published to /cmd_vel, which CHAMP's
  quadruped_controller (cmd_vel/smooth) turns into a walking gait.
- control_mode:=nmpc_goal (Quad-SDK backend): publish the goal as a
  geometry_msgs/PointStamped to Quad-SDK's live goal_state topic (see
  global_body_planner.cpp's goal_state_sub_) and let global_body_planner /
  local_planner / nmpc_controller do the actual walking -- no Twist
  involved, so linear_speed/angular_speed/BLOCKED_LOOKAHEAD don't apply.

Good enough to grow the RTAB-Map map autonomously; not a substitute for a
real planner if obstacles crowd the direct line to a goal (cmd_vel mode
only -- NMPC mode gets real obstacle-aware planning from Quad-SDK itself).

Ported a few safety habits from rosnav's builtin frontier_explorer.py
(goal_pullback/frontier_clearance_radius, explore_lite's progress_timeout):
frontier goals get pulled back off the free/unknown boundary into confirmed
clear space, and a goal that stops making progress for too long (usually
because it's wedged against an obstacle the wavefront saw as reachable but
the point-and-go controller can't actually thread) gets abandoned and
temporarily excluded so the next search doesn't just re-pick it. It also
optionally listens to scripts/obstacle_tracker_go2.py's /obstacle_tracker/state
topic to keep frontier goals and the lookahead check away from moving
obstacles -- rosnav has no equivalent since it has no dynamic-obstacle tracker.

Usage:
    python3 scripts/frontier_explorer_go2.py --ros-args -p use_sim_time:=true
    python3 scripts/frontier_explorer_go2.py --ros-args -p use_sim_time:=true \\
        -p control_mode:=nmpc_goal -p base_frame:=body \\
        -p goal_state_topic:=/robot_1/clicked_point
"""

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import rclpy
from frontier_geometry import find_frontier_goal, raycast_blocked, world_to_cell
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


FREE_MAX = 40          # occupancy value below which a cell counts as "free"
OCCUPIED_MIN = 65       # occupancy value at/above which a cell counts as "occupied"
MIN_FRONTIER_SIZE = 6    # ignore frontier clusters smaller than this many cells
GOAL_TOLERANCE = 0.45    # m, close enough to a frontier goal to replan
BLOCKED_LOOKAHEAD = 0.5  # m, don't drive forward if this close a cell is occupied
NO_FRONTIER_TIMEOUT = 30.0  # s with no frontier found before declaring done
OBSTACLE_STALE_TIMEOUT = 2.0  # s, ignore /obstacle_tracker/state once this stale
MAPS_DIR = Path(__file__).resolve().parents[1] / "maps" / "generated"


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer_go2')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('linear_speed', 0.35)
        self.declare_parameter('angular_speed', 0.6)
        self.declare_parameter('control_period', 0.1)
        self.declare_parameter('control_mode', 'cmd_vel')  # 'cmd_vel' or 'nmpc_goal'
        self.declare_parameter('goal_state_topic', '/robot_1/clicked_point')
        # Standoff pulled a chosen frontier goal back toward the robot, off
        # the free/unknown boundary the wavefront search necessarily leaves
        # it on (rosnav's goal_pullback/frontier_clearance_radius default to
        # 0.55m; kept a bit tighter here since the Go2's own footprint is
        # smaller than the diff-drive base rosnav was tuned against).
        self.declare_parameter('clearance_standoff', 0.3)
        # If a goal stops getting closer for this long, treat it as stuck
        # (wedged against something the wavefront saw as reachable but the
        # point-and-go controller can't actually thread) rather than idling
        # on it forever -- same intent as explore_lite's progress_timeout.
        self.declare_parameter('stuck_timeout', 12.0)
        self.declare_parameter('min_goal_progress', 0.15)
        self.declare_parameter('goal_exclude_radius', 0.6)
        self.declare_parameter('goal_exclude_ttl', 15.0)
        # Empty topic name disables obstacle-awareness entirely (no
        # obstacle_tracker_go2.py running, e.g. track_obstacles:=false).
        self.declare_parameter('obstacle_state_topic', '/obstacle_tracker/state')
        self.declare_parameter('obstacle_avoid_radius', 0.6)
        self.declare_parameter('obstacle_speed_min', 0.05)
        self.declare_parameter('save_map', True)
        self.declare_parameter('map_save_path', '')  # '' -> auto-named under maps/generated/

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.control_mode = self.get_parameter('control_mode').value
        self.clearance_standoff = self.get_parameter('clearance_standoff').value
        self.stuck_timeout = self.get_parameter('stuck_timeout').value
        self.min_goal_progress = self.get_parameter('min_goal_progress').value
        self.goal_exclude_radius = self.get_parameter('goal_exclude_radius').value
        self.goal_exclude_ttl = self.get_parameter('goal_exclude_ttl').value
        self.obstacle_avoid_radius = self.get_parameter('obstacle_avoid_radius').value
        self.obstacle_speed_min = self.get_parameter('obstacle_speed_min').value
        self.save_map = self.get_parameter('save_map').value
        self.map_save_path = self.get_parameter('map_save_path').value
        self.map_saved = False
        self.published_goal_xy = None  # last goal actually sent in nmpc_goal mode

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if self.control_mode == 'nmpc_goal':
            self.goal_state_pub = self.create_publisher(
                PointStamped, self.get_parameter('goal_state_topic').value, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, 1)

        self.obstacle_tracks = []
        self.obstacle_tracks_stamp = None
        obstacle_topic = self.get_parameter('obstacle_state_topic').value
        if obstacle_topic:
            self.create_subscription(String, obstacle_topic, self._on_obstacle_state, 5)

        self.latest_map = None
        self.goal_xy = None
        self.goal_progress_best = None
        self.goal_progress_time = self.get_clock().now()
        self.excluded_goals = []  # [(x, y, expire_time), ...]
        self.last_frontier_seen = self.get_clock().now()

        # champ_joint_trajectory_to_go2_gz.py (cmd_vel mode only) treats
        # /cmd_vel as stale after 0.35s (COMMAND_TIMEOUT_S) and falls back to
        # its idle-drift-reset behaviour, which teleports the robot back
        # upright -- so this needs to keep publishing well under that, not
        # just whenever it replans a goal. nmpc_goal mode has no such
        # deadline (Quad-SDK's global_body_planner just keeps tracking
        # whatever goal it last got), but the expensive frontier search only
        # actually runs inside _step when goal_xy is None either way, so a
        # fast tick here stays cheap in both modes.
        period = self.get_parameter('control_period').value
        self.create_timer(period, self._step)
        self.get_logger().info(
            f'frontier_explorer_go2 started (control_mode={self.control_mode}, '
            f'map_frame={self.map_frame}, base_frame={self.base_frame}, '
            f'obstacle_topic={obstacle_topic or "disabled"})')

    def _on_map(self, msg: OccupancyGrid):
        self.latest_map = msg

    def _on_obstacle_state(self, msg: String):
        try:
            tracks = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(tracks, list):
            self.obstacle_tracks = tracks
            self.obstacle_tracks_stamp = self.get_clock().now()

    def _active_obstacle_tracks(self):
        """Moving tracks from obstacle_tracker_go2.py, or [] if none/stale."""
        if self.obstacle_tracks_stamp is None:
            return []
        age = (self.get_clock().now() - self.obstacle_tracks_stamp).nanoseconds / 1e9
        if age > OBSTACLE_STALE_TIMEOUT:
            return []
        return [t for t in self.obstacle_tracks
                if t.get('speed', 0.0) >= self.obstacle_speed_min]

    def _prune_excluded_goals(self):
        now = self.get_clock().now()
        self.excluded_goals = [g for g in self.excluded_goals if g[2] > now]

    def _robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f'no {self.map_frame}->{self.base_frame} TF yet: {exc}',
                                    throttle_duration_sec=5.0)
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    def _find_frontier_goal(self, msg: OccupancyGrid, robot_x, robot_y):
        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        grid = np.array(msg.data, dtype=np.int16).reshape((h, w))

        rr, rc = world_to_cell(robot_x, robot_y, ox, oy, res)
        if not (0 <= rr < h and 0 <= rc < w):
            self.get_logger().warn('robot cell is outside the current map bounds')
            return None

        self._prune_excluded_goals()
        excluded = [(gx, gy) for gx, gy, _ in self.excluded_goals]
        tracks = self._active_obstacle_tracks()
        return find_frontier_goal(
            grid, w, h, res, ox, oy, robot_x, robot_y, excluded, tracks,
            free_max=FREE_MAX, occupied_min=OCCUPIED_MIN, min_frontier_size=MIN_FRONTIER_SIZE,
            clearance_standoff=self.clearance_standoff, goal_exclude_radius=self.goal_exclude_radius,
            obstacle_avoid_radius=self.obstacle_avoid_radius)

    def _path_blocked(self, msg: OccupancyGrid, robot_x, robot_y, yaw):
        res = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        w, h = msg.info.width, msg.info.height
        grid = np.array(msg.data, dtype=np.int16).reshape((h, w))
        tracks = self._active_obstacle_tracks()
        return raycast_blocked(grid, w, h, res, ox, oy, robot_x, robot_y, yaw,
                                BLOCKED_LOOKAHEAD, tracks, obstacle_margin=0.15,
                                occupied_min=OCCUPIED_MIN)

    def _save_map(self):
        """Fire-and-forget nav2_map_server map_saver_cli against the live /map.

        Runs detached (not waited on) so a slow disk/rtabmap hiccup can't
        stall this node's control-loop timer -- exploration has already
        finished by the time this fires, so there's nothing left to block.
        """
        if self.map_save_path:
            out_path = Path(self.map_save_path)
        else:
            MAPS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = MAPS_DIR / f"go2_room_{time.strftime('%Y%m%d_%H%M%S')}"
        self.get_logger().info(f'saving map to {out_path}.pgm/.yaml')
        try:
            subprocess.Popen([
                'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                '-f', str(out_path), '--ros-args', '-p', 'use_sim_time:=true',
                # RTAB-Map republishes /map continuously with volatile QoS
                # (no transient_local), unlike slam_toolbox/gmapping which
                # latch it -- map_saver_cli defaults to expecting
                # transient_local and otherwise never receives a single
                # message (confirmed: it exits with "Failed to spin map
                # subscription" and writes nothing).
                '-p', 'map_subscribe_transient_local:=false',
            ])
        except OSError as exc:
            self.get_logger().error(f'map_saver_cli failed to start: {exc}')

    def _step(self):
        msg = self.latest_map
        if msg is None:
            return
        pose = self._robot_pose()
        if pose is None:
            return
        rx, ry, yaw = pose
        now = self.get_clock().now()

        if self.goal_xy is not None:
            dist_to_goal = math.hypot(self.goal_xy[0] - rx, self.goal_xy[1] - ry)
            if dist_to_goal < GOAL_TOLERANCE:
                self.goal_xy = None
            elif (self.goal_progress_best is None
                  or dist_to_goal < self.goal_progress_best - self.min_goal_progress):
                self.goal_progress_best = dist_to_goal
                self.goal_progress_time = now
            elif (now - self.goal_progress_time).nanoseconds / 1e9 > self.stuck_timeout:
                self.get_logger().warn(
                    f'no progress toward ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}) for '
                    f'{self.stuck_timeout:.0f}s -- abandoning it')
                self.excluded_goals.append((
                    self.goal_xy[0], self.goal_xy[1],
                    now + Duration(seconds=self.goal_exclude_ttl)))
                self.goal_xy = None

        if self.goal_xy is None:
            self.goal_xy = self._find_frontier_goal(msg, rx, ry)
            if self.goal_xy is not None:
                self.goal_progress_best = None
                self.goal_progress_time = now

        if self.goal_xy is None:
            elapsed = (now - self.last_frontier_seen).nanoseconds / 1e9
            if elapsed > NO_FRONTIER_TIMEOUT:
                self.get_logger().info(
                    f'no frontiers left for {elapsed:.0f}s -- exploration complete, stopping.',
                    throttle_duration_sec=NO_FRONTIER_TIMEOUT)
                if self.save_map and not self.map_saved:
                    self.map_saved = True  # fire once, even if a stray frontier reappears later
                    self._save_map()
            if self.control_mode == 'cmd_vel':
                self.cmd_pub.publish(Twist())
            return

        self.last_frontier_seen = now

        if self.control_mode == 'nmpc_goal':
            # Quad-SDK's global_body_planner already tracks whatever goal it
            # last received (see global_body_planner.cpp's goal_state_sub_)
            # and dedups by header.stamp, so only publish when the frontier
            # goal actually changes -- no need to re-send every tick.
            if self.goal_xy != self.published_goal_xy:
                gx, gy = self.goal_xy
                point = PointStamped()
                point.header.stamp = self.get_clock().now().to_msg()
                point.header.frame_id = self.map_frame
                point.point.x, point.point.y = gx, gy
                self.goal_state_pub.publish(point)
                self.published_goal_xy = self.goal_xy
            return

        gx, gy = self.goal_xy
        angle_to_goal = math.atan2(gy - ry, gx - rx)
        angle_err = math.atan2(math.sin(angle_to_goal - yaw), math.cos(angle_to_goal - yaw))

        cmd = Twist()
        if abs(angle_err) > 0.35:
            cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_err))
        elif self._path_blocked(msg, rx, ry, yaw):
            # Something's in the way of a straight line to the goal -- this
            # explorer has no local planner, so just turn and re-score next
            # tick rather than drive into it.
            cmd.angular.z = self.angular_speed
        else:
            cmd.linear.x = self.linear_speed
            cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, 0.5 * angle_err))
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.control_mode == 'cmd_vel':
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
