#!/usr/bin/env python3
"""Slow or stop the Go2 when a tracked moving obstacle is close and near its path.

Sits between a velocity source (Nav2, the frontier explorer, teleop) and the walking controller:
  /cmd_vel_in  (Twist)  ->  scale by the nearest threatening obstacle  ->  /cmd_vel
Obstacles come from scripts/obstacle_tracker_go2.py (/obstacle_tracker/state, JSON in the map frame); the robot
pose comes from /odom (map ~ odom in the sim stack).

An obstacle only counts when it is in front of the robot (within --half-angle of the commanded direction of
travel) and either moving faster than --min-speed or closer than --stop-dist. Inside --stop-dist the output is
zero (the robot also never reverses into it: turning in place is still allowed). Between --stop-dist and
--slow-dist the speed ramps linearly from 0 to 1, and an obstacle closing on the robot shrinks the scale
further through its time to collision. /obstacle_speed_gate/state (String JSON) reports the scale and the
obstacle that set it, for the dashboard and the stack checker.

  python3 scripts/obstacle_speed_gate_go2.py --ros-args -p use_sim_time:=true -p slow_dist:=2.0 -p stop_dist:=0.7
"""
import json
import math


def gate_scale(robot, cmd, obstacles, slow_dist=2.0, stop_dist=0.7, half_angle=math.radians(60.0),
               min_speed=0.05, ttc_slow=3.0, ttc_stop=1.0):
    """Return (scale in [0, 1], id of the limiting obstacle or None).

    robot: (x, y, yaw) in the map frame. cmd: (vx, vy) commanded in the body frame.
    obstacles: dicts with x, y, vx, vy, optional width/length (map frame).
    """
    rx, ry, yaw = robot
    c, s = math.cos(yaw), math.sin(yaw)
    # commanded direction in the map frame; a pure rotation has no direction, so nothing is gated
    cx, cy = c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1]
    cspeed = math.hypot(cx, cy)
    if cspeed < 1e-3:
        return 1.0, None
    ux, uy = cx / cspeed, cy / cspeed
    best, who = 1.0, None
    for o in obstacles:
        dx, dy = o["x"] - rx, o["y"] - ry
        radius = 0.5 * max(o.get("width", 0.0), o.get("length", 0.0))
        d = max(math.hypot(dx, dy) - radius, 0.0)          # gap to the obstacle's edge
        if d > slow_dist:
            continue
        ang = abs(math.atan2(ux * dy - uy * dx, ux * dx + uy * dy))
        if ang > half_angle and d > stop_dist:
            continue
        speed = math.hypot(o["vx"], o["vy"])
        if speed < min_speed and d > stop_dist:
            continue                                         # a parked obstacle is Nav2's job, not the gate's
        scale = 0.0 if d <= stop_dist else (d - stop_dist) / (slow_dist - stop_dist)
        # closing speed along the line of sight (robot speed toward it + obstacle speed toward the robot)
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            closing = cspeed * (ux * dx + uy * dy) / dist - (o["vx"] * dx + o["vy"] * dy) / dist
            if closing > 1e-3:
                ttc = d / closing
                if ttc <= ttc_stop:
                    scale = 0.0
                elif ttc < ttc_slow:
                    scale = min(scale, (ttc - ttc_stop) / (ttc_slow - ttc_stop))
        if scale < best:
            best, who = scale, o.get("id")
    return max(0.0, min(1.0, best)), who


def main():
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from std_msgs.msg import String

    class Gate(Node):
        def __init__(s):
            super().__init__("obstacle_speed_gate_go2")
            p = lambda n, d: s.declare_parameter(n, d).value
            s.kw = dict(slow_dist=float(p("slow_dist", 2.0)), stop_dist=float(p("stop_dist", 0.7)),
                        half_angle=math.radians(float(p("half_angle_deg", 60.0))), min_speed=float(p("min_speed", 0.05)))
            s.max_age = float(p("max_state_age", 1.0))
            s.robot = None; s.obstacles = []; s.obs_t = None
            s.pub = s.create_publisher(Twist, p("output_topic", "/cmd_vel"), 10)
            s.state_pub = s.create_publisher(String, "/obstacle_speed_gate/state", 10)
            s.create_subscription(Twist, p("input_topic", "/cmd_vel_in"), s.on_cmd, 10)
            s.create_subscription(String, "/obstacle_tracker/state", s.on_state, 10)
            s.create_subscription(Odometry, "/odom", s.on_odom, 20)
            s.get_logger().info(f"gate up: /cmd_vel_in -> /cmd_vel, {s.kw}")

        def on_odom(s, m):
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            s.robot = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)

        def on_state(s, m):
            try: s.obstacles = json.loads(m.data).get("moving_obstacles", [])
            except ValueError: return
            s.obs_t = s.get_clock().now()

        def on_cmd(s, m):
            scale, who = 1.0, None
            fresh = s.obs_t is not None and (s.get_clock().now() - s.obs_t).nanoseconds * 1e-9 < s.max_age
            if s.robot is not None and fresh and s.obstacles:
                scale, who = gate_scale(s.robot, (m.linear.x, m.linear.y), s.obstacles, **s.kw)
            out = Twist(); out.linear.x = m.linear.x * scale; out.linear.y = m.linear.y * scale
            out.angular.z = m.angular.z                      # turning in place stays allowed
            s.pub.publish(out)
            st = String(); st.data = json.dumps({"scale": round(scale, 2), "obstacle": who}); s.state_pub.publish(st)

    rclpy.init(); n = Gate()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally: n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
