#!/usr/bin/env python3
"""Continuous Go2 stack health: the checks of check_stack_go2.py, published once a second.

Publishes
  /diagnostics   diagnostic_msgs/DiagnosticArray, one status per topic, TF link and the fall check
  /go2/health    std_msgs/String JSON: {"ok": bool, "failed": [names], "checks": {name: {"ok", "detail"}}}
so a dashboard (Foxglove, rqt_robot_monitor) or a mission node can react to a stalled map, a dead /points stream
or a fallen robot without anyone running the one-shot checker.

  python3 scripts/stack_health_go2.py --ros-args -p use_sim_time:=true -p skip_obstacles:=true
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_msgs.msg import String

from check_stack_go2 import DEFAULT_TOPICS, StackCheck


def summarize(rows):
    """rows: iterable of (ok, name, detail) -> health dict (pure, for the JSON topic)."""
    checks = {name: {"ok": bool(ok), "detail": detail} for ok, name, detail in rows}
    failed = sorted(n for n, c in checks.items() if not c["ok"])
    return {"ok": not failed, "failed": failed, "checks": checks}


class Health(StackCheck):
    def __init__(self):
        super().__init__(dict(DEFAULT_TOPICS))
        self.skip_obstacles = self.declare_parameter("skip_obstacles", False).value
        self.max_age = self.declare_parameter("max_age", 3.0).value
        self.base = self.declare_parameter("base_frame", "base").value
        self.odom = self.declare_parameter("odom_frame", "odom").value
        self.map = self.declare_parameter("map_frame", "map").value
        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.health_pub = self.create_publisher(String, "/go2/health", 10)
        self.create_timer(1.0, self.tick)

    def tick(self):
        _, topic_rows = self.topic_report(self.max_age)
        _, tf_rows = self.tf_report([(self.odom, self.base), (self.map, self.base)])
        topic_rows = [r for r in topic_rows if not (self.skip_obstacles and r[1] == "/obstacle_tracker/state")]
        rows = [(g, f"topic {n}", d) for g, n, d in topic_rows] + [(g, f"tf {n}", d) for g, n, d in tf_rows] \
            + list(self.fall_report())
        health = summarize(rows)
        msg = DiagnosticArray(); msg.header.stamp = self.get_clock().now().to_msg()
        for ok, name, detail in rows:
            st = DiagnosticStatus(name=f"go2/{name}", hardware_id="go2")
            st.level = DiagnosticStatus.OK if ok else DiagnosticStatus.ERROR
            st.message = detail
            st.values = [KeyValue(key="ok", value=str(bool(ok)))]
            msg.status.append(st)
        self.diag_pub.publish(msg)
        out = String(); out.data = json.dumps(health); self.health_pub.publish(out)


def main():
    rclpy.init(); n = Health()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally: n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
