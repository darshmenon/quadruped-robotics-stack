# Project Growth Plan

This repo already has the hard base: Go2 meshes, Gazebo Harmonic launch files, MuJoCo training, Quad-SDK NMPC, CHAMP walking, RTAB-Map SLAM, frontier exploration, and early arm integration. To make it feel like a larger robotics project like `rosnav` or `UR3_ROS2_PICK_AND_PLACE`, the next work should turn isolated demos into repeatable systems with launchable pipelines, diagnostics, tests, assets, and benchmark results.

## Current Active Work

- Finish and verify the 3D SLAM obstacle-tracking path: `scripts/obstacle_tracker_go2.py`, `launch/slam3d_go2.launch.py`, and `training/envs/go2_gz_world_room.sdf`.
- Validate the Gazebo RL isolation changes: `ROS_DOMAIN_ID` and `GZ_PARTITION` now isolate training sessions from SLAM/Nav2 sessions.
- `ros2/gz_ros2_control/` stays vendored in-repo (decided).
- `ros2/log/latest` and `ros2/log/latest_build` were tracked generated symlinks; untracked them (`ros2/log/` was already gitignored, they predated the rule).

## Best Next Features

1. **Autonomous indoor demo**
   - One command launches the Go2 in the room world with RTAB-Map, frontier exploration, obstacle tracking, and RViz.
   - Add map autosave when exploration completes.
   - Add a short demo GIF under `docs/images/`.

2. **Dynamic obstacle simulation**
   - Add a simple moving cylinder/person model in the room world.
   - Publish obstacle tracker output to `/obstacle_tracker/state` and RViz markers.
   - Add a Nav2 slowdown or stop behavior when tracked obstacle speed crosses a threshold.

3. **Gazebo RL benchmark**
   - Train a fresh Gazebo policy after the reward and isolation fixes.
   - Add `training/eval_gazebo.py` to report distance, fall rate, episode length, mean speed, and command tracking error.
   - Save results as a small CSV/Markdown table in `docs/`.

4. **Manipulator pipeline**
   - Make the arm do one real task: reach, push, pick, or press a target.
   - Add a launch file that starts Go2 plus arm controllers plus a scripted task node.
   - Add a perception-free baseline first, then add camera/depth perception later.

5. **Nav2 production layer**
   - Add a Go2-specific Nav2 params file with footprint, velocity limits, recovery behavior, and costmap settings verified in the room world.
   - Add saved room map assets and a static-map navigation launch.
   - Add a CLI waypoint runner for patrol routes.

6. **Perception and semantic mapping**
   - Add YOLO or color/shape detection for objects in the room/world.
   - Publish semantic objects as world-frame markers and JSON state.
   - Feed semantic obstacles into Nav2 as keepout or slowdown zones.

7. **System diagnostics**
   - Add a health node that reports SLAM status, odom age, TF availability, point cloud rate, controller state, and fall detection.
   - Publish both JSON state and ROS diagnostics.
   - Add a single `scripts/check_stack_go2.py` command for demos.

8. **Session recording**
   - Add a script wrapping `ros2 bag record` for `/tf`, `/odom`, `/points`, `/map`, `/cmd_vel`, `/obstacle_tracker/state`, and controller topics.
   - Add a replay recipe in docs so demos can be debugged without rerunning Gazebo.

9. **Benchmark worlds**
   - Keep the room world for indoor SLAM.
   - Add obstacle course, narrow doorway, slope, rough terrain, and moving obstacle variants.
   - Track success/failure per locomotion backend: CHAMP, Quad-SDK NMPC, MuJoCo RL, Gazebo RL.

10. **Dashboard or Foxglove bridge**
    - Add optional `foxglove_bridge` or `rosbridge_suite` launch arg.
    - Stream map, odom, obstacle tracks, policy status, and mission state.
    - This gives the project a stronger external demo than RViz alone.

## Structural Upgrades

- Split high-level features into ROS packages over time: `go2_navigation`, `go2_perception`, `go2_manipulation`, `go2_rl`, and `go2_system_tests`.
- Add `docs/INSTALL.md`, `docs/DEMOS.md`, `docs/BENCHMARKS.md`, and `docs/TROUBLESHOOTING.md`.
- Move screenshots and GIFs into `docs/images/` and reference them from README sections.
- Add launch wrappers for flagship demos so users do not need three terminals for common flows.
- Add smoke tests for launch files and pure-Python tests for planners, trackers, reward functions, and IK.
- Add a lightweight `Makefile` or `justfile` for build, test, train, play, slam, nav, and record commands.
- Add `.gitignore` rules for ROS build outputs, logs, bags, checkpoints, and generated maps that should not be committed.

## Suggested Order

1. Finish and verify 3D SLAM + obstacle tracking in the room world.
2. Add a single-command autonomous indoor demo and record a GIF.
3. Add diagnostics and a stack checker.
4. Train and benchmark Gazebo RL after the reward fixes.
5. Build one manipulator task that works end to end.
6. Add semantic perception and dynamic obstacle avoidance.
7. Add dashboard/Foxglove streaming and session replay.
