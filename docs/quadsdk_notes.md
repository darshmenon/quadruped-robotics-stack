# Quad-SDK (NMPC) — Porting & Debug Notes

Deep-dive history for the Quad-SDK NMPC backend. The README only keeps the
short version (how to run it, what terrain works); this file has the full
story for anyone maintaining or debugging it further.

## Background

[Quad-SDK](https://github.com/robomechanics/quad-sdk) is vendored in
`ros2/quad_sdk/`. Its solver dependencies, RBDL and IPOPT, are built locally
into `ros2/quad_sdk_external/`, so the workspace does not need anything under
`/usr/local`. The vendored tree includes Go2 URDF/SDF assets, terrain worlds,
NMPC solver code, planners, `robot_driver`, and the Quad-SDK controller
plugins. `robot_driver` pulls in two prebuilt hardware SDK binary blobs under
`ros2/quad_sdk/external/` (`unitree_sdk2`, `mblink`) that it links
unconditionally even in sim mode — no build step for those, just vendored
`.a`/`.so` files.

## Status

The Humble/Jammy port builds and launches Go2 in Gazebo Harmonic. The old
blocker from `ros-humble-gz-ros2-control` is bypassed for Go2 by a native
gz-sim8 system plugin, `quad_sim_effort_controller`, which subscribes to
`/robot_1/control/joint_command` and applies Quad-SDK joint efforts directly
in Gazebo.

Go2 spawns and holds a low **sit** pose (`sit_angles` in
`controller_plugin.cpp`'s bootstrap PD hold, body height ~0.08m). Publishing
`ros2 topic pub /robot_1/control/mode std_msgs/msg/UInt8 "data: 1"` is the
documented "stand" trigger (`robot_driver.cpp`'s `SIT → SIT_TO_READY → READY`
state machine, `READY = 1`, 1s interpolation) — a **single** `--once` publish
can lose the message to a ROS2 discovery race (publisher exits before the
subscription match completes); **publish for ~1-2s at a few Hz instead**
(`--rate 5`, a couple seconds) and it reliably reaches standing height
(verified: body z went from 0.08m → 0.286m).

Once standing, `global_body_planner` computes and publishes a real plan (e.g.
`Solve time: 0.036s, Vertices generated: 6, Path length: 5.06m` for a
`goal_state:=[3.0, 0.0]` override) and `local_planner`/`nmpc_controller` pick
it up and start solving.

**IPOPT was failing every single call with `ApplicationReturnStatus = -12`
(`Invalid_Option`)** — not a convergence problem, but a config bug:
`nmpc_controller.cpp` hardcoded `linear_solver = "ma27"`, an HSL solver
requiring a licensed download this repo's IPOPT build doesn't have (it only
has the open-source `mumps` solver). Diagnosed by adding the actual
status-code log (`RCLCPP_WARN_STREAM(..., "ApplicationReturnStatus = " <<
status)`) around the solve call. Fixed by switching to `"mumps"` and dropping
the now-irrelevant `ma57_pre_alloc` option.

**Result: Go2 walks.** With the fix, NMPC converges and the robot walks to a
goal. The measured flat-ground benchmark (10/10 reached, load under 8) is a
median planner cycle of 19.7 ms, p95 67.4 ms, and 0.624-0.629 m/s over 5 m.
An earlier single observation of 0.69 m/s is not the benchmark number. See
[docs/BENCHMARKS.md](BENCHMARKS.md).

| Milestone | Status |
|-----------|--------|
| Build all Quad-SDK packages | Done |
| Launch Gazebo and spawn Go2 | Done |
| Start `robot_driver` in sim | Done |
| Drive Go2 joints in Harmonic without `gz_ros2_control` | Done |
| Start NMPC planner launch without crashing | Done |
| Robot reaches a valid standing pose | Done (needs a held-down `control/mode` publish, not `--once`) |
| `global_body_planner` computes and publishes a real path | Done |
| NMPC (`local_planner`/`nmpc_controller`) converges reliably (flat ground) | Done |
| Walk with NMPC (flat ground) | Done — 5 m goal at 0.624-0.629 m/s, 10/10, median cycle 19.7 ms |
| Robust across all terrain worlds | Not yet — see [Terrain test results](#terrain-test-results) |

## Porting fixes applied

- ROS package/dependency names moved from Jazzy/Noble assumptions to Humble/Jammy where packages exist.
- RBDL/IPOPT lookup auto-detects the local prefix under `ros2/quad_sdk_external/local`.
- Pinocchio compile definitions linked through packages that include `quad_utils/quad_kd2.hpp`.
- `ros2_control` API calls adjusted for the installed Humble controller-manager version.
- Timestamp arithmetic updated to wrap message stamps in `rclcpp::Time`.
- Humble-missing grid-map UI/filter components are not installed by the setup script and are optional at launch: `enable_grid_map_viz:=false` and `enable_grid_map_filters:=false` by default.
- Gazebo sim defaults to `estimator:=none`; the sim path already consumes ground-truth `RobotState`, so the mocap-oriented complementary filter isn't required for launch.
- Go2 Gazebo Harmonic control uses `quad_sim_effort_controller` instead of `gz_ros2_control`; Humble's packaged `gz_ros2_control` plugin exports the older Ignition/Fortress hook and can't load in `gz-sim8`.
- `body_force_estimator` is treated as optional in `planning.py`; this repo doesn't vendor that package.
- When Humble-missing `grid_map_filters` are disabled, `mapping.py` relays raw mesh terrain to the filtered terrain topic names expected by `robot_bringup` and `local_planner`.
- Local terrain consumers tolerate raw mesh maps that only contain the `z` layer instead of filtered `z_inpainted`/`z_smooth`/traversability layers: `global_body_planner`'s `isTraversable`/`getTraversability` fall back to fully-traversable when the `traversability` layer doesn't exist, and `nmpc_controller`'s `quad_nlp.cpp` (4 call sites) goes through `terrainHeightAtPosition`/`terrainNormalAtPosition` helpers instead of hardcoding `z_inpainted`/`normal_vectors_*` — both crashed with `GridMap::at(...): No map layer ... available` before this.
- `quad_utils` builds as a **static** library (`libquad_utils.a`); a source-only change there needs a full `colcon build` (not `--packages-select quad_utils`) to relink the ~6 downstream packages that statically link it, or they keep running stale code.
- `robot_driver` was vendored (it's not hardware-only despite the name — it's also the sim-mode state estimator/control interface); it needed the same `pinocchio::pinocchio` direct-link fix as the other 5 packages.
- `quad_plan.py` never declared a top-level `goal_state` launch argument — it only read `goal_state` from *inside* each robot's `robot_configs` JSON entry. Passing `goal_state:="[x, y]"` directly to `ros2 launch quad_utils quad_plan.py` was silently ignored, so the robot always walked to the `global_body_planner.yaml` default of `[5.0, 0.0]`. Fixed by adding a top-level `goal_state` argument.

## Terrain test results

`step_20cm.sdf`, `flat.sdf`, and `big_flat.sdf` are the reliably-working
surfaces. Everything else has now actually been run (headless, `goal ≈ (1,
0)`, `timeout 60`) rather than assumed — results below, tested on 2026-07-10.

| World | Result |
|-------|--------|
| `flat.sdf` / `big_flat.sdf` | Works |
| `step_20cm.sdf` | Works (previously verified end-to-end with a walk to a distant goal) |
| `step_25cm.sdf` | 2026-07-10: NMPC solved with no crash; motor effort repeatedly exceeded the then configured 33.5 Nm limit (now 23.7 Nm for hip and thigh, not re-run), up to about 2x (67 warnings in 45 s). A 2026-11-26 run was flagged as a fall, see [BENCHMARKS.md](BENCHMARKS.md). Hardware behaviour untested. |
| `step_30cm.sdf` | Same pattern as `step_25cm.sdf` (46 torque-overload warnings), still no crash. |
| `gap_80cm.sdf` | `global_body_planner_node` segfaults (exit code -11) |
| `slope_20_hole.sdf` | `global_body_planner_node` segfaults |
| `rough_40cm_huge.sdf` | `global_body_planner_node` segfaults |
| `gap_40cm_local_min.sdf` | `global_body_planner_node` segfaults |
| `step_10cm_local_min.sdf` | `global_body_planner_node` segfaults; robot also seen falling through the mesh (ground-truth z ≈ -54m) |
| `step_15cm_local_min.sdf` | Same as above (z ≈ -47m) |
| `parkour_local_min.sdf` | `global_body_planner_node` segfaults (108 torque warnings logged before the crash) |
| `gap_20cm.sdf`, `gap_40cm.sdf`, `slope_20.sdf`, `rough_25cm.sdf`, `step_10cm.sdf` | Not re-tested this pass — treat as unverified, not confirmed working |

Takeaways:
- The crash isn't just the `_local_min`/`_hole` name pattern — `gap_80cm.sdf`
  and `rough_40cm_huge.sdf` hit the same `global_body_planner` segfault, so
  the actual trigger looks like "terrain with a large local discontinuity or
  no easy flat path," not a specific SDF file naming convention.
- This was run back-to-back across ~10 sim launches on a memory-constrained
  box (free RAM dropped to ~400MB, 8GB+ swapped) with software/EGL
  rendering — that's a plausible confound. But the crash pattern tracked
  terrain difficulty, not run order (e.g. `step_30cm.sdf`, run 6th, passed
  clean; `gap_40cm_local_min.sdf`, run 3rd, crashed) — so it's more likely a
  real bug in `global_body_planner`'s path search under distant/awkward
  goals than pure resource exhaustion. Worth re-confirming on a beefier
  machine before trusting it fully.
- Next debugging step: attach a debugger or add crash logging around
  `global_body_planner_node`'s path search to find what specifically
  segfaults — likely a null/out-of-range access when no traversable path
  exists near the goal.

### Re-check 2026-11-22 (global_body_planner segfault)

Update 2026-11-27: the `gap_80cm` crash was traced to unbounded recursion in `RRT::attemptConnect` and fixed in commit `a4960cb` (see [BENCHMARKS.md](BENCHMARKS.md)); the hypothesis below is kept as history.

`gap_80cm.sdf` reproduced the `global_body_planner_node` segfault (exit code -11) at a quiet load average (4.35), and `parkour_local_min.sdf` ended with the planner dying as well (exit code -2, shut down with the launch). So the crash is not only the memory pressure of the earlier back-to-back runs. The log shows it dies right after `valid and shorter or previous plan not valid`, i.e. just after the first plan is accepted in `GlobalBodyPlanner::callPlanner` (`global_body_planner.cpp`). A gdb run (`launch-prefix:="gdb -batch -ex run -ex bt -ex quit --args"` on the planner) ran at load 25, finished without a SIGSEGV and gave no backtrace.

Hypothesis, not confirmed: `GBPL::postProcessPath` (`gbpl.cpp`) reads `back()` or calls `pop_back()` on an empty vector. A path has N states and N-1 actions, and on a gap or local minimum every straight shortcut could fail, so the shortcut walk may pop both vectors empty. Bounds guards were written for this in `gbpl.cpp`, `rrt.cpp`, `planning_utils.cpp`, `global_body_planner.cpp` and `global_body_plan.cpp` (uncommitted). **Negative result, 2026-11-23:** with those guards built (installed binary newer than all five sources), `gap_80cm.sdf` still segfaulted, `global_body_planner_node` exit code -11 (`docs/data/gap80_check.csv`, `docs/overnight_logs/gap80_planner_lines.log`). So the guards do not fix this crash and the real fault site is still unknown. A backtrace has not been obtained (the earlier gdb run did not crash, and no core or apport report exists for the planner).

The 25 cm and 30 cm step runs still command feedforward torque (joint not confirmed; the kept flat-world logs peak at 64.5 Nm on the hip) at about twice the then configured 33.5 Nm limit. `robot_driver` already clamps the total effort it sends and logs the warning. Clamping the feedforward earlier would change the NMPC command, so that warning stays.
