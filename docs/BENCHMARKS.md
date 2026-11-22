# Benchmarks

Use this file to track repeatable results instead of relying on one-off RViz observations.

## Gazebo RL

Command:

```bash
python3 training/eval_gazebo.py \
  --model training/logs/gazebo/go2_gazebo_final.zip \
  --episodes 5 \
  --csv training/eval/gazebo_eval.csv
```

Metrics to report:

| Date | Model | Command | Episodes | Mean steps | Fall rate | Mean speed | Speed error | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| TBD | TBD | `0.5 0.0 0.0` | TBD | TBD | TBD | TBD | TBD | Train a fresh checkpoint after the Gazebo reward fixes. |

## LiDAR-inertial odometry (RKO-LIO) on the Go2

RKO-LIO runs as a sidecar on the 16-beam `/points` and `/imu/data` (`launch/lio_go2.launch.py`); it never owns `odom->base`. A walk is recorded to a bag, then the bag is **replayed offline** (`ros2 bag play`, 0.5x, no Gazebo) through RKO-LIO and the estimate is scored against the simulator ground truth `/odom` with [evo](https://github.com/MichaelGrupp/evo) (SE3-aligned APE, RPE over 1 m). Offline replay is the reference: the same weave bag scored APE 0.285 m on the estimate recorded live while the machine was loaded.

```bash
python3 scripts/lio_walk_test.py --goal 5 --bag training/logs/bags/go2_arena_walk1                       # record: straight 5 m, arena world
python3 scripts/lio_walk_test.py --weave --weave-wz 0.2 --vx 0.35 --duration 70 \
  --world training/envs/go2_gz_world_outdoor.sdf --bag training/logs/bags/go2_outdoor_weave2             # record: S-curves, outdoor world
python3 scripts/lio_replay_eval.py training/logs/bags/go2_outdoor_weave2 [--extrinsics on]              # replay + score
```

RKO-LIO's own offline drain (`mode:=offline`, same identity extrinsics, deskew off) was scored on 2026-11-20, three repeats per bag. The repeats are byte-identical and had 0 resets, so the weave spread below is the 0.5x replayer, not the estimator. Offline APE: straight 0.054 m (RPE 0.051 m), weave 0.182 m (RPE 0.215 m). That weave number is the reference for later methods. The bag-play table stays as the record of what `ros2 bag play` produced.

Six replays per setting (two batches of three), mean (min to max), 0 odometry resets in all 24:

| Bag | Extrinsics | APE RMSE (m) | RPE RMSE, 1 m (m) |
|---|---|---|---|
| Straight walk (6.5 m, arena) | identity (default) | 0.054 (0.052 to 0.055) | 0.056 (0.033 to 0.089) |
| Straight walk (6.5 m, arena) | real lidar offset (-0.05, 0, 0.15) | 0.050 (0.013 to 0.194) | 0.050 (0.023 to 0.127) |
| Weaving walk (19.4 m, outdoor) | identity (default) | 0.299 (0.200 to 0.455) | 0.281 (0.185 to 0.382) |
| Weaving walk (19.4 m, outdoor) | real lidar offset (-0.05, 0, 0.15) | 0.299 (0.227 to 0.367) | 0.262 (0.208 to 0.353) |

Notes:
- The real lidar mount offset does not help: it is no better on average and its straight-walk spread is much larger (0.013 to 0.194 m), so identity extrinsics are the default (`config/lio_go2_identity.yaml`; the real offset is `config/lio_go2_extrinsics.yaml`). rko_lio needs extrinsics passed explicitly, otherwise it looks them up in TF, which a bag replay lacks.
- Replay-to-replay spread is large on the weave (APE 0.20 to 0.46 m for identical settings), so differences of a few centimetres are not meaningful. Single replays earlier scored 0.070 m and 0.209 to 0.275 m on the same bag.
- LIO-SAM (`--method lio_sam`, `launch/lio_sam_go2.launch.py`) is not usable on this 16-beam sim cloud: on the weave bag APE 4.05 m (RPE 0.82 m), on the straight walk it recorded only 48 to 91 poses per attempt (APE 169 m, a failure rather than a number). RKO-LIO is the odometry used.
- On the 0.5x replays, turning while walking costs about 4 to 5 times the straight-walk APE. The offline drain (above) cuts that to about 3.4 times (0.182 m vs 0.054 m).
- One bag per walk type, one walker (`go2_walk_lat2`), 0.2 to 0.35 m/s. The walker itself is erratic (median 0.17 m/s while moving, roll up to 17 deg, the heading wanders), and an identical weave later covered only 1.7 m.
- The estimated IMU lag against `/odom` is about +0.045 to +0.06 s (signed gyro-z correlation, 0.7). Its cause is not established: shifting the IMU stamps by +0.06 s made one replay clearly worse (APE 0.63 m), -0.06 s one replay 0.22 m, within the unshifted spread.
- LIO-SAM (`launch/lio_sam_go2.launch.py`, `config/lio_sam_go2.yaml`, ring/time adapter) does not track on these bags: APE 139.7 m on the straight walk (it only produced 71 poses) and 7.4 m on the weave, one replay each, so it is experimental here. The sim IMU has no noise model, so its IMU noise values are guesses.
- Leg odometry from `/joint_states` (`scripts/leg_odom_go2.py`) is a weak source in this sim: velocity error 0.66 m/s against a true speed of ~0.16 m/s, 1.7 m APE on the 19.4 m weave, and fusing it with RKO-LIO in an EKF gave no gain (APE within 1 mm of RKO-LIO alone). A single velocity scale fitted on one bag does not transfer: the best scale is about 1.0 on the straight walk (APE 0.71 m) and about 1.3 on the weave (0.96 m); the weave scale gives 1.04 m on the straight walk and the straight-walk scale gives 1.73 m on the weave. Even the best fit is far worse than RKO-LIO.
- `online_node` always segfaults when it is shut down (after printing its timing stats); the estimate recorded before that is complete.
- `deskew:=false` (the sim cloud has no per-point time), voxel 0.25 m, max range 30 m. `rko_lio`'s launch file reads only `param:=value` from the command line, so it is started as a subprocess. Turning deskew on, after writing a per-point `time` with `scripts/lidar_ring_time_adapter.py`, made both bags worse on the offline drain (straight APE 0.076 m, weave 0.580 m). The scan is captured at one instant, so the invented sweep times warp it. Deskew stays off.

## KISS-ICP and LIORF on the same two bags (2026-11-22)

A new method only replaces RKO-LIO if it beats the offline reference: straight APE 0.054 m and weave 0.182 m, same bags, same evo SE3 APE.

KISS-ICP is lidar only (no IMU, no extrinsics). `scripts/kiss_icp_eval.py` reads `/points` out of the bag, so there is no `ros2 bag play` spread. Config is `config/kiss_icp_go2.yaml`: deskew off, max range 30 m, min range 0.4 m, voxel size left at the library default (max range / 100 = 0.3 m). Three repeats per bag, trajectories byte-identical.

| Bag | Method | APE RMSE (m) | RPE RMSE, 1 m (m) | Poses |
|---|---|---:|---:|---:|
| Straight walk | KISS-ICP | 0.205 | 0.392 | 975 |
| Weaving walk | KISS-ICP | 1.802 | 0.886 | 777 |
| Offline RKO-LIO (reference) | straight / weave | 0.054 / 0.182 | 0.051 / 0.215 | 975 / 777 |

KISS-ICP loses on both bags. On this 16-beam cloud the IMU is earning its keep.

LIORF (LIO-SAM with feature extraction removed) was the follow-up, because KISS-ICP lost. One `ros2 bag play` at 0.5x per bag (`scripts/lio_replay_eval.py --method liorf`, `launch/liorf_go2.launch.py`, identity extrinsics, loop closure off, ring/time adapter on). It diverges: straight APE 445 m (220 poses), weave APE 368 m (372 poses). That is the same class of failure as stock LIO-SAM, not a result that sits inside the bag-play spread. RKO-LIO stays the odometry.

FAST-LIO2 and Point-LIO were not scored. Both target a Mid-360 or a Unitree L1, and the ROS 2 FAST-LIO2 tree here requires `livox_ros_driver2`, which is not installed.

## NMPC vs RL walker in Gazebo

Task: walk 5 m straight ahead on flat ground (goal counted within 0.25 m), 3 trials per controller, each in a fresh Gazebo. Measured from simulator ground truth.

```bash
scripts/benchmark_gazebo.sh --controller nmpc --trials 3 --goal 5.0
scripts/benchmark_gazebo.sh --controller rl --trials 3 --goal 5.0 --rl-vx 0.3 --rl-steer 1.5 --rl-wz-max 0.4
```

| Date | Controller | Reached goal | Falls | Time to goal (s) | Mean speed (m/s) | Max lateral drift (m) | Final heading change (deg) | Path / straight |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-11-04 | Quad-SDK NMPC | 3/3 | 0/3 | 7.58 | 0.63 | 0.02 | -0.2 | 1.01 |
| 2026-11-05 | RL walker (`go2_walk_lat2`), vx 0.3, heading gain 1.5 | 3/3 | 0/3 | 15.4 | 0.35 | 0.83 | 17.6 | 1.25 |
| 2026-11-22 | Quad-SDK NMPC, 10 trials, machine under heavy load (see note) | 6/10 | 0/10 | 7.55 | 0.63 | 0.01 | 1.11 | 1.03 |
| 2026-11-22 | Quad-SDK NMPC, 10 trials, one per run, load 4.8-7.7 | 10/10 | 0/10 | 7.58 | 0.63 | 0.01 | n/a | 1.00 |

2026-11-22 NMPC re-run, `scripts/benchmark_gazebo.sh --controller nmpc --trials 10 --goal 5.0` (rows appended to [data/gazebo_benchmark.csv](data/gazebo_benchmark.csv)): 10 trials, 6 reached, 0 falls. The 6 that reached walked at 0.626-0.636 m/s (7.47-7.59 s). Trials 3, 4, 5 and 7 stalled at the start (1.47, 0.12, 0.04, 0.01 m covered) without falling. The machine load average was 42-49 during trials 3-5 (other sessions' simulations) and about 12 by the end. Three of the stalled runs logged 3300-3900 NMPC solve-fail warnings each (domains 74, 97, 180); the fourth (domain 178) logged no solve samples at all. Cause not isolated: CPU starvation fits but was not tested. This does not replace the 2026-11-04 3/3 row. The lightly loaded rerun is the next paragraph.

NMPC solve time, from the `LocalPlanner took X ms` lines in `training/logs/run_logs/bench_nmpc_plan_<domain>.log` (`ros2/quad_sdk/local_planner/src/local_planner.cpp`), first 5 samples per run dropped. Caveats: the log is throttled to 100 ms, so it is a sample, and the timer covers the whole planner cycle, not only the IPOPT solve (that is `RobotPlan.diagnostics.compute_time`, not logged here). Pooled over all 10 runs (n=478): median 36.5 ms, p95 109.6 ms. Over the 6 runs that reached the goal (n=455): median 35.7 ms, p95 106.4 ms. Per-run medians 21.5 to 46.9 ms, the lowest under the lightest load. The README now quotes this lightly loaded median (about 20 ms) and the p95 (about 67 ms), not a flat 20-30 ms.

2026-11-22 idle-machine rerun: `scripts/benchmark_gazebo.sh --controller nmpc --trials 1 --goal 5.0` run ten times in a row, load average (1 min) checked before each run (4.8, 7.1, 7.7, 7.3, 6.6, 5.7, 5.7, 5.8, 6.0, 6.3; the limit was 8, so none was paused). Result: 10/10 reached, 0 falls, 7.56-7.62 s, 0.624-0.629 m/s. The machine was never fully idle (load 4.8-7.7 from other sessions), so this is a lightly loaded result. Solve time (`LocalPlanner took X ms`, same method and caveats as above, first 5 samples per run dropped) over 9 plan logs, n=662: median 19.7 ms, p95 67.4 ms, max 128 ms; per-run medians 17.1-24.1 ms. Two runs drew the same random domain (72), so one of them lost its plan log to the other; 9 logs, not 10. The p95 is dominated by start-up samples and a few slow cycles. The per-run median is in the README's 20-30 ms range; the p95 is not.

### What `data/gazebo_benchmark.csv` contains (26 data rows)

The CSV has no world, date or load column, so the groups below are by row order; the dates come from the sections above and the load figures come from `uptime` or `/proc/loadavg` readings taken during the runs (the 1-minute value before each run) and are not stored in the CSV. All simulation, Gazebo Harmonic, flat world `big_flat.sdf` for NMPC, goal 5 m.

| CSV rows (data rows) | Controller | Trials | Reached | Falls | Load | Notes |
|---|---|---:|---:|---:|---|---|
| 1-3 | NMPC | 3 | 3 | 0 | not recorded | 2026-11-04, the original 3/3 (0.626-0.628 m/s) |
| 4-6 | RL walker, first attempt | 3 | 0 | 2 | not recorded | heading gain 4.0, vx 0.35; the gentler 3/3 run is in `gazebo_benchmark_rl_gentle.csv`, not here |
| 7-16 | NMPC | 10 | 6 | 0 | 42-49 during trials 3-5, about 12 at the end (see the 2026-11-22 note) | trials 3, 4, 5, 7 stalled at the start; cause not isolated |
| 17-26 | NMPC | 10 (ten runs of `--trials 1`, so every row says `trial` 1) | 10 | 0 | 4.8-7.7 before each run | lightly loaded rerun, 0.624-0.629 m/s |

Total: 23 NMPC rows and 3 RL rows. `sim_s` and `wall_s` are the last two columns.

Per-trial rows: [data/gazebo_benchmark.csv](data/gazebo_benchmark.csv) (NMPC and the first RL attempt) and [data/gazebo_benchmark_rl_gentle.csv](data/gazebo_benchmark_rl_gentle.csv).

MuJoCo reference (same command logic, same walker, 5 trials each): reaches 5 m in about 11.6-12.2 s with 0 falls at heading gain 1-2 (max lateral 0.6-0.7 m); with no heading correction it still reaches 5 m but curves away (about 2 m sideways, heading turns ~30 deg).

An earlier RL attempt used heading gain 4.0 / vx 0.35 and failed in Gazebo (0/3 reached, 2 falls). That setting is not a fair test of the sim gap: in MuJoCo the same walker falls in 10 of 10 episodes with gain 4.0, because it cannot take large yaw commands while walking. Keep the heading gain at 1-2.

Read this with care:
- The two stacks run in different worlds and robot models (NMPC: Quad-SDK `big_flat.sdf`; RL: `go2_gz_world_open.sdf` with the arm-free URDF), so it compares the full stacks, not the controllers on identical physics.
- The machine was heavily loaded (load 20-40 on 22 cores) during these runs and Gazebo ran below real time. The RL policy node ticks on `/clock`, so this is not a wall-clock tick problem, but CPU contention can still add jitter.
- The RL walker drifts sideways (0.2 to 1.9 m over 5 m) in Gazebo and in MuJoCo; NMPC holds the line to about 2 cm.
- Three trials per controller is a small sample.

## Indoor Autonomy

Command:

```bash
ros2 launch launch/indoor_autonomy_go2.launch.py headless:=true rviz:=false
```

Metrics to report:

| Date | Locomotion | World | Map available | Exploration complete | Obstacle tracks | Notes |
|---|---|---|---|---|---|---|
| 2026-08-27 | CHAMP | `go2_gz_world_room.sdf` | Yes, 343 WM nodes over ~6.3min | No -- 20 frontier goals abandoned ("no progress... for 12s"), repeatedly re-targeting the same 2 unreached corners | Tracker started, 1 track seen | After the odom-transport and gait-damping fixes (see below): 0 TF-extrapolation warnings, 1 octomap queue-drop (startup-only, vs. constant before), idle drift down to ~0.1-0.15m x2 (was 2.78m). Map is real and growing but two corners never got reached -- likely a residual CHAMP path-following gap, not the TF/gait issue. |
| 2026-08-27 | Quad-SDK NMPC | `go2_room.sdf` (copy of the room world under quad_sim_scripts' worlds/, renamed to Quad-SDK's expected `default` world name, with a matching flat terrain mesh) | No -- WM stuck at 3 nodes | False positive: reported "exploration complete" after 30s of no frontiers, but only because the robot never left spawn (tiny map => no frontiers to find) | Not reached (robot never explored) | Spawns and loads cleanly now (model URIs + terrain mesh resolve, no resource errors) after wiring `ros2/quad_sdk_external/setup_env.sh`'s env vars into `slam3d_go2.launch.py`. But `local_planner_node` failed "NMPC solving fail, ApplicationReturnStatus = -4" continuously (260 times in ~90s) and the robot never moved. Needs its own investigation (solver config/IPOPT), separate from the world-wiring fix -- do not read the "exploration complete" line as success for this backend yet. |

## Locomotion Backends

| Backend | Flat world | Room world | Outdoor world | Rough world | Notes |
|---|---|---|---|---|---|
| CHAMP | Working baseline | Working (maps successfully; frontier nav struggles on 2 far corners) -- see Indoor Autonomy 2026-08-27 | Needs tracker verification | TBD | Native `/cmd_vel` backend. Not physics-tuned for Go2 (generic reference gait, see README) -- produced visible hopping on contact until the 2026-08-27 d_gain fix. |
| Quad-SDK NMPC | Working baseline | Spawns/loads cleanly (2026-08-27 world-wiring fix) but local planner fails to solve continuously in this world -- robot never walks. Needs solver-level debugging. | Working baseline | TBD | Goal-driven backend, verified end-to-end on flat/outdoor terrain per README; the room-world failure above is new information, not previously covered. |
| MuJoCo RL | Working with stability issue | N/A | N/A | Needs longer sighted/blind run | Fastest training loop. |
| Gazebo RL | Needs fresh training | TBD | TBD | TBD | Real ROS/Gazebo loop, slower but closer to deployment. |

## Open issues (not fixed, not re-run for this note)

Both are simulation-only findings in the vendored Quad-SDK (`ros2/quad_sdk`).

1. **`global_body_planner_node` segfaults on `gap_80cm.sdf`** (exit code -11, reproduced on 2026-11-22 at load 4.35). It dies right after `valid and shorter or previous plan not valid`, shortly after the first plan is accepted in `GlobalBodyPlanner::callPlanner`. Repro, about 2 minutes, headless:
   ```bash
   scripts/benchmark_gazebo.sh --controller nmpc --trials 1 --goal 5.0 --nmpc-world gap_80cm.sdf --timeout 40
   grep "exit code" training/logs/run_logs/bench_nmpc_plan_<domain>.log   # domain is printed by the script
   ```
   A gdb attempt (planner under `launch-prefix`, load 25) finished without a crash and gave no backtrace, so it did not locate the fault. Suspected cause (unverified, no sim run yet against a fix): `GBPL::postProcessPath` in `gbpl.cpp`, see [quadsdk_notes.md](quadsdk_notes.md). `parkour_local_min.sdf` also ended with the planner dead (exit -2), not confirmed as a segfault. Details: [quadsdk_notes.md](quadsdk_notes.md#re-check-2026-11-22-global_body_planner-segfault).
2. **Front-leg torque about 2x the 33.5 Nm limit on `step_25cm.sdf` and `step_30cm.sdf`** (67 and 46 warnings in a 45 s run, from `docs/quadsdk_notes.md`, an earlier run; not re-measured). NMPC still solves and the robot does not crash. Repro: `scripts/benchmark_gazebo.sh --controller nmpc --trials 1 --goal 5.0 --nmpc-world step_25cm.sdf`; the warning text and log file were not checked for this note, so search the run logs in `training/logs/run_logs/` for "torque".
