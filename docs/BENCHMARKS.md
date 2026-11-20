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
- Turning while walking costs about 4 to 5 times the straight-walk APE.
- One bag per walk type, one walker (`go2_walk_lat2`), 0.2 to 0.35 m/s. The walker itself is erratic (median 0.17 m/s while moving, roll up to 17 deg, the heading wanders), and an identical weave later covered only 1.7 m.
- The estimated IMU lag against `/odom` is about +0.045 to +0.06 s (signed gyro-z correlation, 0.7). Its cause is not established: shifting the IMU stamps by +0.06 s made one replay clearly worse (APE 0.63 m), -0.06 s one replay 0.22 m, within the unshifted spread.
- LIO-SAM (`launch/lio_sam_go2.launch.py`, `config/lio_sam_go2.yaml`, ring/time adapter) does not track on these bags: APE 139.7 m on the straight walk (it only produced 71 poses) and 7.4 m on the weave, one replay each, so it is experimental here. The sim IMU has no noise model, so its IMU noise values are guesses.
- Leg odometry from `/joint_states` (`scripts/leg_odom_go2.py`) is a weak source in this sim: velocity error 0.66 m/s against a true speed of ~0.16 m/s, 1.7 m APE on the 19.4 m weave, and fusing it with RKO-LIO in an EKF gave no gain (APE within 1 mm of RKO-LIO alone). A single velocity scale fitted on one bag does not transfer: the best scale is about 1.0 on the straight walk (APE 0.71 m) and about 1.3 on the weave (0.96 m); the weave scale gives 1.04 m on the straight walk and the straight-walk scale gives 1.73 m on the weave. Even the best fit is far worse than RKO-LIO.
- `online_node` always segfaults when it is shut down (after printing its timing stats); the estimate recorded before that is complete.
- `deskew:=false` (the sim cloud has no per-point time), voxel 0.25 m, max range 30 m. `rko_lio`'s launch file reads only `param:=value` from the command line, so it is started as a subprocess.

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
