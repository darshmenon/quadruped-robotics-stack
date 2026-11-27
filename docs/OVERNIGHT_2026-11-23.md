# Overnight 2026-11-23 (Go2 repo)

Status written at 02:05 IST. Nothing committed or pushed tonight; edits stay in the working tree.

## Gate at 02:05
Load 3.9 (limit 8) but MemAvailable 2.8 GB (limit 4000 MB), so the sim gate was not clear. rosnav-ed (nav2_doctor) and ur3-ros2-pick-and-place-b1 (50-trial run) were ahead in the queue and had not messaged yet.

## Armed at 02:06 IST
Detached gate: `scripts/gate_go2_overnight.sh` (pid in `docs/overnight_logs/gate_go2_overnight.log`, started with setsid nohup, 6 h window). It waits until load < 8, MemAvailable >= 4000 MB, no gz sim / benchmark_mtc / nav2_fault_inject / benchmark_gazebo / gate_mtc / ros2 launch / colcon build process, clear for 3 checks 30 s apart. Then, one at a time:
1. `nice -n 19 colcon build --packages-select global_body_planner` -> `docs/overnight_logs/build_global_body_planner.log`
2. gap_80cm NMPC run (own random domain, <= 232) -> `docs/overnight_logs/gap80_check.log`, planner exit lines in `docs/overnight_logs/gap80_planner_lines.log`, row in `docs/data/gap80_check.csv` (not the main benchmark CSV)
3. MuJoCo climber evals, 3 cm and 5 cm, 10 episodes, seeds 0-9, `scripts/eval_climber_mujoco.py` (syntax-checked only) -> `docs/overnight_logs/climber_h3.log`, `climber_h5.log`, rows in `docs/data/climber_eval.csv`
Progress lines: `docs/overnight_logs/gate_go2_overnight.log`. The old-code crash reproduction is not repeated: the old binary no longer exists. The crash was reproduced earlier on 2026-11-22 (gap_80cm, exit -11, load 4.35). To stop the gate: `kill <pid from the log>`.
Gate state at 02:07: load 7.2, MemAvailable 1.9 GB, so it is waiting.

## Not started yet
1. gap_80cm check: reproduce the planner crash on the old code, then run with the guards. The guards are in ros2/quad_sdk/global_body_planner/src (global_body_plan.cpp from this session; gbpl.cpp, rrt.cpp, planning_utils.cpp, global_body_planner.cpp from another session). The build in ros2/install was made before the other session's later edits, so rebuild first. Command: `scripts/benchmark_gazebo.sh --controller nmpc --trials 1 --goal 5.0 --nmpc-world gap_80cm.sdf --timeout 40`, then `grep "exit code" training/logs/run_logs/bench_nmpc_plan_<domain>.log`.
2. MuJoCo measurements for the unsupported README claims (3 cm climber, stair rows). Nothing measured yet. Policies: training/pretrained/go2_climb_scan/go2_stairs_h03.zip and go2_stairs_h05.zip; no committed script prints per-episode falls and steps cleared for these scenes (training/office_demo.py --steps only records a clip), so an eval script has to be written first.
3. README softening for claims that cannot be backed: wait for item 2.

## Results
None yet.

## Results (gate finished 06:54 IST; nothing committed or pushed)
- Build: global_body_planner "up to date" (0.39 s). The installed binary (23:27) is newer than all five edited sources (latest 23:26), so the run below used the guarded code.
- gap_80cm, one trial, domain 153: the planner STILL segfaulted (global_body_planner_node exit code -11, local_planner -2), robot did not move (reached 0, falls 0). `docs/data/gap80_check.csv`, `docs/overnight_logs/gap80_check.log`, `docs/overnight_logs/gap80_planner_lines.log`. So the guards in gbpl.cpp, rrt.cpp, planning_utils.cpp, global_body_planner.cpp and global_body_plan.cpp do not fix this crash. The "fault is in postProcessPath" statement in the uncommitted docs/quadsdk_notes.md paragraphs is therefore not confirmed. Not re-run on the old code.
- Climber, MuJoCo, seeds 0-9, `docs/data/climber_eval.csv` (script `scripts/eval_climber_mujoco.py`): 3 cm: 10/10 reached the top platform, 0 falls, 0 stalls, 4 steps cleared each. 5 cm: 3/10 reached top, 7 stalled (about 6.5 s, stuck before the first step, max_x about 0.75 m), 0 falls. The 5 cm settings (MoE, from office_demo --h) are unverified for this checkpoint, so this does not prove the README's 18/30 wrong.
- README (working tree, uncommitted): added the measured 3 cm and 5 cm results next to the existing claims. SLAM with NMPC and the other unsupported claims untouched, no evidence run.

## Follow-up 2026-11-23 midday (asimov-38 tasks; nothing committed or pushed)
1. Docs reworded (working tree): docs/quadsdk_notes.md now calls `GBPL::postProcessPath` an unconfirmed hypothesis, adds the negative gap_80cm result (guards built, still exit -11) and removes the "Rebuilt ... not re-run" sentence. docs/BENCHMARKS.md open issue 1 the same.
2. Backtrace: NOT obtained yet. No core file exists, /var/crash has no report for the planner (apport ignores it), and `ulimit -c` is 0 here. The earlier gdb run under load 25 did not crash, probably because gdb turns ASLR off. Armed a gated run: `scripts/gate_go2_gdb.sh` (pid in `docs/overnight_logs/gate_go2_gdb.log`, 8 h window, same gate as before, /usr/bin/python3, clean PATH). When clear it builds a RelWithDebInfo copy of global_body_planner in a SEPARATE install (`docs/overnight_logs/dbg_install`, `ros2/install` untouched), then runs gap_80cm once with the planner under gdb with ASLR on, using `scripts/tmp_overnight/bench_gdb_gap80.py`. Results: `docs/overnight_logs/gap80_gdb.log`, `gap80_backtrace.txt`. The gate has not fired yet (load was 10.6 at last check).
3. 5 cm climber settings: the shipped `go2_stairs_h05.zip` is a plain MLP (net_arch [512, 256, 128], no MoE keys), so `GO2_MOE` plays no role at load time. Its md5 matches none of the training outputs under logs/mujoco_curriculum. The first 5 cm eval (3/10) used `vecnorm_h05.pkl`; `training/office_demo.py` loads `vecnorm_final.pkl` for any model. With `vecnorm_final.pkl` the same checkpoint gives 8/10 reached top, 1 fall, 1 stall (seeds 0-9) and 26/30 reached top, 1 fall, 3 stalls (seeds 100-129). Scene variant (unitree vs plain xml) and the MoE env vars changed nothing (3/10 each with vecnorm_h05). Files: docs/data/climber_eval.csv (3 cm 10 rows, 5 cm 10 rows, now with a variant column), climber_eval_5cm_30ep.csv, climber_eval_5cm_variants.csv (other checkpoints and variants), climber_eval_first_run_5cm_vecnorm_h05.csv. The other training outputs (moe2/moe3/moe4b/cur5/cur5b) gave 0-1 of 10 at the top. Caveat: which vecnorm the "18 of 30" figure used is unknown; that figure is replaced in the README by the measured numbers.
4. README (working tree): only measured claims for the 3 cm and 5 cm rows; blind rows marked "README figure, unmeasured here, no committed eval"; SLAM with NMPC marked "demonstrated in a clip, unmeasured". The earlier extra table column on the 5 cm row was a mistake of mine and is fixed.

## Round 2 (asimov-38; nothing committed or pushed)
### 1. NMPC replication, armed
`scripts/gate_go2_nmpc_rep2.sh` (pid 215606, log `docs/overnight_logs/gate_go2_nmpc_rep2.log`, 8 h window). Before each run it needs load < 8 and MemAvailable >= 4000 MB on two checks a minute apart, with no other sim/build process (gz sim, benchmark_mtc, nav2_fault_inject, benchmark_gazebo, bench_gdb, gate_mtc, ros2 launch, colcon build, eval_climber); the run is `scripts/benchmark_gazebo.sh --controller nmpc --trials 1 --goal 5.0 --csv docs/data/gazebo_benchmark_nmpc_rep2.csv`, 20 times, one fresh Gazebo each, own random domain 60-230. The 1-min/5-min load and MemAvailable before each run go to `docs/overnight_logs/nmpc_rep2_loads.csv`, per-run output to `nmpc_rep2_run<i>.log`. It runs after or between the gdb gate's sims because both check for the other's processes (small race window between two gates starting in the same minute is possible). Report with `python3 scripts/nmpc_wilson_report.py` (reads the CSVs only). Not yet run: only the existing 10 are available: n=10, reached 10 (Wilson 95% 72.2-100.0%), falls 0 (0.0-27.8%). For reference only, not measured: if the 20 new trials all reached the goal, 30/30 would give 88.6-100.0%.

### 2. Backend comparison, measured rows only (Gazebo, simulation only, 5 m flat goal unless noted)
| Backend | Measurement | Result | Source |
|---|---|---|---|
| Quad-SDK NMPC | 5 m, `big_flat.sdf`, 3 trials (2026-11-04) | 3/3 reached, 0 falls, 0.626-0.628 m/s | gazebo_benchmark.csv rows 1-3 |
| Quad-SDK NMPC | 10 trials, load 42-49 during trials 3-5 | 6/10 reached, 0 falls, 4 stalled at start, cause not isolated | rows 7-16 |
| Quad-SDK NMPC | 10 runs, load 4.8-7.7 | 10/10 reached, 0 falls, 0.624-0.629 m/s, planner cycle median 19.7 ms, p95 67.4 ms (n=662, throttled log) | rows 17-26, plan logs |
| Quad-SDK NMPC | gap_80cm world | planner segfaults (exit -11), robot does not move | gap80_check.csv |
| PPO walker `go2_walk_lat2` | Gazebo `go2_gz_world_open.sdf`, heading gain 4.0 | 0/3 reached, 2 falls | rows 4-6 |
| PPO walker `go2_walk_lat2` | Gazebo, gentler setting (vx 0.3, heading gain 1.5) | 3/3 reached, 0 falls, mean speed 0.21-0.45 m/s, lateral drift 0.16-1.93 m | gazebo_benchmark_rl_gentle.csv |
| PPO walker `go2_walk_lat2` | MuJoCo flat, command tracking | 0% falls at the commands in the file; speed tracking loose | walk_tracking_lat2.csv |
| PPO climber `go2_climb_scan` | MuJoCo 3 cm steps, seeds 0-9 | 10/10 top, 0 falls | climber_eval.csv |
| PPO climber `go2_climb_scan` | MuJoCo 5 cm steps (vecnorm_final.pkl) | 8/10 top (seeds 0-9), 26/30 (seeds 100-129) | climber_eval.csv, climber_eval_5cm_30ep.csv |
| CHAMP | Gazebo indoor autonomy, room world, 2026-08-27 | maps (343 RTAB-Map nodes in ~6.3 min), 20 frontier goals abandoned; this is a SLAM run, not a walking benchmark | BENCHMARKS.md Indoor Autonomy table |
| CHAMP | straight-walk goal benchmark | not measured | none |
| PPO | stairs/terrain in Gazebo | not measured (README says partial) | none |
| Quad-SDK NMPC | SLAM with locomotion:=nmpc, terrain worlds beyond big_flat | not measured / crashes on several | none |
Caveat: the NMPC and RL rows use different worlds and robot models, so this compares stacks, not controllers; trial counts are small.

### 3. PPO training claims, evidence in the repo
- Domain randomisation: CODE present, tracked: `training/envs/go2_mujoco_env.py` (gains, gyro, latency, friction and mass ranges, push; env vars GO2_RAND_GAINS/GYRO/LAT, GO2_FRIC_RANGE, GO2_MASS_RANGE, L216-268, L829-842) and the training scripts export them (`training/run_step_curriculum.sh`). The recipe for `go2_walk_lat2` is only in README text; the shipped weights (tracked: `training/pretrained/go2_walk_lat2`) have no config file or log beside them, and `training/logs/` is gitignored, so training logs exist only on this machine (e.g. `training/logs/mujoco_curriculum/flat_lat_run`; I did not verify that it produced the shipped zip). No measured effect of randomisation is committed except the robustness sweep tooling (`training/eval_robustness.py`, partial output `docs/data/robust2_sweep_partial.txt`, untracked).
- Staged curriculum: CODE present: `training/train_curriculum.py` stages flat -> rough -> stairs (timesteps 1M/1M/0.5M) with `--init-from` resume. Local, gitignored final folders exist for flat, rough, stairs and mixed, with curriculum_level.txt values 0.73, 0.40, 0.27 and 0.21 (so those runs did not finish the curriculum), and the README itself says the baseline walker's curriculum stayed at level 0.02. Nothing tracked shows that a policy was trained through flat -> rough -> stairs in one chain; the shipped climbers come from `run_step_curriculum.sh` (steps scenes), not from train_curriculum's stairs stage, as far as I can tell.
- "Flat-to-rough-to-stairs" as a headline claim: supported as an available pipeline, not as a measured result. Reword to "staged training pipeline (flat, rough, stairs stages) and a step-height curriculum for the climber".

## NMPC fall counts: caveat (2026-11-26)

The fall counts for NMPC above used an absolute base height below 0.12 m, which on `big_flat.sdf` (surface at z = +0.1) is only 0.02 m above the ground. Under the ground-relative rule, `data/gazebo_benchmark_nmpc_rep2.csv` data row 18 (`min_z` 0.16) and `data/gazebo_benchmark.csv` data row 11 (`min_z` 0.20, heavy-load trial 5) would be flagged, unconfirmed: only min height was recorded. The heavy-load 10 would read 1/10 falls (Wilson 95% 1.8-40.4%) instead of 0/10; the lightly loaded 10 plus the 20-run replication would read 1/30 (0.6-16.7%) instead of 0/30. Rows 1-3 and 17-26 are unchanged. The recorded CSVs are unchanged. Details: [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md), section 8.
