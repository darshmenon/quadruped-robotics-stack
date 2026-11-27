# Benchmark protocol

One protocol for every controller row (Quad-SDK NMPC, CHAMP, the RL walker), so the numbers in `docs/BENCHMARKS.md` can be compared. It is built on `scripts/benchmark_gazebo.py`, whose CSV already carries the NMPC and RL rows.

Status: the protocol is written down here. `benchmark_gazebo.py` runs `nmpc` and `rl`, and writes the load columns below. `champ` is a stub (see "Gaps").

## 1. What is held fixed

| Item | Value |
|---|---|
| Simulator | Gazebo Harmonic, headless, one sim at a time on the machine |
| World | `big_flat.sdf` for flat runs (the script default); named world for terrain runs |
| Task | Walk forward `--goal 5.0` m from a standing start, tolerance `--tol 0.25` m |
| Time limit | `--timeout 60` sim seconds per trial |
| Fall | base less than `--fall-z 0.12` m above the ground (`z - ground_z`); `min_z` stays the raw world z |
| Trials | 10 per controller and world (3 is a smoke test only, never a table row) |
| Truth source | simulator ground-truth pose, not the controller's own estimate |
| Command | Each controller gets its own native forward command. Record it: NMPC goal 5.0 m, RL `--rl-vx` (0.4), CHAMP `/cmd_vel` vx. Speeds are compared as achieved `mean_speed`, not as commanded |

## 2. Metrics per row

From the CSV columns: `reached`, `fell`, `time_s`, `dist_m`, `mean_speed`, `max_lateral_m`, `final_lateral_m`, `heading_change_deg`, `min_z`, `path_m`, `sim_s`, `wall_s`.

Report per controller and world:

- reached N/10 and falls N/10 (both always shown, even when 0)
- mean speed over reached trials, with min and max
- max lateral drift
- planner solve time p50 and p95 where the controller has a planner (NMPC: from `training/logs/run_logs/bench_nmpc_plan_*.log`; CHAMP and RL: "n/a", not 0)

Never average only the successes without also giving reached N/10.

## 3. Load tag

Machine load changes results: the 2026-11-22 NMPC run reached 6/10 at load 42-49 and 10/10 at load 4.8-7.7.

- Record the 1-minute load average (`uptime`) at trial start and at trial end. Use the larger as `load_max`.
- `load_max` < 8: valid, goes in the table untagged.
- `load_max` >= 8: tag the row **`[LOAD>8]`**. A tagged row is reported as a note and never counts as controller evidence, in particular not for a failure count. Re-run it when the machine is quiet.
- Also record `MemAvailable` (needs > 3000 MB to start) and whether any other `gz` or `ros2 launch` was running (`ps`). If one was, tag `[SHARED]`.
- The 1-minute load average lags. If the load rose during a trial, `load_max` captures it; do not use only the start value.

CSV columns to append (additive, existing columns unchanged): `load_start,load_end,load_max,mem_avail_mb,tag`.

## 4. Per-run header

Write above each table or in the CSV notes:

```
date, git commit (short), controller, world, goal, trials,
command line, ROS_DOMAIN_ID, load_start/end/max, MemAvailable MB, tag
```

## 5. Files

- Raw CSV: `docs/data/gazebo_benchmark_<controller>_<world>[_repN].csv`. Do not overwrite another run's CSV. The script's default `docs/data/gazebo_benchmark.csv` is shared, so pass `--csv` explicitly. Delete throwaway rows (for example a crash check) so they don't stay in the shared file.
- Summary table: `docs/BENCHMARKS.md`, one row per controller and world, linking the CSV.
- One sim at a time, private `ROS_DOMAIN_ID` in [50,232], and set `GZ_PARTITION` to match.

## 6. Command

```bash
uptime; grep MemAvailable /proc/meminfo; ps aux | grep -E "gz sim|ros2 launch" | grep -v grep
ROS_DOMAIN_ID=<fresh> GZ_PARTITION=<fresh> \
  scripts/benchmark_gazebo.sh --controller nmpc --trials 10 --goal 5.0 \
  --csv docs/data/gazebo_benchmark_nmpc_flat.csv
```

## 7. Gaps (not done yet)

1. `benchmark_gazebo.py --controller champ` is a stub that raises NotImplementedError. A real backend needs a launcher plus a `/cmd_vel` drive, like the RL one, and can only be tested with a sim.
2. Done (untested in a sim): the script writes `load_start,load_end,load_max,mem_avail_mb,tag` to the CSV. It refuses to append to a CSV with the old header, so pass a new `--csv`.
3. The flat worlds differ (see "Flat world differences" below). Not fixed.
4. Solve-time logging exists for NMPC only.

## 8. Flat world differences (read from the files, not measured in a sim)

| | NMPC: `big_flat.sdf` | RL: `go2_gz_world_open.sdf` |
|---|---|---|
| World name | `default` | `go2_rl` (hardcoded in `gazebo_rl.launch.py` topics) |
| Physics step | 1 ms, ODE solver 300 iterations | 5 ms, default iterations |
| Ground | static mesh slab, walking surface at z = +0.1 m | plane at z = 0 |
| Friction | mu = mu2 = 1.0 | mu = mu2 = 1.0 |
| Scenery | none | visual-only pillars, lane marks, a camera model |

**Fall threshold (fixed in code, not run in a sim).** The script used to compare absolute base z with `--fall-z 0.12`. On `big_flat` that was 0.02 m above the surface, against 0.12 m for RL, so NMPC and RL fall counts in older CSVs are not comparable. In the existing CSVs, two NMPC runs that did not reach the goal have `min_z` below 0.22 (less than 0.12 m above the slab) and are recorded `fell=0` (row numbers count data rows from 1):

| Run | `min_z` (above slab) | Other recorded values | Log evidence (`training/logs/run_logs`) |
|---|---|---|---|
| `gazebo_benchmark_nmpc_rep2.csv` row 18 (2026-11-26, load 1.0 before the run) | 0.16 (0.06) | 0.88 m covered, heading -126.5 deg, lateral 0.53 m | domain 157: 803 NMPC solve-fail warnings up to the end of the run |
| `gazebo_benchmark.csv` row 11 (2026-11-22, heavy load) | 0.20 (0.10) | 0.04 m covered, heading 10.1 deg | domain 180 (matched by log time, not recorded): 3339 solve-fail warnings |

Applying the new rule to the recorded `min_z` gives these counts (Wilson 95%):

| Set | Falls, recorded | Falls, ground-relative |
|---|---|---|
| Lightly loaded 10 + replication 20 | 0/30 (0-11.4%) | 1/30 (0.6-16.7%) |
| Replication 20 | 0/20 (0-16.1%) | 1/20 (0.9-23.6%) |
| Lightly loaded 10 | 0/10 (0-27.8%) | 0/10 (0-27.8%) |
| Heavy-load 10 (2026-11-22) | 0/10 (0-27.8%) | 1/10 (1.8-40.4%) |
| 3 trials (2026-11-04) | 0/3 (0-56.2%) | 0/3 (0-56.2%) |

These are runs the rule would flag, not confirmed falls. The CSV keeps only the minimum height: no height trace, roll or pitch, and no bag was found, and the logs read here do not report the body pose. So a collapse or tip-over cannot be told apart from a deep crouch. The recorded CSVs are unchanged.


**Proposed fix, in order**
1. Done: the test is `z - ground_z < fall_z` (`ground_z` 0.1 for `big_flat.sdf`, 0 otherwise, override with `--ground-z`), and the CSV records `world` and `ground_z`. A single height cannot describe step, gap or rough worlds: there the ground under the robot changes along the path, so check `fell` by hand.
2. Needs a sim to validate: one shared bench world. Renaming or re-stepping either world changes what that controller was run or tuned on, so it is not a safe code-only change. Until then, label the world on every row.
