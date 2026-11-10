# Ideas from the blog and concept guides: what we can still do in the quadruped stack

Cross-pollination notes from `robocloud-dashboard` posts (base URL `https://robocloud-dashboard.vercel.app/learn/blog/`).
Rule: ideas only. Do **not** commit or push unless asked.
Novelty and "does it help" are not verified. Blog sections were read by heading, not end to end.
Facts about the repo are from `README.md` (Roadmap and "What's working"); re-check before relying on them.

_Written 2026-11-07._

## 1. Give the arm a job (Roadmap item 5)
**Posts:** `mobile-manipulation-systems`, `moveit2-whole-body-mobile-manipulation-planar-joint`.
**Gap here:** the README says the manipulator arm is "currently decorative only". `intelligence/manipulation/` exists.
**Build:** the smallest honest task: Go2 walks to a table-height target under the Quad-SDK or RL backend, stops, and the arm reaches a marked point. Treat the base as a stationary platform during the reach first, then add the whole-body step from the planar-joint post only if the reach works. Report arm-induced tilt of the trunk while walking with the arm attached vs `go2_walk_noarm`.
**Measure:** reach error, trunk pitch/roll during the reach, falls.
**Effort:** medium.

## 2. Variable impedance only counts if it does something
**Post:** `quadruped-variable-impedance-ppo` ("checking whether the learned stiffness is doing anything", checkpoint compatibility, motor-aware rewards).
**Gap here:** `impedance` appears in only two files and there is no ablation anywhere in the repo (no file mentions "ablation").
**Build:** train `go2_walk_lat2`-style policies with fixed gains vs learned per-joint stiffness, same seeds and terrain, and compare. Include a probe: freeze the learned stiffness at its mean and see if success drops.
**Measure:** success on the step-climbing task (reaches the top in about 60% of episodes today), energy per metre, falls.
**Effort:** medium to large (training time).

## 3. Close the step-climbing gap with a curriculum audit
**Posts:** `legged-locomotion`, `robot-policy-distillation`, `domain-randomization-physical-ai`.
**Gap here:** the climbing clip is the best of three attempts and the policy reaches the top in about 60% of episodes. The repo has `train_stairs.py`, `run_step_curriculum.sh` and `train_parkour.py`, plus privileged-information code.
**Build:** report the success rate per step height with a confidence interval over a fixed seed set, then try distilling a privileged-observation teacher into a height-scan student, and add randomization ranges one at a time.
**Measure:** success vs height curve before and after each change.
**Effort:** medium.

## 4. A fall detector and safe-stop layer
**Posts:** `runtime-assurance-safety-monitor`, `safe-reinforcement-learning`, `robot-safety-case-deployment`.
**Gap here:** `train_recovery.py` and a fall-recovery env exist, and the roadmap says "need more training/verification". Nothing watches the policy at run time and forces a safe behaviour.
**Build:** a monitor node with simple invariants (trunk roll/pitch limits, joint-limit proximity, command latency). On violation: stop the policy, hold a stand pose, log the cause. Also useful as the trigger for the recovery policy.
**Measure:** falls with the monitor vs without under pushes in MuJoCo.
**Effort:** small to medium.

## 5. Sim-to-sim as a deployment check
**Posts:** `domain-randomization-physical-ai`, `robotics-simulators-physics-engines-2026`, `mujoco-in-gazebo-physics-engine-integration`.
**Gap here:** one PPO policy trained in MuJoCo also runs in Gazebo, but the Gazebo result is "~25% of 0.25 m/s over a short window", flagged as probably needing gait tuning.
**Build:** a single script that runs a checkpoint in MuJoCo and Gazebo with identical commands and prints a gap table (speed tracking, tilt, energy, falls). Use the gap to decide what to randomize (friction, latency, motor strength).
**Effort:** small; `eval_gazebo.py` and `eval_checkpoints.py` already do most of it.

## 6. Navigation on a legged base with the lessons from the wheeled stack
**Posts:** `ros2-outdoor-terrain-navigation-slope-costmap-localization`, `nav2-costmap-filters-keepout-speed-binary`, `quadruped-outdoor-hazardous-environments`.
**Gap here:** Nav2 runs on both locomotion backends, but the costmap treats stairs and ledges like any wall. The sister repo `rosnav` has terrain speed masks and CBF safety filters.
**Build:** a traversability layer from the existing height scan, so a Go2 that can climb 5 cm steps treats them as passable and larger ledges as keepout; then a speed filter on rough patches.
**Measure:** goals reached on the stairs/ledge worlds with the layer on vs off.
**Effort:** medium. Could share code with rosnav's `ply_to_traversability.py`.

## 7. Calibrated claims for the README
**Post:** `conformal-prediction-robot-safety` (and general habit from the benchmarks posts).
**Gap here:** several README numbers are single runs ("best of 3", short windows).
**Build:** a small `docs/BENCHMARKS.md` convention: every number gets N, seed list and an interval. Cheap, and it removes the "best of 3" caveat from the headline GIF.
**Effort:** small.

## Suggested order
7 and 5 first (cheap, make everything else measurable) -> 4 -> 3 -> 2 -> 6 -> 1.
