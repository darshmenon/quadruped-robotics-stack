# Troubleshooting

## No `/map`

- Wait at least 20 seconds; `slam3d_go2.launch.py` starts RTAB-Map after Gazebo and TF are ready.
- Check `/points`:

```bash
ros2 topic hz /points
```

- Check TF:

```bash
python3 scripts/check_stack_go2.py --skip-obstacles
```

## Obstacle Tracker Publishes Nothing

- Launch with `track_obstacles:=true`, or use `launch/indoor_autonomy_go2.launch.py`.
- Confirm point cloud input:

```bash
ros2 topic hz /points
```

- Confirm map TF exists:

```bash
ros2 run tf2_ros tf2_echo map base
```

## Gazebo RL Cannot Find Topics

Use matching isolation values for the sim and evaluator/trainer:

```bash
python3 training/train_gazebo.py --ros-domain-id 177 --gz-partition go2rltrain
python3 training/eval_gazebo.py --ros-domain-id 177 --gz-partition go2rltrain --no-launch
```

Do not reuse the same `ROS_DOMAIN_ID` and `GZ_PARTITION` as a separate SLAM/Nav2 launch unless the sessions are intentionally sharing topics.

## RViz Shows No Robot Or Cloud

- Source both ROS and this workspace:

```bash
source /opt/ros/humble/setup.bash
source ros2/install/setup.bash
```

- Keep `ROS_DOMAIN_ID` consistent across terminals.
- For the default indoor demo, use `ROS_DOMAIN_ID=157` unless overridden.

## MuJoCo Policy Doesn't Walk (falls in ~2 s, or stands still)

`ep_rew_mean` is not evidence of walking: the curriculum raises command speed and pushes as it climbs, so
reward can drop while the policy improves (and a policy that sits can still score positive). Check with the
fixed-difficulty tools instead:

```bash
# fall rate / distance / speed per checkpoint, clean env, in-range commands
python3 training/eval_checkpoints.py --models <ckpt.zip> ... --scenes flat --cmds 0.2 0.35
# per-term reward breakdown + the last steps before a fall (add --video out.mp4 with MUJOCO_GL=osmesa)
python3 training/analyze_rollout.py --model <ckpt.zip> --scene flat --cmd 0.2
```

Keep `--cmd` inside what the checkpoint was trained on (training samples `vx <= 0.15 + 1.05 * curriculum_level`).
Use `--blind` with `play_policy.py` on the stairs scene or the obs size won't match.

Failure modes seen and their fixes:

- **Action std stuck near 1.0** (check `policy.log_std.exp()`): with actions clipped to [-1, 1] the policy is mostly
  noise and saturates. Fixed by `LOG_STD_INIT = -1.0`, `ent_coef = 0`, and clamping std to `MAX_RESUME_STD` on resume.
- **KL early-stops every update on resume**: loaded policies keep their saved lr; `RESUME_LR = 1e-4` fixes it.
- **Policy sits down**: `collision`/`soft_limit` penalties at full weight from step 0, a wide `lin` kernel that paid
  ~1.3/step for standing still, and a near-zero `height` weight. See `LIN_TRACK_SIGMA`, `HEIGHT_WEIGHT`,
  `PENALTY_RAMP_*` in `training/envs/go2_mujoco_env.py`.
- **Stands but never steps**: `air_time` only pays at touchdown, so a standing robot never sees it. Train with
  `--gait` (dense contact-clock reward, `GAIT_CONTACT_WEIGHT`); legs-only obs then include the gait clock.
- **Resuming a `--asymmetric` run**: pass `--asymmetric` again or `VecNormalize.load` fails on the Dict obs space.
- **Per-checkpoint VecNormalize**: checkpoints now save `go2_<stage>_vecnormalize_<steps>_steps.pkl` next to the zip.
