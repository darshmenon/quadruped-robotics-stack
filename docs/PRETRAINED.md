# Go2 model files

Model names, local paths, and the command to use each one.

## Use now (this repo)

### go2_mujoco_walk

SB3 `.zip` only (works in this repo). Headless smoke test:

```bash
export MUJOCO_GL=osmesa          # use glfw if you have a display
export CUDA_VISIBLE_DEVICES=""   # optional if CUDA is flaky

python3 training/play_policy.py \
  --model training/logs/mujoco/best_model.zip \
  --vecnorm training/logs/mujoco/vecnorm_final.pkl \
  --no-display --episodes 1 --max-steps 100 --cmd 0.5 0 0
```

Current local result: `RESULT: PASS (policy moving forward)`.

GUI / video:

```bash
# OpenCV window
python3 training/play_policy.py --model training/logs/mujoco/best_model.zip --cmd 0.5 0 0

# Headless record (Ctrl+C to stop; file written on exit)
python3 training/play_policy.py \
  --model training/logs/mujoco/best_model.zip \
  --no-display --record /tmp/go2_policy.mp4 --cmd 0.5 0 0

# Final checkpoint
python3 training/play_policy.py --model training/logs/mujoco/go2_mujoco_final.zip --cmd 0.5 0 0
```

Controls (GUI): `R` reset, `ESC` quit.

### go2_walk_trotprior

Current flat-ground walker (PPO + clocked trot prior), committed at `training/pretrained/go2_walk_trotprior/`.
Walks forward at the commanded 0.30 m/s with 0% falls; trained on forward commands only. Needs `GO2_TROT_PRIOR=1`.

```bash
GO2_TROT_PRIOR=1 python3 training/eval_checkpoints.py \
  --models training/pretrained/go2_walk_trotprior/go2_flat_walk.zip --scenes flat --cmds 0.2 0.3 0.45
python3 training/teleop_mujoco.py --model training/pretrained/go2_walk_trotprior/go2_flat_walk.zip --trot-prior
```

### go2_mujoco_stairs

Train the stairs model, then play it:

```bash
./scripts/train_stairs.sh --blind --init-from-flat --n_envs 4 --timesteps 300000 --device cpu

python3 training/play_policy.py \
  --model training/logs/stairs/best_model.zip --scene stairs --blind

# Headless smoke check
python3 training/play_policy.py \
  --model training/logs/stairs/best_model.zip \
  --vecnorm training/logs/stairs/vecnorm_final.pkl \
  --scene stairs --blind \
  --no-display --episodes 2 --max-steps 300
```

Current local result: `RESULT: PASS (policy moving forward)`.

Sighted stairs uses the 94-dim height scan:

```bash
./scripts/train_stairs.sh --n_envs 4 --timesteps 300000 --device cpu
python3 training/play_policy.py --model training/logs/stairs/best_model.zip --scene stairs

# Sighted headless check
python3 training/play_policy.py \
  --model training/logs/stairs/best_model.zip --scene stairs \
  --no-display --episodes 3 --cmd 0.4 0 0
```

### go2_champ_stairs and go2_quadsdk_step

```bash
ros2 launch launch/stairs_ledges_go2.launch.py course:=stairs
./scripts/walk_quadsdk_go2.sh 1.0 0.0 gui step_20cm.sdf
```

## Download model files

```bash
python3 scripts/download_pretrained.py                # all groups
python3 scripts/download_pretrained.py --only stairs  # sim2real + CTS stairs
python3 scripts/download_pretrained.py --only locomotion
python3 scripts/download_pretrained.py --only parkour
python3 scripts/download_pretrained.py --list
```

| Model name | Local path | Use with |
|------------|------------|----------|
| `flat_model_6800` | `training/pretrained/go2_locomotion/flat_model_6800.pt` | [Go2_Isaac_ros2](https://github.com/sallu-786/Go2_Isaac_ros2) |
| `rough_model_7850` | `training/pretrained/go2_locomotion/rough_model_7850.pt` | [Go2_Isaac_ros2](https://github.com/sallu-786/Go2_Isaac_ros2) |
| `rpl_rough_go2_model_2000` | `training/pretrained/go2_parkour/rpl_rough_go2_model_2000.pt` | [parkour-drl checkpoints](https://huggingface.co/real-jiashu-yu/parkour-drl-checkpoints) |
| `rpl_field_go2_model_40000` | `training/pretrained/go2_parkour/rpl_field_go2_model_40000.pt` | [parkour-drl checkpoints](https://huggingface.co/real-jiashu-yu/parkour-drl-checkpoints) |
| `rpl_visual_distill_go2_model_100000` | `training/pretrained/go2_parkour/rpl_visual_distill_go2_model_100000.pt` | [parkour-drl checkpoints](https://huggingface.co/real-jiashu-yu/parkour-drl-checkpoints) |
| `sim2real_walk` | `training/pretrained/go2_stairs/sim2real_walk.pt` | [go2-sim2real-deploy](https://github.com/saifahmadgit/go2-sim2real-deploy) |
| `sim2real_stairs` | `training/pretrained/go2_stairs/sim2real_stairs.pt` | [go2-sim2real-deploy](https://github.com/saifahmadgit/go2-sim2real-deploy) |
| `sim2real_stairs_39cm_104000` | `training/pretrained/go2_stairs/sim2real_stairs_39cm_104000.pt` | [go2-sim2real-deploy](https://github.com/saifahmadgit/go2-sim2real-deploy) |
| `cts_moe_policy` | `training/pretrained/go2_stairs/cts_moe_policy.pt` | [go2_rl_gym_data](https://huggingface.co/wty-yy/go2_rl_gym_data) |

The `*.pt` files do not load in `training/play_policy.py`; that script loads SB3 `.zip` files. Use the stack named in the `Use with` column for each `.pt` file.

```bash
# Expected failure in this repo: .pt is not an SB3 checkpoint
python3 training/play_policy.py \
  --model training/pretrained/go2_locomotion/flat_model_6800.pt \
  --no-display --episodes 1 --max-steps 10
```

Current local result: `ERROR: this script loads Stable-Baselines3 .zip checkpoints only.`

Official [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) currently ships `deploy/pre_train/{g1,h1,h1_2}/motion.pt` only, with no Go2 `motion.pt`.

## Stack commands

| Stack | URL | Notes |
|-------|-----|-------|
| Genesis sim2real stairs | https://github.com/saifahmadgit/go2-sim2real-deploy | `stairs.pt` / `walk.pt` |
| CTS MoE stairs | https://github.com/wty-yy/go2_rl_gym | HF checkpoints, MuJoCo stairs.xml |
| Isaac Lab parkour | https://github.com/CAI23sbP/Isaaclab_Parkour | Google Drive teacher/student |
| Extreme Parkour | https://github.com/chengxuxin/extreme-parkour | Isaac Gym |
| Blind stairs Newton | https://github.com/NMadhub/go2-blind-stairs-newton | Train yourself |

Papers: [docs/papers/README.md](papers/README.md). Worlds: [docs/DEMOS.md](DEMOS.md).
