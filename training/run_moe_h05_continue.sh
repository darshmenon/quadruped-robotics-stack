#!/bin/bash
# Continue the 5 cm MoE climber from the last finished checkpoint.
#   bash training/run_moe_h05_continue.sh [steps]
cd "$(dirname "$0")"
export GO2_STD_FLOOR=0.45 GO2_ENT_COEF=0.005 GO2_TROT_PRIOR=1 GO2_WIDE_CMD=1 GO2_RAND_GAINS=1 GO2_RAND_GYRO=1 GO2_KP_MAX=2 GO2_GYRO_MAX=1.6 GO2_RAND_LAT=1
export GO2_STAIRS_VISION=1 GO2_MOE=3 GO2_STABILITY_W=0.02 GO2_FOOTHOLD_W=0.5 GO2_NO_STAND_SUBSIDY=1
steps=${1:-1500000}
init=$PWD/logs/mujoco_curriculum/moe2_h05/go2_stairs_final.zip
ts=$(date +%m%d_%H%M)
mkdir -p logs/run_logs
mv logs/mujoco_curriculum/stairs logs/mujoco_curriculum/stairs_before_moe3_h05_$ts
GO2_STALL_PENALTY=120 GO2_LIFT_SCALE=2.0 GO2_STAIRS_SCENE=$PWD/envs/go2_steps_h05.xml nice -n 19 /usr/bin/python3 train_curriculum.py --stages stairs \
  --timesteps $steps --n_envs 4 --asymmetric --obs-history 5 --gait --init-from "$init" > logs/run_logs/train_moe3_h05.log 2>&1 \
  && mkdir -p logs/mujoco_curriculum/moe3_h05 && cp -r logs/mujoco_curriculum/stairs/. logs/mujoco_curriculum/moe3_h05/
