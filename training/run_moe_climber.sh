#!/bin/bash
# Height-scan climber with the mixture-of-experts actor, stability + edge-foothold rewards (3 cm then 5 cm).
#   bash training/run_moe_climber.sh [init.zip]   # default: resume the saved 300k-step 3 cm checkpoint
cd "$(dirname "$0")"
export GO2_STD_FLOOR=0.45 GO2_ENT_COEF=0.005 GO2_TROT_PRIOR=1 GO2_WIDE_CMD=1 GO2_RAND_GAINS=1 GO2_RAND_GYRO=1 GO2_KP_MAX=2 GO2_GYRO_MAX=1.6 GO2_RAND_LAT=1
export GO2_STAIRS_VISION=1 GO2_MOE=3 GO2_STABILITY_W=0.02 GO2_FOOTHOLD_W=0.5 GO2_NO_STAND_SUBSIDY=1
init=${1:-$PWD/logs/mujoco_curriculum/moe_h03_resume/go2_stairs_300000_steps.zip}
mkdir -p logs/run_logs
for stage in "03 700000 60 2.0" "05 1500000 120 2.0"; do
  set -- $stage; h=$1; steps=$2
  [ -f envs/go2_steps_h$h.xml ] || python3 envs/make_step_scene.py --height 0.$h --out envs/go2_steps_h$h.xml
  [ -d logs/mujoco_curriculum/stairs ] && mv logs/mujoco_curriculum/stairs logs/mujoco_curriculum/stairs_before_moe2_h$h
  GO2_STALL_PENALTY=$3 GO2_LIFT_SCALE=$4 GO2_STAIRS_SCENE=$PWD/envs/go2_steps_h$h.xml nice -n 19 /usr/bin/python3 train_curriculum.py --stages stairs \
    --timesteps $steps --n_envs 4 --asymmetric --obs-history 5 --gait ${init:+--init-from "$init"} > logs/run_logs/train_moe2_h$h.log 2>&1 || exit 1
  init=$PWD/logs/mujoco_curriculum/stairs/go2_stairs_final.zip
  mkdir -p logs/mujoco_curriculum/moe2_h$h && cp -r logs/mujoco_curriculum/stairs/. logs/mujoco_curriculum/moe2_h$h/
done
