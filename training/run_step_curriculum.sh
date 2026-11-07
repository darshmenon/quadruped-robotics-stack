#!/bin/bash
# Climbing curriculum: stairs stage on steps of 0.03 m -> 0.05 m -> 0.10 m -> 0.15 m -> 0.20 m, each resumed from the previous.
#   bash training/run_step_curriculum.sh pretrained/go2_walk_lat2/go2_flat_walk.zip 600000
# Needs the same env vars as the walker run; one stage at a time (shared machine). Logs: logs/run_logs/train_steps_hNN.log
cd "$(dirname "$0")"
init=$1; steps=${2:-600000}
export GO2_STD_FLOOR=${GO2_STD_FLOOR:-0.45} GO2_ENT_COEF=${GO2_ENT_COEF:-0.005}
export GO2_TROT_PRIOR=1 GO2_WIDE_CMD=1 GO2_RAND_GAINS=1 GO2_RAND_GYRO=1 GO2_KP_MAX=2 GO2_GYRO_MAX=1.6 GO2_RAND_LAT=1
# prior calf-lift scale per stage (peak foot clearance at zero residual: 2.0 -> 7.8 cm, 3.0 -> 12.2 cm, saturates ~13 cm)
declare -A LIFT=([03]=2.0 [05]=2.0 [10]=3.0 [15]=3.5 [20]=4.0)
for h in 03 05 10 15 20; do
  [ -f envs/go2_steps_h$h.xml ] || python3 envs/make_step_scene.py --height 0.$h --out envs/go2_steps_h$h.xml
  [ -d logs/mujoco_curriculum/stairs ] && mv logs/mujoco_curriculum/stairs logs/mujoco_curriculum/stairs_before_h$h
  GO2_LIFT_SCALE=${LIFT[$h]} GO2_STAIRS_SCENE=$PWD/envs/go2_steps_h$h.xml /usr/bin/python3 train_curriculum.py --stages stairs --timesteps $steps \
    --n_envs 8 --asymmetric --obs-history 5 --gait --init-from "$init" > logs/run_logs/train_steps_h$h.log 2>&1
  init=$PWD/logs/mujoco_curriculum/stairs/go2_stairs_final.zip
  mkdir -p logs/mujoco_curriculum/steps_h$h && cp -r logs/mujoco_curriculum/stairs/. logs/mujoco_curriculum/steps_h$h/
done
