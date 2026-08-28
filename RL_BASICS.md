  # RL Basics: How Training and Inference Work Here

  A short conceptual guide to the reinforcement learning pipeline in this repo — what's actually happening, not just which commands to type. For the full command reference see [README.md § RL Policy Training](README.md#rl-policy-training).

  ## The idea

  A neural network (the **policy**) looks at the robot's state 50 times a second and outputs joint targets for all 12 legs. Nobody hand-writes a gait. The policy starts as random noise, and PPO (Proximal Policy Optimization) nudges its weights after every batch of simulated steps so that action sequences which score a higher **reward** become more likely. After a few million steps, "walk forward without falling" is the only strategy left that scores well, so that's what the network converges to.

  Everything runs in **MuJoCo**, not real hardware or Gazebo — physics steps are fast and cheap, so training can simulate months of robot-time in a few hours on one GPU.

  ## The three pieces

  **1. Environment** (`training/envs/go2_mujoco_env.py`) — a Gym-style env wrapping the Go2 MuJoCo model. Each `step()`:
  - Applies the policy's 12 joint-position deltas as an action.
  - Advances physics.
  - Builds the next **observation**: base angular velocity, gravity direction (tells the policy which way is up), the commanded velocity `(vx, vy, yaw_rate)`, joint positions/velocities, the previous action, per-foot contact state, and end-effector position/target (for the arm+gripper).
  - Computes a **reward** — one scalar per step that's a weighted sum of terms: track the commanded velocity, don't bounce vertically, hold a target body height, stay level, minimize joint torque and jerk, keep feet in contact, avoid foot slip, don't collide non-foot links with the world, stay off joint limits, plus an alive bonus. See `_compute_reward()` for the exact weights — they're tuned by hand and occasionally get exploited (see [CHANGELOG](docs/CHANGELOG.md) for two examples: a policy that learned to freeze in a crouch to farm the alive bonus, and one that learned to fake "reaching" while standing still — both fixed by gating reward terms off during those exploit conditions).
  - Ends the episode on a fall, or after a fixed number of steps.

  **2. Algorithm** — Stable-Baselines3's `PPO`. It runs many parallel environment copies (`SubprocVecEnv`, `--n_envs`), collects a batch of transitions from all of them, and does a few epochs of gradient descent on a clipped objective that keeps each policy update small and stable. `VecNormalize` keeps a running mean/std of observations and rewards so the network sees roughly unit-scale inputs — this is why a `.zip` checkpoint and its matching `vecnorm_*.pkl` have to be loaded together at play time (see `training/logs/mujoco/`).

  **3. Training loop** (`training/train_mujoco.py`) — wires the two together, adds:
  - **Domain randomization**: friction, mass, pushes, etc. vary per-episode so the policy doesn't overfit to one exact physics instance.
  - **Curriculum**: terrain/task difficulty ramps up as the policy improves (`train_curriculum.py`: flat → rough → stairs).
  - Checkpointing every N steps + a `best_model.zip` tracked by held-out eval reward, so training is resumable (`--resume`) and you can always fall back to the best checkpoint rather than the last one.
  - TensorBoard logging of the reward, and of each individual reward *component* separately (`RewardComponentCallback`) — critical for noticing an exploit early, since total reward going up doesn't mean the robot is doing what you wanted.

  ## Training

  ```bash
  pip install -r requirements.txt
  ./scripts/train_policy.sh mujoco --timesteps 2000000 --n_envs 8
  tensorboard --logdir training/logs/mujoco   # watch reward curves live
  ```
  Resume from a checkpoint:
  ```bash
  ./scripts/train_policy.sh mujoco --resume training/logs/mujoco/checkpoints/go2_mujoco_500000_steps.zip
  ```
  What to watch for: `ep_rew_mean` climbing is necessary but not sufficient — check the per-component curves and actually watch the policy (below) before trusting a "good" number. A reward curve that jumps unusually high, fast is the first sign of an exploit, not a breakthrough.

  ## Using (running) a trained policy

  Inference is just a forward pass through the frozen network — no more gradient updates, no reward computation needed:
  ```bash
  python3 training/play_policy.py --model training/logs/mujoco/best_model.zip \
    --vecnorm training/logs/mujoco/vecnorm_final.pkl --cmd 0.5 0 0
  ```
  This loads the `.zip` (network weights) + `.pkl` (normalization stats), steps the same MuJoCo env with the policy's argmax/mean action each tick, and renders it. `--no-display --episodes N` runs it headless and prints mean reward / distance traveled / fall rate — the fastest way to sanity-check a checkpoint after training or after any reward-function change.

  ## Mental model, end to end

  ```
  random weights --(millions of PPO updates, each nudged by reward)--> policy that walks
                            ↑
          reward = "did the thing I want, not something that gamed the score"
  ```
  The entire skill of the repo's RL work is in that reward function and in noticing, from the component curves plus actually watching play_policy.py, when the policy found a shortcut instead of the intended behavior.
