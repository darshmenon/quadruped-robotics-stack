"""MuJoCo-based Gymnasium environment for Unitree Go2 locomotion training."""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

SCENE_XML = os.path.join(os.path.dirname(__file__), "go2_scene.xml")

# intelligence/ lives at the repo root, two levels up from training/envs/ --
# not on sys.path by default when this module is imported via training/'s
# own sys.path.insert (see train_mujoco.py). Pure-Python (math + dataclasses
# only), so safe to import here without ROS sourced.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from intelligence.manipulation.arm_ik import inverse_kinematics, JOINT_LIMIT

# inverse_kinematics(x, y, z) defaults to wrist_pitch=0 (last link horizontal),
# which leaves much of the sampled reach-target workspace unreachable: holding
# the wrist horizontal often forces an elbow bend past its own +/-90deg limit
# even when the target is well within the arm's overall reach envelope
# (confirmed empirically -- e.g. a plain 0.35m-forward target already fails
# at wrist_pitch=0). This task only cares about fingertip position, not
# end-effector orientation, so Go2MujocoEnv._best_ik_candidate sweeps
# wrist_pitch and keeps the best-scoring solution rather than assuming 0
# works (see that method for the collision/margin ranking).
_WRIST_PITCH_SWEEP = sorted(np.linspace(-JOINT_LIMIT, JOINT_LIMIT, 13), key=abs)

# Arm+gripper stow pose, matches scripts/make_go2_stand.py STANDING_POSE /
# intelligence/manipulation/arm_reach_node.py STOW_POSE (gripper closed).
ARM_STOW = [0.0, 1.4, 0.8, 0.3, 0.0, 0.0, 0.0]  # base, lower_arm, upper_arm, wrist1, wrist2, L/R finger

# qpos/qvel order follows go2_scene.xml's body tree (depth-first): the 4 legs
# leg-by-leg, then the arm+gripper chain appended last (see that file's
# comment on why it must stay last).
DEFAULT_QPOS = np.array([
    0.1,   # FL_hip
    0.8,   # FL_thigh
    -1.5,  # FL_calf
    -0.1,  # FR_hip
    0.8,   # FR_thigh
    -1.5,  # FR_calf
    0.1,   # RL_hip
    1.0,   # RL_thigh
    -1.5,  # RL_calf
    -0.1,  # RR_hip
    1.0,   # RR_thigh
    -1.5,  # RR_calf
] + ARM_STOW, dtype=np.float32)

# Actuator order (independent of qpos order -- set by <actuator> declaration
# order in go2_scene.xml): [FL_hip, FR_hip, RL_hip, RR_hip,
#                            FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#                            FL_calf, FR_calf, RL_calf, RR_calf,
#                            base, lower_arm, upper_arm, wrist1, wrist2, L/R finger]
# The arm+gripper segment happens to match DEFAULT_QPOS's order too, since
# (unlike the 4 parallel legs) it's a single serial chain declared once.
ACT_DEFAULT = np.array([
    0.1, -0.1,  0.1, -0.1,
    0.8,  0.8,  1.0,  1.0,
   -1.5, -1.5, -1.5, -1.5,
] + ARM_STOW, dtype=np.float32)

# 3 ang_vel + 3 gravity + 3 cmd + 19 dof_pos + 19 dof_vel + 19 prev_action
# + 4 contacts + 3 ee_pos + 3 reach_target
OBS_DIM = 76
# When gait_conditioned=True, append: 5 gait params + 4 foot clock sins
# (freq, phase, offset, bound, footswing_height, clock_FL/FR/RL/RR)
GAIT_OBS_DIM = 9
ACT_DIM = 19
ACT_SCALE = 0.25

EPISODE_LEN_S = 20.0
SIM_DT = 0.005
CTRL_DECIMATION = 4   # policy at 50 Hz, sim at 200 Hz
CTRL_DT = SIM_DT * CTRL_DECIMATION

# Named gait presets: (phase, offset, bound). Duration is fixed at 0.5
# (half the cycle in stance). Trotting is the default curriculum sample.
GAIT_PRESETS = {
    "trotting": (0.5, 0.0, 0.0),
    "bounding": (0.0, 0.0, 0.5),
    "pacing":   (0.0, 0.5, 0.0),
    "pronking": (0.0, 0.0, 0.0),
}
GAIT_FREQ_RANGE = (1.5, 3.5)       # Hz
GAIT_FOOTSWING_RANGE = (0.04, 0.18)  # metres, command only (clearance cue)
GAIT_CONTACT_WEIGHT = 0.25
GAIT_CONTACT_KAPPA = 0.08          # soft-step width for desired contact

# Step-quality terms (air-time encourages real strides; slip penalizes
# sliding while planted). Only score air-time when a nontrivial velocity
# command is active so standing still is not rewarded for "long stance".
AIR_TIME_TARGET_S = 0.25
AIR_TIME_WEIGHT = 0.35
FEET_SLIP_WEIGHT = 0.12
CONTACT_FORCE_THRESH = 0.3  # after /50 clip, ~15 N raw touch

TARGET_HEIGHT = 0.27  # nominal base height above ground while standing

ALIVE_BONUS  = 0.3    # per-step credit for still standing, so ending an
                       # episode early is never a shortcut to avoid penalties
FALL_PENALTY = -8.0    # one-time hit applied on the step that trips termination

# arm_base's pos= in go2_scene.xml, i.e. where the arm mounts relative to the
# "base" body -- reach targets are sampled around this point, in the same
# frame as the ee_pos sensor.
ARM_MOUNT_POS = np.array([0.08, 0.0, 0.057], dtype=np.float32)

# REACH_MIN_RADIUS used to be 0.12 with REACH_MAX_RADIUS_EASY at 0.22 -- empirically
# (via arm_ik.py's IK across the full yaw/pitch sampling range below) 0% of targets in
# that band are actually reachable at ANY wrist orientation: holding the fingertip that
# close to the mount requires an elbow bend past the joint's own +/-90deg limit
# regardless of wrist_pitch. The "easy" end of the curriculum was asking for targets no
# controller could ever hit. Reachability climbs sharply from there (~12% at 0.22,
# ~72% at 0.25, ~100% by 0.28), so the floor needs to sit past that knee.
REACH_MIN_RADIUS = 0.28   # closest a target is ever placed from the mount point
REACH_MAX_RADIUS_EASY = 0.32  # max radius at curriculum_level=0
REACH_MAX_RADIUS_HARD = 0.42  # max radius at curriculum_level=1 (< arm_ik.py's
                               # ~0.51 structural max, so targets stay solvable
                               # across the yaw/elevation range sampled below)
# _sample_target_and_baseline still rejection-samples against the real (collision-aware)
# IK solver, not just this radius heuristic -- reachability and self-collision both
# depend on yaw/pitch too, not radius alone.
REACH_SUCCESS_DIST = 0.05  # fingertip-to-target distance counted as "reached"
REACH_SIGMA = 0.12         # width of the reach-distance reward kernel
# Matched to r_lin's scale (2.0) below, not left as a minor auxiliary term --
# a reach reward smaller than the dominant locomotion term gets washed out by
# PPO's advantage estimates even once REACH_DENSE_WEIGHT gives it a
# non-vanishing gradient (see aCodeDog/legged-robots-manipulation's Go2-ARX
# config, which sets its equivalent term equal to tracking_lin_vel).
REACH_WEIGHT = 2.0

# The ARM_STOW pose (see DEFAULT_QPOS) tucks the arm down and back so it
# doesn't interfere with walking -- its fingertip sits ~0.4m from
# ARM_MOUNT_POS, roughly opposite the direction targets are sampled in, so a
# typical starting reach_dist is ~0.5-0.7m. REACH_SIGMA=0.12 alone gives
# exp(-(d/sigma)^2) and its gradient both underflow to ~0 anywhere past
# ~0.3m -- confirmed by hand FK on the stow pose matching the observed
# frozen fingertip exactly, and by every recorded "best" checkpoint having
# reach reward ~1e-8 the whole episode. The policy was never once rewarded
# for moving the arm toward the target; this dense term gives a real
# (non-vanishing) gradient across the actual starting-distance range, with
# the sigma kernel above still providing the fine-precision signal near the
# goal.
REACH_DENSE_WEIGHT = 0.3
REACH_DENSE_NORM = 0.8    # roughly the max plausible stow-to-target distance

# Actuator indices for the 5 arm joints (see actuator-order comment above);
# these get an IK-baseline target instead of ACT_DEFAULT+action*ACT_SCALE.
ARM_ACT_SLICE = slice(12, 17)
# arm_ik.py's analytic IK (already used by the scripted Gazebo pick demo,
# intelligence/manipulation/pick_demo.py) solves for the exact joint angles
# that place the fingertip at reach_target, so the policy no longer has to
# discover inverse kinematics from the distance reward alone -- it only
# needs to learn a small residual correction (gravity sag, PD tracking
# error, whole-body coordination while walking). Residual is comparable to
# ACT_SCALE rather than tiny, so the policy can still override the IK
# baseline where useful (e.g. briefly retracting the arm for balance).
ARM_RESIDUAL_SCALE = 0.3

# Bridges the gap between REACH_DENSE (weak linear pull, active everywhere)
# and REACH_WEIGHT's exp kernel (only meaningful inside ~0.3m) -- nothing
# previously gave extra incentive to actually cross that boundary, so the
# dense term could plateau without ever pulling the fingertip close enough
# for the precision kernel to take over (observed: reach_dense ~0.07-0.09,
# flat, over 700k+ steps; reach ~0 throughout). Modeled on FR-Net's staged
# reward gating (coarse precondition unlocks a following-stage reward rather
# than everything summing unconditionally from the start).
REACH_MID_RADIUS = 0.25
REACH_MID_WEIGHT = 0.4

# Two training runs both saw eval reward peak (~460) with curriculum_level
# around 0.8-0.85 (max cmd speed ~1.0 m/s, reach radius ~0.37m) and then
# degrade as the episode-length-only success signal kept pushing curriculum_
# level toward 1.0 -- by ~0.99, reach reward had collapsed to ~0 (arm no
# longer tracking targets at all) and eval reward had dropped to ~140. The
# success metric (did the episode survive) doesn't require the walk+reach
# task to still be going well, so it kept escalating difficulty past the
# point the policy could actually hold both. Capping below 1.0 keeps
# training in the region both runs demonstrated actually works.
MAX_CURRICULUM_LEVEL = 0.85
REACH_SUCCESS_BONUS = 0.5

# Mid-episode push DR (get-up-isaaclab / unitree_rl_gym _push_robots) — not
# previously wired into the Go2 MuJoCo walk env (only mass/friction/Kp DR).
PUSH_INTERVAL_S = 3.0
PUSH_VEL_XY = 0.55          # m/s impulse magnitude cap
COLLISION_WEIGHT = 0.2      # non-foot geom vs floor / self
SOFT_LIMIT_WEIGHT = 2.0     # soft DoF-limit proximity (unitree_rl_gym style)
SOFT_LIMIT_MARGIN = 0.05    # rad inside hard range before penalty starts
GAIT_SYMMETRY_WEIGHT = 0.15 # LocoTouch-style diagonal pair agreement (--gait)


class Go2MujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cmd=(0.5, 0.0, 0.0), render_mode=None,
                 randomize_domain=True, use_curriculum=True,
                 initial_curriculum_level=0.0, reach_target=None,
                 gait_conditioned=False, gait_name="trotting",
                 push_robots=True, legs_only=False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        self.cmd = np.array(cmd, dtype=np.float32)
        self.render_mode = render_mode
        self.randomize_domain = randomize_domain
        self.use_curriculum = use_curriculum
        self.gait_conditioned = bool(gait_conditioned)
        self.push_robots = bool(push_robots)
        # legs_only holds the arm+gripper fixed at ARM_STOW and drops it from
        # the action/observation space entirely, giving a 12-action/45-obs
        # interface that matches Go2GazeboEnv exactly (same feature order and
        # scaling in _get_obs) -- so a policy trained this way is the only
        # kind that can be loaded straight into eval_gazebo.py without a
        # shape mismatch. The normal (arm-enabled) env stays ACT_DIM=19/
        # OBS_DIM=76 and is untouched by this flag.
        self.legs_only = bool(legs_only)
        self._renderer = None
        self._prev_action = np.zeros(12 if self.legs_only else ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._steps_since_push = 0
        self._max_steps = int(EPISODE_LEN_S / (SIM_DT * CTRL_DECIMATION))
        self._last_episode_steps = self._max_steps
        self._stall_steps = 0

        self.curriculum_level = float(initial_curriculum_level)
        # Mirrors `cmd`: a fixed reach target for non-curriculum use (e.g.
        # play_policy.py), resampled every reset when use_curriculum=True
        # (see reset()). Default is a modest forward reach, not a random
        # sample, so eval runs are reproducible unless the caller asks
        # otherwise. 0.28m specifically (not e.g. 0.3) because the IK
        # baseline's destination pose being collision-free doesn't guarantee
        # the straight-line joint-space path from ARM_STOW to it is too --
        # 0.3m forward settles >0.5m off target (verified: the destination
        # pose is valid, but the arm gets stuck on the body en route from
        # stow); 0.28m verified clean (<3cm) across seeds with zero residual
        # action (see _best_ik_candidate/_sample_target_and_baseline for the
        # collision-aware IK selection this depends on).
        self.reach_target = (
            np.array(reach_target, dtype=np.float32) if reach_target is not None
            else ARM_MOUNT_POS + np.array([0.28, 0.0, 0.0], dtype=np.float32)
        )
        # Recomputed from self.reach_target at the top of every reset()
        # (see _compute_arm_ik_baseline); this default is only live if
        # step() were ever called before reset(), which gym doesn't do.
        self._arm_ik_baseline = np.array(ARM_STOW[:5], dtype=np.float32)

        # Gait command: [freq_hz, phase, offset, bound, footswing_m].
        # Clocks and desired contacts are derived each step from these.
        self._gait_index = 0.0
        self._clock_inputs = np.zeros(4, dtype=np.float32)
        self._desired_contact = np.ones(4, dtype=np.float32)
        self.gait_cmd = self._default_gait_cmd(gait_name)

        # cache original model params for domain randomization
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._base_mass = float(self.model.body_mass[self._base_body_id])
        self._base_floor_friction = self.model.geom_friction[self._floor_geom_id].copy()
        self._base_gainprm = self.model.actuator_gainprm[:, 0].copy()
        self._base_biasprm1 = self.model.actuator_biasprm[:, 1].copy()

        # Foot bodies for air-time / slip rewards (FL, FR, RL, RR).
        self._foot_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        ], dtype=np.int32)
        self._foot_geom_ids = set()
        for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            if gid >= 0:
                self._foot_geom_ids.add(int(gid))
        # The closed gripper's two fingers always slightly interpenetrate by
        # design (that's what "closed" means) -- exclude their mutual contact
        # from arm-placement collision checks (_best_ik_candidate) so it isn't
        # mistaken for the arm colliding with the leg/torso. The finger geoms
        # themselves are unnamed in go2_scene.xml (only their parent bodies
        # are), so look them up by body id rather than mj_name2id(GEOM, ...).
        self._finger_geom_ids = set()
        for n in ("left_finger", "right_finger"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid >= 0:
                self._finger_geom_ids.update(
                    int(g) for g in range(self.model.ngeom)
                    if self.model.geom_bodyid[g] == bid)
        # Freejoint has no limits; hinge joints for legs+arm start at jnt 1.
        self._hinge_qposadr = []
        self._hinge_range = []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            adr = int(self.model.jnt_qposadr[j])
            lo, hi = self.model.jnt_range[j]
            if hi > lo:
                self._hinge_qposadr.append(adr)
                self._hinge_range.append((float(lo), float(hi)))
        self._feet_air_time = np.zeros(4, dtype=np.float32)
        self._last_contacts = np.zeros(4, dtype=bool)
        self._foot_vel_buf = np.zeros(6, dtype=np.float64)
        self._last_push = np.zeros(2, dtype=np.float32)

        if self.legs_only:
            obs_dim = 45  # 3 ang_vel + 3 gravity + 3 cmd + 12 dof_pos + 12 dof_vel + 12 prev_action
        else:
            obs_dim = OBS_DIM + (GAIT_OBS_DIM if self.gait_conditioned else 0)
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12 if self.legs_only else ACT_DIM,), dtype=np.float32)
        self._act_default = ACT_DEFAULT.copy()

    # ------------------------------------------------------------------ #

    def _gravity_vec(self) -> np.ndarray:
        w, x, y, z = self.data.sensor("orientation").data.astype(np.float32)
        return np.array([
            2 * (-z * x - w * y),
            -2 * (z * y - w * x),
            1 - 2 * (w * w + z * z),
        ], dtype=np.float32)

    def _get_contacts(self) -> np.ndarray:
        raw = np.array(
            [self.data.sensor(n).data[0]
             for n in ("FL_contact", "FR_contact", "RL_contact", "RR_contact")],
            dtype=np.float32)
        return np.clip(raw / 50.0, 0.0, 1.0)

    def _foot_lin_vels(self) -> np.ndarray:
        """World-frame linear velocity of each foot body, shape (4, 3)."""
        out = np.empty((4, 3), dtype=np.float32)
        for i, bid in enumerate(self._foot_body_ids):
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                int(bid), self._foot_vel_buf, 0)
            out[i] = self._foot_vel_buf[:3]
        return out

    def _air_time_and_slip(self, contacts: np.ndarray):
        """Update air-time clocks; return (air_time_reward, slip_penalty)."""
        in_contact = contacts > CONTACT_FORCE_THRESH
        contact_filt = np.logical_or(in_contact, self._last_contacts)
        first_contact = (self._feet_air_time > 0.0) & contact_filt

        self._feet_air_time += CTRL_DT
        air = 0.0
        cmd_speed = float(np.hypot(self.cmd[0], self.cmd[1]))
        if cmd_speed > 0.1 and np.any(first_contact):
            # Reward strides that stay aloft near AIR_TIME_TARGET_S.
            air = AIR_TIME_WEIGHT * float(np.sum(
                (self._feet_air_time - AIR_TIME_TARGET_S) * first_contact))

        self._feet_air_time = np.where(contact_filt, 0.0, self._feet_air_time)
        self._last_contacts = in_contact

        # Penalize horizontal foot velocity while planted.
        foot_vel = self._foot_lin_vels()
        slip_speeds = np.linalg.norm(foot_vel[:, :2], axis=1)
        slip = -FEET_SLIP_WEIGHT * float(np.sum(slip_speeds * in_contact))
        return air, slip

    def _default_gait_cmd(self, gait_name: str) -> np.ndarray:
        phase, offset, bound = GAIT_PRESETS.get(gait_name, GAIT_PRESETS["trotting"])
        freq = 0.5 * (GAIT_FREQ_RANGE[0] + GAIT_FREQ_RANGE[1])
        swing = 0.5 * (GAIT_FOOTSWING_RANGE[0] + GAIT_FOOTSWING_RANGE[1])
        return np.array([freq, phase, offset, bound, swing], dtype=np.float32)

    def _sample_gait_cmd(self) -> np.ndarray:
        """Sample a gait style + frequency. Early curriculum stays on
        trotting; higher levels mix in bounding/pacing/pronking."""
        rng = self.np_random
        if self.curriculum_level < 0.35:
            name = "trotting"
        else:
            names = list(GAIT_PRESETS.keys())
            # Bias toward trotting even late in curriculum.
            weights = [0.55, 0.2, 0.15, 0.1]
            name = names[int(rng.choice(len(names), p=weights))]
        phase, offset, bound = GAIT_PRESETS[name]
        freq_lo, freq_hi = GAIT_FREQ_RANGE
        freq = float(rng.uniform(freq_lo, freq_lo + (freq_hi - freq_lo) * max(
            0.4, self.curriculum_level)))
        swing_lo, swing_hi = GAIT_FOOTSWING_RANGE
        swing = float(rng.uniform(swing_lo, swing_lo + (swing_hi - swing_lo) * max(
            0.3, self.curriculum_level)))
        return np.array([freq, phase, offset, bound, swing], dtype=np.float32)

    def _step_gait(self) -> None:
        """Advance gait phase and refresh clock / desired-contact targets."""
        if not self.gait_conditioned:
            return
        freq, phase, offset, bound, _swing = self.gait_cmd
        duration = 0.5
        self._gait_index = (self._gait_index + CTRL_DT * float(freq)) % 1.0
        # Per-foot phase offsets: FL, FR, RL, RR
        foot_phases = np.array([
            self._gait_index + phase + offset + bound,
            self._gait_index + offset,
            self._gait_index + bound,
            self._gait_index + phase,
        ], dtype=np.float64) % 1.0

        clocks = np.empty(4, dtype=np.float32)
        desired = np.empty(4, dtype=np.float32)
        kappa = GAIT_CONTACT_KAPPA
        for i, p in enumerate(foot_phases):
            # Warp so stance occupies [0, 0.5) and swing [0.5, 1).
            if p < duration:
                warped = p * (0.5 / duration)
            else:
                warped = 0.5 + (p - duration) * (0.5 / (1.0 - duration))
            clocks[i] = np.sin(2.0 * np.pi * warped)
            # Soft stance indicator: high in first half of warped cycle.
            desired[i] = 1.0 / (1.0 + np.exp((warped - 0.5) / kappa))
        self._clock_inputs = clocks
        self._desired_contact = desired

    def _get_obs(self) -> np.ndarray:
        d = self.data
        ang_vel   = d.sensor("ang_vel").data.astype(np.float32) * 0.25
        gravity   = self._gravity_vec()
        cmd_scaled = self.cmd * np.array([2.0, 2.0, 0.25], dtype=np.float32)
        if self.legs_only:
            dof_pos = (d.qpos[7:19].astype(np.float32) - DEFAULT_QPOS[:12])
            dof_vel = d.qvel[6:18].astype(np.float32) * 0.05
            return np.concatenate([
                ang_vel, gravity, cmd_scaled, dof_pos, dof_vel, self._prev_action,
            ])
        dof_pos   = (d.qpos[7:].astype(np.float32) - DEFAULT_QPOS)
        dof_vel   = d.qvel[6:].astype(np.float32) * 0.05
        contacts  = self._get_contacts()
        # Gripper fingertip position and its current target, both relative
        # to the base body (see go2_scene.xml's ee_pos sensor comment) --
        # given as two absolute points rather than a precomputed error
        # vector, matching how dof_pos/cmd are split into achieved vs.
        # desired elsewhere in this observation.
        ee_pos    = d.sensor("ee_pos").data.astype(np.float32)
        parts = [
            ang_vel, gravity, cmd_scaled, dof_pos, dof_vel, self._prev_action,
            contacts, ee_pos, self.reach_target,
        ]
        if self.gait_conditioned:
            # Normalize freq into roughly [-1, 1]-ish range for the MLP.
            freq_n = (self.gait_cmd[0] - GAIT_FREQ_RANGE[0]) / (
                GAIT_FREQ_RANGE[1] - GAIT_FREQ_RANGE[0] + 1e-6)
            swing_n = self.gait_cmd[4] / GAIT_FOOTSWING_RANGE[1]
            gait_feat = np.array(
                [freq_n, self.gait_cmd[1], self.gait_cmd[2], self.gait_cmd[3],
                 swing_n],
                dtype=np.float32)
            parts.extend([gait_feat, self._clock_inputs])
        return np.concatenate(parts)

    def _compute_reward(self, action: np.ndarray):
        d = self.data
        lin_vel  = d.sensor("lin_vel").data.astype(np.float32)
        ang_vel  = d.sensor("ang_vel").data.astype(np.float32)
        gravity  = self._gravity_vec()

        r_lin    = 2.0 * float(np.exp(
            -((lin_vel[0] - self.cmd[0])**2 + (lin_vel[1] - self.cmd[1])**2) / 0.1))
        r_ang    = 0.5 * float(np.exp(-((ang_vel[2] - self.cmd[2])**2) / 0.25))
        r_z      = -2.0  * float(lin_vel[2]**2)
        r_height = -1.0  * (float(d.qpos[2]) - TARGET_HEIGHT)**2
        r_orient = -0.5  * float(gravity[0]**2 + gravity[1]**2)
        r_torque = -2e-4 * float(np.sum(d.actuator_force**2))
        r_smooth = -5e-3 * float(np.sum((action - self._prev_action)**2))

        contacts  = self._get_contacts()
        r_contact = 0.15 * min(float(np.sum(contacts > 0.3)) / 2.0, 1.0)

        # Explicit stall penalty: standing still while a real command is
        # active must never out-earn walking, no matter how forgiving the
        # tracking kernel above is (previously the policy converged to
        # standing still — see README "Known issue").
        cmd_speed = float(np.hypot(self.cmd[0], self.cmd[1]))
        actual_speed = float(np.hypot(lin_vel[0], lin_vel[1]))
        is_stalling = cmd_speed > 0.15 and actual_speed < 0.3 * cmd_speed
        r_stall = -0.6 if is_stalling else 0.0

        # ALIVE_BONUS was meant to stop early termination from ever being a
        # shortcut to dodge penalties, but it's unconditional -- a policy
        # that freezes at a safe crouch (never falls, never moves) still
        # earns ALIVE_BONUS every step, which alone funds riding out the
        # -0.6 stall penalty for a full 1000-step episode rather than
        # actually attempting to walk (confirmed: a converged checkpoint
        # settled into z~0.16m, 0.00m displacement, steady ~-0.8 reward/step
        # for 1000/1000 steps -- a new instance of the same "reward-gate the
        # exploited term" pattern reach_gate above was added for). Gating it
        # off during a stall keeps the original protection for a genuinely
        # complying robot while removing the subsidy for frozen-in-place
        # stalling specifically.
        r_alive = 0.0 if is_stalling else ALIVE_BONUS

        r_air, r_slip = self._air_time_and_slip(contacts)
        r_col = -COLLISION_WEIGHT * float(self._nonfoot_collision_count())
        r_lim = -SOFT_LIMIT_WEIGHT * self._soft_dof_limit_penalty()

        components = dict(
            lin=r_lin, ang=r_ang, vz=r_z, height=r_height,
            orient=r_orient, torque=r_torque, smooth=r_smooth, contact=r_contact,
            stall=r_stall, alive=r_alive,
            air_time=r_air, slip=r_slip,
            collision=r_col, soft_limit=r_lim,
        )

        # legs_only holds the arm perfectly stowed, so ee_pos never moves --
        # the reach terms would just be a fixed, meaningless constant (or
        # noise from the still-resampled reach_target) rather than a real
        # task signal, so drop them from the sum entirely instead of scoring
        # an arm that physically cannot respond.
        if not self.legs_only:
            ee_pos = d.sensor("ee_pos").data.astype(np.float32)
            reach_dist = float(np.linalg.norm(self.reach_target - ee_pos))
            # Once REACH_WEIGHT was raised to match locomotion's scale, standing
            # still and just reaching (eating the -0.6 stall penalty) became more
            # profitable than actually walking-while-reaching -- confirmed by
            # play_policy.py eval on trained checkpoints showing dist=0.00m,
            # mean_vx~=0.00 despite reach/reach_dense/reach_mid all firing and
            # ep_rew_mean far above any historical walking-only peak. Gating all
            # reach terms off during a stall closes that exploit at the source
            # instead of trying to out-tune the stall penalty's magnitude.
            reach_gate = 0.0 if is_stalling else 1.0
            r_reach = reach_gate * REACH_WEIGHT * float(np.exp(-(reach_dist ** 2) / (REACH_SIGMA ** 2)))
            r_reach_dense = reach_gate * REACH_DENSE_WEIGHT * max(0.0, 1.0 - reach_dist / REACH_DENSE_NORM)
            r_reach_mid = reach_gate * (REACH_MID_WEIGHT
                           if REACH_SUCCESS_DIST <= reach_dist < REACH_MID_RADIUS else 0.0)
            r_reach_bonus = reach_gate * (REACH_SUCCESS_BONUS if reach_dist < REACH_SUCCESS_DIST else 0.0)
            components.update(
                reach=r_reach, reach_dense=r_reach_dense, reach_mid=r_reach_mid,
                reach_bonus=r_reach_bonus,
            )

        if self.gait_conditioned:
            # Match measured contacts to the commanded gait's stance/swing.
            # contacts are already in [0,1]; desired_contact is soft [0,1].
            match = 1.0 - np.abs(contacts - self._desired_contact)
            components["gait_contact"] = GAIT_CONTACT_WEIGHT * float(np.mean(match))
            # Diagonal pair symmetry (FL↔RR, FR↔RL) — LocoTouch idea, no tactile.
            diag = 0.5 * (
                1.0 - abs(float(contacts[0] - contacts[3]))
                + 1.0 - abs(float(contacts[1] - contacts[2])))
            components["gait_symmetry"] = GAIT_SYMMETRY_WEIGHT * float(diag)

        return float(sum(components.values())), components

    def _sample_cmd(self) -> np.ndarray:
        max_vx = 0.15 + 1.05 * self.curriculum_level
        vx = float(self.np_random.uniform(-0.1, max_vx))
        vy = float(self.np_random.uniform(-0.2, 0.2)) * self.curriculum_level
        wz = float(self.np_random.uniform(-0.5, 0.5)) * self.curriculum_level
        return np.array([vx, vy, wz], dtype=np.float32)

    def _best_ik_candidate(self, offset: np.ndarray):
        """Best wrist_pitch candidate (base/shoulder/elbow/wrist1/wrist2)
        placing the fingertip at ARM_MOUNT_POS + offset, ranked collision-
        free first then by joint-limit margin, plus whether any candidate
        was collision-free at all.

        arm_ik.py solves pure arm-chain geometry with no awareness of the
        rest of the robot -- some wrist_pitch solutions swing the arm back
        into the legs/torso, and the resulting contact forces then stop the
        PD controller from ever reaching that commanded angle (observed:
        settled fingertip off by >0.5m despite an exact, joint-limit-legal
        IK solution). Must be called with self.data already posed at this
        episode's actual reset stance (legs, base height/orientation) --
        the collision probe checks candidates against the real body
        configuration, not whatever pose the previous episode ended in.
        Temporarily overwrites the arm qpos slice per candidate; caller is
        responsible for restoring it (and re-running mj_forward) once done.
        """
        best, best_score, found_collision_free = None, None, False
        for wrist_pitch in _WRIST_PITCH_SWEEP:
            pose = inverse_kinematics(
                float(offset[0]), float(offset[1]), float(offset[2]),
                wrist_pitch=wrist_pitch)
            if pose is None:
                continue
            candidate = np.array(pose.as_list(), dtype=np.float32)
            self.data.qpos[19:24] = candidate
            mujoco.mj_forward(self.model, self.data)
            collision_free = self._arm_placement_collision_count() == 0
            found_collision_free = found_collision_free or collision_free
            margin = min(JOINT_LIMIT - abs(a) for a in candidate)
            score = (collision_free, margin)
            if best_score is None or score > best_score:
                best, best_score = candidate, score
        return best, found_collision_free

    def _sample_target_and_baseline(self):
        """Sample a reach target (around ARM_MOUNT_POS, in the base body
        frame) together with its IK baseline, rejecting candidates that
        are only reachable via a self-colliding arm pose -- otherwise a
        fraction of episodes would get an impossible target no policy
        could ever satisfy, silently reintroducing the zero-gradient
        problem the IK baseline is meant to fix (see _best_ik_candidate).
        Radius range widens with curriculum_level (same level that drives
        _sample_cmd's walking speed), so training starts with the arm
        holding a near, easy point while standing close to still, and only
        asks for farther reaches once the body is also walking faster --
        "standing and reaching" progressing to "reaching while walking",
        both gated by the one curriculum_level rather than two independent
        schedules that could drift out of sync.

        Must run after self.data is posed at this episode's actual reset
        stance (see reset()), since the collision probe needs the real
        leg/base configuration. Leaves self.data's arm qpos restored to
        its pre-call value (the episode still starts stowed).
        """
        max_radius = REACH_MAX_RADIUS_EASY + (
            REACH_MAX_RADIUS_HARD - REACH_MAX_RADIUS_EASY) * self.curriculum_level
        saved_arm_qpos = self.data.qpos[19:24].copy()
        baseline, offset = None, None
        for _ in range(20):
            radius = float(self.np_random.uniform(REACH_MIN_RADIUS, max_radius))
            # yaw within base_joint's own +/-90deg limit (arm.urdf.xacro), with
            # margin so the shoulder/elbow aren't also pinned at their limits
            # trying to hit the same point; elevation modestly above/below the
            # mount plane.
            yaw = float(self.np_random.uniform(-1.1, 1.1))
            pitch = float(self.np_random.uniform(-0.5, 0.6))
            direction = np.array([
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ], dtype=np.float32)
            candidate_offset = radius * direction
            candidate_baseline, collision_free = self._best_ik_candidate(candidate_offset)
            if collision_free:
                baseline, offset = candidate_baseline, candidate_offset
                break
        self.data.qpos[19:24] = saved_arm_qpos
        mujoco.mj_forward(self.model, self.data)
        if baseline is None:
            # Exhausted retries (rare -- most of the sampled cone has a
            # collision-free solution). Straight ahead at the min radius
            # is always solvable and collision-free (points away from the
            # body, not back into it).
            offset = np.array([REACH_MIN_RADIUS, 0.0, 0.0], dtype=np.float32)
            baseline = np.array(ARM_STOW[:5], dtype=np.float32)
        return ARM_MOUNT_POS + offset, baseline

    def _compute_arm_ik_baseline(self) -> np.ndarray:
        """IK baseline for a reach_target set externally (e.g. play_policy.py
        passing a fixed target), rather than sampled by this env. Since the
        target wasn't vetted by _sample_target_and_baseline's rejection
        loop, falls back to the stow pose if every candidate collides or is
        unreachable -- callers setting reach_target directly are on their
        own for picking something sane.
        """
        saved_arm_qpos = self.data.qpos[19:24].copy()
        offset = self.reach_target - ARM_MOUNT_POS
        best, _ = self._best_ik_candidate(offset)
        self.data.qpos[19:24] = saved_arm_qpos
        mujoco.mj_forward(self.model, self.data)
        if best is None:
            return np.array(ARM_STOW[:5], dtype=np.float32)
        return best

    def _apply_domain_rand(self) -> None:
        if not self.randomize_domain:
            return
        rng = self.np_random
        self.model.body_mass[self._base_body_id] = (
            self._base_mass * float(rng.uniform(0.85, 1.15)))
        self.model.geom_friction[self._floor_geom_id] = (
            self._base_floor_friction * float(rng.uniform(0.7, 1.3)))
        kp_scale = float(rng.uniform(0.85, 1.15))
        self.model.actuator_gainprm[:, 0] = self._base_gainprm * kp_scale
        self.model.actuator_biasprm[:, 1] = self._base_biasprm1 * kp_scale

    def _maybe_push(self) -> None:
        """Impulse on base xy velocity mid-episode (simulates bumps / trips)."""
        self._last_push[:] = 0.0
        if not (self.push_robots and self.randomize_domain):
            return
        self._steps_since_push += 1
        if self._steps_since_push * CTRL_DT < PUSH_INTERVAL_S:
            return
        self._steps_since_push = 0
        # Scale push strength up with curriculum so early training stays calm.
        vmax = PUSH_VEL_XY * (0.35 + 0.65 * self.curriculum_level)
        dx = float(self.np_random.uniform(-vmax, vmax))
        dy = float(self.np_random.uniform(-vmax, vmax))
        self.data.qvel[0] += dx
        self.data.qvel[1] += dy
        self._last_push[:] = (dx, dy)

    def _nonfoot_collision_count(self) -> int:
        """Contacts involving non-foot geoms (thigh/calf/base vs floor or self)."""
        n = 0
        floor = int(self._floor_geom_id)
        feet = self._foot_geom_ids
        for i in range(self.data.ncon):
            g1 = int(self.data.contact[i].geom1)
            g2 = int(self.data.contact[i].geom2)
            if g1 == floor or g2 == floor:
                other = g2 if g1 == floor else g1
                if other not in feet:
                    n += 1
            elif g1 not in feet and g2 not in feet:
                n += 1
        return n

    def _arm_placement_collision_count(self) -> int:
        """Like _nonfoot_collision_count, but ignores the gripper fingers'
        own mutual contact -- they're mechanically closed in ARM_STOW and
        most candidate poses, which is a normal closed-gripper self-contact,
        not a sign the arm placement itself collides with the leg/torso.
        Used by _best_ik_candidate; _compute_reward's collision penalty
        intentionally keeps using _nonfoot_collision_count as-is."""
        n = 0
        floor = int(self._floor_geom_id)
        feet = self._foot_geom_ids
        fingers = self._finger_geom_ids
        for i in range(self.data.ncon):
            g1 = int(self.data.contact[i].geom1)
            g2 = int(self.data.contact[i].geom2)
            if g1 in fingers and g2 in fingers:
                continue
            if g1 == floor or g2 == floor:
                other = g2 if g1 == floor else g1
                if other not in feet:
                    n += 1
            elif g1 not in feet and g2 not in feet:
                n += 1
        return n

    def _soft_dof_limit_penalty(self) -> float:
        """Penalize hinge joints within SOFT_LIMIT_MARGIN of hard range."""
        pen = 0.0
        m = SOFT_LIMIT_MARGIN
        q = self.data.qpos
        for adr, (lo, hi) in zip(self._hinge_qposadr, self._hinge_range):
            v = float(q[adr])
            if v < lo + m:
                pen += (lo + m - v) / m
            elif v > hi - m:
                pen += (v - (hi - m)) / m
        return pen

    def _is_terminated(self) -> bool:
        z = float(self.data.qpos[2])
        if z < 0.15 or z > 0.8:
            return True
        # 1 - 2*(w^2 + z_q^2) == 2*(x^2 + y^2) - 1 == -cos(tilt) for a unit
        # quaternion, so the old `> 0.5` threshold only tripped past 120
        # degrees of pitch/roll - permissive enough that the policy could
        # rear up onto its hind legs and sustain a ~70-80 degree "wheelie"
        # for many steps before actually falling. -0.5 corresponds to ~60
        # degrees, a sane cutoff for a quadruped.
        w, x, y, z_q = self.data.sensor("orientation").data
        return bool(1 - 2 * (w * w + z_q * z_q) > -0.5)

    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.use_curriculum:
            # Surviving alone isn't "success" -- a policy that freezes at a
            # safe crouch and never actually complies with cmd also survives
            # the full episode (confirmed: a converged checkpoint sat at
            # z~0.16m, 0.00m displacement, for 1000/1000 steps), which was
            # advancing curriculum_level toward harder commands/pushes for a
            # policy that had never learned to walk at all. Require it
            # wasn't stalling (see is_stalling in _compute_reward) for most
            # of the episode too.
            success = (self._last_episode_steps >= 0.75 * self._max_steps
                       and self._stall_steps < 0.5 * self._last_episode_steps)
            self.curriculum_level = float(np.clip(
                self.curriculum_level + (0.005 if success else -0.002),
                0.0, MAX_CURRICULUM_LEVEL))
            self.cmd = self._sample_cmd()
            if self.gait_conditioned:
                self.gait_cmd = self._sample_gait_cmd()
            # reach_target is sampled below, after the body is posed at its
            # reset stance -- the collision-aware search needs the real
            # leg/base configuration, not whatever pose the previous
            # episode ended in (see _sample_target_and_baseline).

        mujoco.mj_resetData(self.model, self.data)
        self._apply_domain_rand()

        self.data.qpos[2]   = 0.42
        self.data.qpos[3:7] = [1, 0, 0, 0]
        # Domain-randomize the leg joints only: the arm+gripper stow pose
        # (DEFAULT_QPOS[12:]) starts exact, since the same +/-0.05 rad noise
        # scale would swing the finger joints (0.025m full range) past their
        # limits.
        self.data.qpos[7:19] = DEFAULT_QPOS[:12] + (self.np_random.random(12) - 0.5) * 0.1
        self.data.qpos[19:]  = DEFAULT_QPOS[12:]
        self.data.ctrl[:]   = self._act_default
        mujoco.mj_forward(self.model, self.data)

        if not self.legs_only:
            if self.use_curriculum:
                self.reach_target, self._arm_ik_baseline = self._sample_target_and_baseline()
            else:
                self._arm_ik_baseline = self._compute_arm_ik_baseline()

        self._prev_action = np.zeros(12 if self.legs_only else ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._steps_since_push = 0
        self._last_push[:] = 0.0
        self._last_episode_steps = 0
        self._stall_steps = 0
        self._gait_index = 0.0
        self._feet_air_time[:] = 0.0
        self._last_contacts[:] = False
        self._step_gait()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.legs_only:
            ctrl = self._act_default.copy()
            ctrl[:12] = self._act_default[:12] + action * ACT_SCALE
        else:
            ctrl = self._act_default + action * ACT_SCALE
            ctrl[ARM_ACT_SLICE] = (
                self._arm_ik_baseline + action[ARM_ACT_SLICE] * ARM_RESIDUAL_SCALE)
        self.data.ctrl[:] = ctrl
        for _ in range(CTRL_DECIMATION):
            mujoco.mj_step(self.model, self.data)

        self._maybe_push()
        self._step_gait()
        reward, components = self._compute_reward(action)
        if components["stall"] < 0:
            self._stall_steps += 1
        self._prev_action = action.copy()
        self._step_count += 1
        self._last_episode_steps = self._step_count

        obs = self._get_obs()
        terminated = self._is_terminated()
        truncated  = self._step_count >= self._max_steps

        if terminated:
            reward += FALL_PENALTY
            components["fall"] = FALL_PENALTY

        if self.render_mode == "human":
            self.render()

        # Privileged extras for logging / future asymmetric critic (DreamWaQ-lite).
        # Actor obs stays proprio-only deployable; true lin_vel is training-only.
        lin_vel = self.data.sensor("lin_vel").data.astype(np.float32)
        info = {
            "reward_components": components,
            "privileged": {
                "lin_vel": lin_vel.copy(),
                "base_height": float(self.data.qpos[2]),
                "push_xy": self._last_push.copy(),
            },
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = mujoco.viewer.launch_passive(self.model, self.data)
            self._renderer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            if hasattr(self._renderer, "close"):
                self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    for gait in (False, True):
        env = Go2MujocoEnv(render_mode=None, randomize_domain=False,
                           use_curriculum=False, gait_conditioned=gait)
        obs, _ = env.reset(seed=0)
        expected = OBS_DIM + (GAIT_OBS_DIM if gait else 0)
        print(f"gait_conditioned={gait}: obs shape {obs.shape}")
        assert obs.shape == (expected,), f"expected {expected}, got {obs.shape[0]}"
        saw_air = False
        for _ in range(200):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            comps = info["reward_components"]
            assert "air_time" in comps and "slip" in comps
            if gait:
                assert "gait_contact" in comps
            if comps["air_time"] != 0.0:
                saw_air = True
            if term or trunc:
                obs, _ = env.reset()
        print(f"gait_conditioned={gait}: smoke-test passed (air_nonzero={saw_air})")
