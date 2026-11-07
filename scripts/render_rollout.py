"""Roll a PPO checkpoint out in MuJoCo and render a follow-cam GIF + frame strip.

    GO2_TROT_PRIOR=1 python3 scripts/render_rollout.py \
        --model training/logs/mujoco_curriculum/best_walk_trotprior/go2_flat_walk.zip \
        --cmd 0.3 --seconds 15 --out docs/images/go2_walk_demo

Runs as two processes (--_render is the child): importing SB3 before mujoco segfaults and
mujoco before SB3 hangs under OSMesa, so the rollout (SB3) and the render (OSMesa) can't
share one interpreter. Writes <out>.gif and <out>_strip.png.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.modules.setdefault("tensorflow", None)  # SB3 -> tensorboard imports all of TensorFlow (minutes on a loaded box); unused here

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "training"))


def rollout(model, cmd, steps, scene, out, wz=0.0, tries=1):
    import analyze_rollout as ar  # imports play_policy (mujoco env) before SB3
    if scene == "stairs" and os.environ.get("GO2_STAIRS_VISION", "0") == "1":
        # height-scan policies were trained under domain randomisation + curriculum; the strict eval env is much harder for them
        import train_curriculum as tc
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        env = DummyVecEnv([tc._make_stage_env("stairs", (cmd, 0.0, wz), 0, int.from_bytes(os.urandom(2), "little"), 0.02, 5, True, True)])
        env = VecNormalize.load(ar._find_vecnorm(model), env)
        env.training, env.norm_reward = False, False
        core = env.venv.envs[0].unwrapped
        ppo = PPO.load(model, env=env)
    else:
        env, core, ppo = ar._load(model, scene, cmd, wz)
    best = None
    for t in range(tries):  # keep the attempt that gets furthest along +x (the policy is stochastic across resets)
        obs = env.reset()
        q = [core.data.qpos.copy()]
        for _ in range(steps):
            a, _ = ppo.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(a)
            if done[0]:  # DummyVecEnv auto-resets on done; stop before the reset pose is recorded
                break
            q.append(core.data.qpos.copy())
        fwd = q[-1][0] - q[0][0]
        if tries > 1:
            print(f"try {t + 1}/{tries}: {len(q)} steps, forward {fwd:+.2f} m")
        if best is None or fwd > best[0]:
            best = (fwd, q)
    q = best[1]
    np.save(out, np.array(q))
    print(f"rollout: {len(q)} steps, forward {q[-1][0] - q[0][0]:+.2f} m, min z {min(x[2] for x in q):.2f}")


def render(qpath, out, every, ctrl_dt, cam_mode, hide_arm=False, scene="flat"):
    import mujoco
    from PIL import Image
    from envs.go2_mujoco_env import Go2MujocoEnv
    q = np.load(qpath)
    if scene == "stairs":  # draws the step scene (GO2_STAIRS_SCENE or the default course); the walk comes from the saved qpos
        from envs.go2_mujoco_stairs_env import Go2MujocoStairsEnv
        core = Go2MujocoStairsEnv(render_mode=None, legs_only=True, randomize_domain=False, use_curriculum=False, use_vision=False)
    else:
        core = Go2MujocoEnv(render_mode=None, legs_only=True, randomize_domain=False, use_curriculum=False)
    m, d = core.model, core.data
    if hide_arm:  # visual only: the arm bodies (arm_base .. right_finger) stay in the physics
        first = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_base")
        m.geom_rgba[m.geom_bodyid >= first, 3] = 0.0
    r = mujoco.Renderer(m, 360, 480)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 1.6, -38, 140  # steep enough to keep the horizon (glitchy under OSMesa) out of frame
    if cam_mode == "side":  # fixed side-on camera framing the whole path, so forward travel is visible
        a, b = q[0][:2], q[-1][:2]
        cam.lookat[:] = [*(0.5 * (a + b)), 0.2]
        cam.distance, cam.elevation, cam.azimuth = max(2.5, 1.1 * float(np.linalg.norm(b - a))), -12, 90
    if cam_mode == "profile":
        cam.distance, cam.elevation, cam.azimuth = 1.5, -16, 70
    imgs = []
    for i in range(0, len(q), every):
        d.qpos[:] = q[i][: m.nq]
        mujoco.mj_forward(m, d)
        if cam_mode == "follow":
            cam.lookat[:] = d.xpos[bid]
        elif cam_mode == "profile":  # low side-on view that tracks the robot: step heights read in profile
            cam.lookat[:] = d.xpos[bid] + np.array([0.35, 0.0, -0.05])
        r.update_scene(d, camera=cam)
        imgs.append(Image.fromarray(r.render()))
    imgs[0].save(out + ".gif", save_all=True, append_images=imgs[1:], loop=0,
                 duration=int(1000 * ctrl_dt * every))
    pick = [imgs[i] for i in np.linspace(0, len(imgs) - 1, min(6, len(imgs))).astype(int)]
    strip = Image.new("RGB", (480 * len(pick), 360))
    for i, im in enumerate(pick):
        strip.paste(im, (480 * i, 0))
    strip.save(out + "_strip.png")
    print(f"render: {len(imgs)} frames -> {out}.gif, {out}_strip.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--scene", default="flat", choices=["flat", "stairs", "rough"])
    ap.add_argument("--cmd", type=float, default=0.3)
    ap.add_argument("--wz", type=float, default=0.0, help="yaw-rate command (rad/s)")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--every", type=int, default=2, help="render every Nth control step (50 Hz)")
    ap.add_argument("--no-arm", action="store_true", help="hide the stowed arm in the render (physics unchanged)")
    ap.add_argument("--tries", type=int, default=1, help="roll out N times and render the best (furthest +x) attempt")
    ap.add_argument("--cam", default="side", choices=["side", "follow", "profile"])
    ap.add_argument("--save-q", default=None, help="also keep the rolled-out qpos (.npy) so other angles can be rendered from it")
    ap.add_argument("--load-q", default=None, help="skip the rollout and render this saved qpos")
    ap.add_argument("--out", default="go2_rollout")
    ap.add_argument("--_render", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._render:  # child: render only
        render(args._render, args.out, args.every, 0.02, args.cam, args.no_arm, args.scene)
        return
    steps = int(args.seconds / 0.02)
    with tempfile.TemporaryDirectory() as tmp:
        qpath = args.load_q or os.path.join(tmp, "q.npy")
        if not args.load_q:
            rollout(args.model, args.cmd, steps, args.scene, qpath, args.wz, args.tries)
            if args.save_q:
                import shutil
                shutil.copy(qpath, args.save_q)
        env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
        subprocess.run([sys.executable, os.path.abspath(__file__), "--_render", qpath, "--out",
                        args.out, "--every", str(args.every), "--cam", args.cam, "--scene", args.scene] + (["--no-arm"] if args.no_arm else []), check=True, env=env)


if __name__ == "__main__":
    main()
