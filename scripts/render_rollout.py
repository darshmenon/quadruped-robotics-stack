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

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "training"))


def rollout(model, cmd, steps, scene, out):
    import analyze_rollout as ar  # imports play_policy (mujoco env) before SB3
    env, core, ppo = ar._load(model, scene, cmd)
    obs = env.reset()
    q = [core.data.qpos.copy()]
    for _ in range(steps):
        a, _ = ppo.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(a)
        if done[0]:  # DummyVecEnv auto-resets on done; stop before the reset pose is recorded
            break
        q.append(core.data.qpos.copy())
    np.save(out, np.array(q))
    print(f"rollout: {len(q)} steps, forward {q[-1][0] - q[0][0]:+.2f} m, min z {min(x[2] for x in q):.2f}")


def render(qpath, out, every, ctrl_dt, cam_mode):
    import mujoco
    from PIL import Image
    from envs.go2_mujoco_env import Go2MujocoEnv
    q = np.load(qpath)
    core = Go2MujocoEnv(render_mode=None, legs_only=True, randomize_domain=False, use_curriculum=False)
    m, d = core.model, core.data
    r = mujoco.Renderer(m, 360, 480)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 1.5, -25, 140
    if cam_mode == "side":  # fixed side-on camera framing the whole path, so forward travel is visible
        a, b = q[0][:2], q[-1][:2]
        cam.lookat[:] = [*(0.5 * (a + b)), 0.2]
        cam.distance, cam.elevation, cam.azimuth = max(2.5, 1.1 * float(np.linalg.norm(b - a))), -12, 90
    imgs = []
    for i in range(0, len(q), every):
        d.qpos[:] = q[i][: m.nq]
        mujoco.mj_forward(m, d)
        if cam_mode == "follow":
            cam.lookat[:] = d.xpos[bid]
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
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--every", type=int, default=2, help="render every Nth control step (50 Hz)")
    ap.add_argument("--cam", default="side", choices=["side", "follow"])
    ap.add_argument("--out", default="go2_rollout")
    ap.add_argument("--_render", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._render:  # child: render only
        render(args._render, args.out, args.every, 0.02, args.cam)
        return
    steps = int(args.seconds / 0.02)
    with tempfile.TemporaryDirectory() as tmp:
        qpath = os.path.join(tmp, "q.npy")
        rollout(args.model, args.cmd, steps, args.scene, qpath)
        env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
        subprocess.run([sys.executable, os.path.abspath(__file__), "--_render", qpath, "--out",
                        args.out, "--every", str(args.every), "--cam", args.cam], check=True, env=env)


if __name__ == "__main__":
    main()
