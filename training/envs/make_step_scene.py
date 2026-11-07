#!/usr/bin/env python3
"""Write a stairs scene with a chosen riser height, for the climbing curriculum (0.05 m -> 0.10 -> 0.15 -> 0.20).

    python3 training/envs/make_step_scene.py --height 0.10 --out training/envs/go2_steps_h10.xml
    GO2_STAIRS_SCENE=training/envs/go2_steps_h10.xml GO2_TROT_PRIOR=1 python3 training/train_curriculum.py --stages stairs ...

Replaces the fixed easy/mid/hard course of go2_stairs_scene.xml with --steps solid steps of the given riser height
(--depth m tread, full width), starting at --start m along +X and ending in a long top platform, so a policy
that climbs all of them keeps walking on the top. Each step is a box from the floor up to its own height.
"""
import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = ('    <geom name="{n}" type="box" pos="{x:.4f} 0.0000 {z:.4f}" size="{hx:.4f} 0.8000 {z:.4f}" '
        'rgba="0.55 0.55 0.6 1" condim="3" friction="1.0 0.005 0.0001" group="1"/>\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=float, required=True, help="riser height per step (m)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--depth", type=float, default=0.30, help="tread depth (m)")
    ap.add_argument("--start", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = open(os.path.join(HERE, "go2_stairs_scene.xml")).read()
    a = src.index("    <!-- Stair / ledge course")
    b = src.index("    <!-- GO2 floating base")
    course = "    <!-- Steps: %d x %.2f m riser, %.2f m tread, top platform -->\n" % (args.steps, args.height, args.depth)
    for i in range(1, args.steps + 1):
        h = args.height * i
        course += GEOM.format(n=f"step_{i}", x=args.start + args.depth * (i - 0.5), z=h / 2, hx=args.depth / 2).replace(
            f'size="{args.depth / 2:.4f} 0.8000 {h / 2:.4f}"', f'size="{args.depth / 2:.4f} 0.8000 {h / 2:.4f}"')
    top_h = args.height * args.steps
    course += GEOM.format(n="top", x=args.start + args.depth * args.steps + 3.0, z=top_h / 2, hx=3.0)
    open(args.out, "w").write(src[:a] + course + "\n" + src[b:])
    print(f"wrote {args.out}: {args.steps} steps of {args.height:.2f} m, top at z={top_h:.2f} m")


if __name__ == "__main__":
    main()
