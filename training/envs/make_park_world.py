#!/usr/bin/env python3
"""Generate a tidy "park" Gazebo world for walking / stair-climbing recordings.

    python3 training/envs/make_park_world.py --out training/envs/go2_gz_world_park.sdf
    python3 training/envs/make_park_world.py --stairs 0.05 --out training/envs/go2_gz_world_park_stairs.sdf

Layout (robot spawns at the origin facing +X): a paved lane 2.4 m wide and 40 m long on mown grass, kerbs and
lane dashes, tree rows and cones/barriers/benches along the sides (Fuel models already in ~/.gz/fuel, so no
download), and a raised stepped viewing terrace to the left of the lane that is purely scenery. With --stairs H the
lane also crosses a flight of 4 solid steps of riser H m (light treads, dark risers' edge strips so each step reads
clearly on camera) followed by a landing; without it the lane stays flat.

Includes the `showcase_cam` follow-camera model that scripts/record_go2_rl_gz.py moves with the set_pose service.
Physics/plugins/scene are copied from go2_gz_world_open.sdf so launch files treat it like the other worlds.
"""
import argparse
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/.gz/fuel/fuel.gazebosim.org/openrobotics/models")
# Gazebo's online Fuel resolver fails offline, so props are included straight from the local Fuel cache.
PROPS = {"Oak tree": "oak tree", "Pine Tree": "pine tree", "Construction Cone": "construction%20cone",
         "Jersey Barrier": "jersey barrier", "Bench": "foodcourtbenchlong"}


def prop_uri(model):
    d = os.path.join(CACHE, PROPS[model])
    ver = str(max(int(v) for v in os.listdir(d) if v.isdigit()))
    return "file://" + os.path.join(d, ver).replace(" ", "%20")


def box(name, x, y, z, sx, sy, sz, rgb, collide=True, yaw=0.0):
    r, g, b = rgb
    col = (f'<collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
           '<surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>') if collide else ""
    return (f'    <model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {z:.4f} 0 0 {yaw:.3f}</pose><link name="l">{col}'
            f'<visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
            f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material></visual></link></model>\n')


def include(name, model, x, y, yaw=0.0, scale=None):
    sc = f"<scale>{scale} {scale} {scale}</scale>" if scale else ""
    return (f'    <include><name>{name}</name><uri>{prop_uri(model)}</uri><static>true</static>'
            f'<pose>{x:.2f} {y:.2f} 0 0 0 {yaw:.3f}</pose>{sc}</include>\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stairs", type=float, default=0.0, help="riser height (m) of a 4-step flight on the lane; 0 = flat")
    ap.add_argument("--length", type=float, default=40.0)
    ap.add_argument("--stairs-at", type=float, default=6.0, help="x where the flight begins")
    ap.add_argument("--tread", type=float, default=0.30)
    ap.add_argument("--no-props", action="store_true", help="skip Fuel includes (primitive scenery only)")
    args = ap.parse_args()
    rng = random.Random(7)

    src = open(os.path.join(HERE, "go2_gz_world_open.sdf")).read()
    head = src[:src.index("    <!-- Open flat ground")]
    cam = src[src.index("    <!-- Follow camera"):src.index("</world>")]
    head = head.replace('<background>0.75 0.75 0.78 1</background>', '<background>0.62 0.78 0.92 1</background>')
    # the open world's grey ground is replaced by a green one drawn below; keep its collision plane (mu 1.0)
    head = head.replace("<ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse>",
                        "<ambient>0.36 0.58 0.28 1</ambient><diffuse>0.36 0.58 0.28 1</diffuse>")

    L, W = args.length, 2.4
    out = [head, "    <!-- Park: paved lane, kerbs, dashes, trees, props, terrace -->\n"]
    out.append(box("lane", L / 2 - 4, 0, 0.004, L + 8, W, 0.008, (0.72, 0.70, 0.66), collide=False))
    for s in (-1, 1):
        out.append(box(f"kerb_{s}", L / 2 - 4, s * (W / 2 + 0.07), 0.015, L + 8, 0.14, 0.03, (0.88, 0.88, 0.86), collide=False))
    for k, x in enumerate(range(-4, int(L), 2)):
        out.append(box(f"dash_{k}", x, 0, 0.0095, 0.7, 0.07, 0.003, (0.98, 0.98, 0.98), collide=False))

    if args.stairs > 0:
        h, d, x0 = args.stairs, args.tread, args.stairs_at
        for i in range(1, 5):
            top = h * i
            cx = x0 + d * (i - 0.5)
            out.append(box(f"step_{i}", cx, 0, top / 2, d, W, top, (0.80, 0.78, 0.72)))            # solid block, light tread
            out.append(box(f"edge_{i}", x0 + d * (i - 1) + 0.02, 0, top + 0.001, 0.04, W, 0.004, (0.25, 0.25, 0.28), collide=False))
        out.append(box("landing", x0 + 4 * d + 3.0, 0, h * 4 / 2, 6.0, W, h * 4, (0.74, 0.72, 0.68)))
        out.append(box("landing_edge", x0 + 4 * d + 0.02, 0, h * 4 + 0.001, 0.04, W, 0.004, (0.25, 0.25, 0.28), collide=False))

    # raised viewing terrace on the left: three broad visible steps (scenery, off the lane)
    for i in range(3):
        top = 0.12 * (i + 1)
        out.append(box(f"terrace_{i}", 16, 6.0 + 1.2 * i, top / 2, 10 - 1.5 * i, 1.2, top,
                       (0.80 - 0.05 * i, 0.77 - 0.05 * i, 0.70 - 0.05 * i)))
    for i in range(6):
        out.append(box(f"post_{i}", 11.5 + 1.8 * i, -3.2, 0.45, 0.12, 0.12, 0.9, (0.45, 0.30, 0.18), collide=False))
    out.append(box("rail", 16.4, -3.2, 0.82, 9.4, 0.06, 0.06, (0.55, 0.38, 0.22), collide=False))

    if not args.no_props:
        for i, x in enumerate(range(-2, int(L) + 6, 6)):
            out.append(include(f"tree_n{i}", "Oak tree" if i % 2 else "Pine Tree", x + rng.uniform(-1, 1), 5.5 + rng.uniform(-0.6, 0.6)))
            out.append(include(f"tree_s{i}", "Pine Tree" if i % 2 else "Oak tree", x + 3 + rng.uniform(-1, 1), -5.5 + rng.uniform(-0.6, 0.6)))
        for i, x in enumerate((9, 15, 21, 27)):
            out.append(include(f"cone_{i}", "Construction Cone", x, 1.55 if i % 2 else -1.55))
        out.append(include("barrier_a", "Jersey Barrier", 24, 2.6, 1.5708))
        out.append(include("barrier_b", "Jersey Barrier", 24, -2.6, 1.5708))
        out.append(include("bench_a", "Bench", 12.0, -2.4, 0.0))
        out.append(include("bench_b", "Bench", 30.0, 2.4, 3.1416))

    out.append(cam)
    out.append("</world>\n</sdf>\n")
    open(args.out, "w").write("".join(out))
    print(f"wrote {args.out} ({'stairs %.2f m' % args.stairs if args.stairs else 'flat lane'}, "
          f"{'no props' if args.no_props else 'with Fuel props'})")


if __name__ == "__main__":
    main()
