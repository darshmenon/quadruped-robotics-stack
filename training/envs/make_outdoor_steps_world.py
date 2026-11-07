#!/usr/bin/env python3
"""Outdoor Gazebo world (trees and rocks from go2_gz_world_outdoor.sdf) with a flight of low steps on the walking line.

    python3 training/envs/make_outdoor_steps_world.py --height 0.03 --out training/envs/go2_gz_world_outdoor_steps.sdf

The robot spawns at the origin facing +X. 4 solid steps of riser --height (tread 0.30 m, 2.4 m wide, same geometry as
make_step_scene.py's MuJoCo scene) start at --stairs-at, followed by a landing. The online Fuel URIs of the outdoor
world do not resolve offline, so props are included from the local ~/.gz/fuel cache. Ground is a green plane with
friction 1.0 (matching the MuJoCo floor) instead of the Fuel Grass Plane, and the world carries the `showcase_cam`
model that scripts/record_go2_rl_gz.py moves with the set_pose service.
"""
import argparse
import os
import re

import make_park_world as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDOOR_PROPS = {"Pine Tree": ("openrobotics", "pine tree"), "Oak tree": ("openrobotics", "oak tree")}  # the cached Ocean Rock model does not load offline, so the rocks are left out


def uri(owner, folder):
    d = os.path.join(os.path.expanduser("~/.gz/fuel/fuel.gazebosim.org"), owner, "models", folder)
    ver = str(max(int(v) for v in os.listdir(d) if v.isdigit()))
    return "file://" + os.path.join(d, ver).replace(" ", "%20")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=float, default=0.03, help="riser height (m)")
    ap.add_argument("--stairs-at", type=float, default=2.0, help="x where the flight begins")
    ap.add_argument("--tread", type=float, default=0.30)
    ap.add_argument("--width", type=float, default=2.4, help="width of the flight (m); a wide flight keeps a drifting robot on it")
    args = ap.parse_args()

    outdoor = open(os.path.join(HERE, "go2_gz_world_outdoor.sdf")).read()
    opened = open(os.path.join(HERE, "go2_gz_world_open.sdf")).read()
    head = opened[:opened.index("    <!-- Open flat ground")]
    head = head.replace("<ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse>",
                        "<ambient>0.36 0.58 0.28 1</ambient><diffuse>0.36 0.58 0.28 1</diffuse>")
    head = head.replace("<background>0.75 0.75 0.78 1</background>", "<background>0.62 0.78 0.92 1</background>")
    cam = opened[opened.index("    <!-- Follow camera"):opened.index("</world>")]

    out = [head, "    <!-- Outdoor props from go2_gz_world_outdoor.sdf (local Fuel cache) and a low-step flight -->\n"]
    for m in re.finditer(r"<include>\s*<uri>[^<]*/models/([^<]+)</uri>\s*<name>([^<]+)</name>\s*<pose>([^<]+)</pose>", outdoor):
        model, name, pose = m.group(1), m.group(2), m.group(3)
        if model not in OUTDOOR_PROPS:  # the Fuel Grass Plane is replaced by the plane above
            continue
        out.append(f'    <include><name>{name}</name><uri>{uri(*OUTDOOR_PROPS[model])}</uri><static>true</static>'
                   f'<pose>{pose}</pose></include>\n')

    h, d, x0, W = args.height, args.tread, args.stairs_at, args.width
    for i in range(1, 5):
        top = h * i
        out.append(mp.box(f"step_{i}", x0 + d * (i - 0.5), 0, top / 2, d, W, top, (0.80, 0.78, 0.72)))
        out.append(mp.box(f"edge_{i}", x0 + d * (i - 1) + 0.02, 0, top + 0.001, 0.04, W, 0.004, (0.25, 0.25, 0.28), collide=False))
    out.append(mp.box("landing", x0 + 4 * d + 3.0, 0, h * 4 / 2, 6.0, W, h * 4, (0.74, 0.72, 0.68)))
    out.append(cam)
    out.append("</world>\n</sdf>\n")
    open(args.out, "w").write("".join(out))
    print(f"wrote {args.out}: 4 steps of {h:.3f} m at x={x0}")


if __name__ == "__main__":
    main()
