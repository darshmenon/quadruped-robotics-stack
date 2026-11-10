#!/usr/bin/env python3
"""Build go2_office_scene.xml: the trained Go2 physics (go2_scene.xml) wearing the official
Unitree Go2 meshes from mujoco_menagerie, inside a furnished office.

Physics is untouched (same masses, inertias, gains, sensors, foot names) so every trained policy
still runs; only visuals change: the simplified capsule/box geoms are made transparent and the
menagerie mesh geoms are attached to the same bodies. The arm is hidden (still simulated, stowed).

  python3 training/envs/make_office_scene.py
"""
import copy, os, xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.environ.get("GO2_MENAGERIE", "/home/asimov/mujoco_menagerie/unitree_go2/go2.xml")
import sys
def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
PLAIN = "--plain" in sys.argv
OUT = os.path.join(HERE, _arg("--out", "go2_plain_scene.xml" if PLAIN else "go2_office_scene.xml"))

BASE = _arg("--base", "go2_scene.xml")      # e.g. --base go2_steps_h05.xml --plain --out go2_steps_h05_unitree.xml
base = ET.parse(os.path.join(HERE, BASE)).getroot()
men = ET.parse(MENAGERIE).getroot()

base.find("compiler").set("meshdir", "assets_go2")

# ---- assets: menagerie meshes + materials, office textures ----
asset = ET.SubElement(base, "asset")
for m in men.find("asset"):
    e = copy.deepcopy(m)
    if e.tag == "material":
        e.set("name", "go2_" + e.get("name"))
    asset.append(e)
ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.75 0.82 0.9", rgb2="0.35 0.4 0.5", width="512", height="512")
ET.SubElement(asset, "texture", name="floortex", type="2d", builtin="checker", rgb1="0.55 0.42 0.30", rgb2="0.50 0.38 0.27", width="256", height="256", mark="edge", markrgb="0.42 0.32 0.22")
ET.SubElement(asset, "material", name="floor_wood", texture="floortex", texrepeat="14 10", texuniform="true", reflectance="0.12")
ET.SubElement(asset, "texture", name="carpettex", type="2d", builtin="flat", rgb1="0.2 0.3 0.45", width="64", height="64")
for n, c in dict(wall="0.92 0.91 0.88 1", wood="0.45 0.30 0.18 1", deskTop="0.72 0.58 0.40 1", metalgrey="0.55 0.57 0.6 1",
                 screen="0.05 0.08 0.12 1", chair="0.15 0.15 0.18 1", plant="0.15 0.45 0.2 1", pot="0.6 0.3 0.2 1",
                 carpet="0.2 0.3 0.45 1", glass="0.6 0.8 0.9 0.35", red="0.75 0.15 0.12 1", paper="0.95 0.95 0.9 1").items():
    ET.SubElement(asset, "material", name=n, rgba=c)

# ---- visuals on the robot bodies ----
def bodies(root):
    return {b.get("name"): b for b in root.iter("body")}
rb, mb = bodies(base), bodies(men)
for b in rb.values():                       # hide the simplified geoms but keep their collisions
    for g in b.findall("geom"):
        g.set("rgba", "0 0 0 0")
for name, mbody in mb.items():
    if name not in rb:
        continue
    for g in mbody.findall("geom"):
        if g.get("mesh") is None:
            continue
        e = ET.Element("geom", type="mesh", mesh=g.get("mesh"), material="go2_" + g.get("material"),
                       contype="0", conaffinity="0", group="2", density="0")
        for k in ("pos", "quat"):
            if g.get(k):
                e.set(k, g.get(k))
        rb[name].append(e)

# ---- worldbody: office ----
wb = base.find("worldbody")
for g in list(wb.findall("geom")):
    if g.get("name") == "floor":
        wb.remove(g)
for l in wb.findall("light"):
    wb.remove(l)
W = lambda **k: ET.SubElement(wb, "geom", **{kk: str(v) for kk, v in k.items()})
if PLAIN:
    ET.SubElement(asset, "texture", name="gridtex", type="2d", builtin="checker", mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
    ET.SubElement(asset, "material", name="grid", texture="gridtex", texuniform="true", texrepeat="5 5", reflectance="0.2")
    W(name="floor", type="plane", size="100 100 0.1", group="1", material="grid", condim="3", friction="1.0 0.005 0.0001")
    ET.SubElement(wb, "light", pos="0 0 4", dir="0 0 -1", directional="true", diffuse="0.8 0.8 0.8")
    ET.indent(base); ET.ElementTree(base).write(OUT, encoding="unicode"); print("wrote", OUT); sys.exit(0)
W(name="floor", type="plane", size="20 20 0.1", material="floor_wood", condim="3", friction="1.0 0.005 0.0001")
X0, X1, Y0, Y1, H = -2.0, 9.0, -3.5, 3.5, 2.8
for i, (px, py) in enumerate([(1.5, 0), (4.5, 0), (7.5, 0)]):
    ET.SubElement(wb, "light", pos=f"{px} {py} 2.7", dir="0 0 -1", diffuse="0.9 0.9 0.85", specular="0.1 0.1 0.1", castshadow="false")
ET.SubElement(wb, "light", pos="3.5 0 6", dir="0 0 -1", directional="true", diffuse="0.35 0.35 0.35", castshadow="true")
# walls (south/north solid, west solid, east with a glass partition)
cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
W(name="wall_s", type="box", pos=f"{cx} {Y0} {H/2}", size=f"{(X1-X0)/2} 0.05 {H/2}", material="wall")
W(name="wall_n", type="box", pos=f"{cx} {Y1} {H/2}", size=f"{(X1-X0)/2} 0.05 {H/2}", material="wall")
W(name="wall_w", type="box", pos=f"{X0} {cy} {H/2}", size=f"0.05 {(Y1-Y0)/2} {H/2}", material="wall")
W(name="wall_e", type="box", pos=f"{X1} {cy} {H/2}", size=f"0.05 {(Y1-Y0)/2} {H/2}", material="glass")
# carpet strip down the middle corridor (visual only)
W(type="box", pos=f"{cx} 0 0.002", size=f"{(X1-X0)/2-0.3} 0.6 0.002", material="carpet", contype=0, conaffinity=0)

def desk(x, y, side):                       # side=+1 faces +y wall
    W(type="box", pos=f"{x} {y} 0.72", size="0.8 0.38 0.025", material="deskTop")
    for dx in (-0.74, 0.74):
        for dy in (-0.32, 0.32):
            W(type="box", pos=f"{x+dx} {y+dy} 0.35", size="0.025 0.025 0.35", material="metalgrey")
    W(type="box", pos=f"{x} {y+side*0.18} 0.98", size="0.26 0.02 0.17", material="screen")   # monitor
    W(type="box", pos=f"{x} {y+side*0.18} 0.78", size="0.04 0.04 0.03", material="metalgrey")
    W(type="box", pos=f"{x} {y-side*0.1} 0.748", size="0.18 0.06 0.004", material="chair")  # keyboard
    W(type="box", pos=f"{x+0.5} {y} 0.76", size="0.1 0.14 0.012", material="paper")
    # chair
    cyy = y - side * 0.75
    W(type="box", pos=f"{x} {cyy} 0.45", size="0.22 0.22 0.03", material="chair")
    W(type="box", pos=f"{x} {cyy-side*0.2} 0.72", size="0.22 0.03 0.25", material="chair")
    W(type="cylinder", pos=f"{x} {cyy} 0.22", size="0.03 0.22", material="metalgrey")
    W(type="cylinder", pos=f"{x} {cyy} 0.015", size="0.25 0.015", material="metalgrey", contype=0, conaffinity=0)

for x in (1.0, 3.2, 5.4, 7.4):
    desk(x, 2.9, +1)                         # north row against the wall
    desk(x, -2.9, -1)                        # south row
# bookshelves on the west wall
for y in (-2.0, 2.0):
    W(type="box", pos=f"{X0+0.25} {y} 1.0", size="0.22 0.5 1.0", material="wood")
    for z in (0.5, 1.0, 1.5):
        W(type="box", pos=f"{X0+0.4} {y} {z}", size="0.05 0.46 0.12", material=("red" if z == 1.0 else "paper"), contype=0, conaffinity=0)
# meeting table with chairs, off the corridor
W(type="cylinder", pos="6.3 1.45 0.74", size="0.55 0.025", material="deskTop")
W(type="cylinder", pos="6.3 1.45 0.37", size="0.05 0.37", material="metalgrey")
for a in range(4):
    import math
    px, py = 6.3 + 0.9 * math.cos(a * math.pi / 2 + 0.4), 1.45 + 0.9 * math.sin(a * math.pi / 2 + 0.4)
    W(type="box", pos=f"{px} {py} 0.45", size="0.2 0.2 0.03", material="chair")
    W(type="cylinder", pos=f"{px} {py} 0.22", size="0.03 0.22", material="metalgrey")
# plants + water cooler + a low cable-cover ramp to drive over
for px, py in ((8.4, 3.0), (8.4, -3.0), (-1.6, 0.0), (4.4, -1.7)):
    W(type="cylinder", pos=f"{px} {py} 0.2", size="0.18 0.2", material="pot")
    W(type="sphere", pos=f"{px} {py} 0.62", size="0.3", material="plant")
W(type="box", pos="8.3 0.0 0.5", size="0.18 0.18 0.5", material="metalgrey")
W(type="box", pos="3.0 0.0 0.012", size="0.03 1.0 0.012", material="red")             # cable cover on the floor

ET.indent(base)
ET.ElementTree(base).write(OUT, encoding="unicode")
print("wrote", OUT)
