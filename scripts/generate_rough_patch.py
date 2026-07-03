"""Print the SDF <model> block for the rough-terrain patch.

Grid of small boxes with smoothly-varying heights (numpy box-blur, no
external deps) standing in for a heightmap. Plain box geometry is used
instead of <heightmap> because Ogre2's heightmap-terrain shader fails to
compile under this environment's software/EGL rendering fallback when a
GPU lidar sensor is also present in the scene.

Usage:
    python3 scripts/generate_rough_patch.py > /tmp/rough_patch.sdf
"""

import numpy as np

GRID = 7
PATCH_SIZE = 4.0  # meters, matches the 14-18m x-zone / -2..2m y-zone
CENTER = (16.0, 0.0)
MIN_H, MAX_H = 0.02, 0.07
SEED = 42


def _box_blur(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    return (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0


def main():
    rng = np.random.default_rng(SEED)
    field = rng.uniform(0.0, 1.0, size=(GRID, GRID))
    field = _box_blur(field)
    field -= field.min()
    field /= field.max()
    heights = MIN_H + field * (MAX_H - MIN_H)

    step = PATCH_SIZE / GRID
    x0 = CENTER[0] - PATCH_SIZE / 2
    y0 = CENTER[1] - PATCH_SIZE / 2

    print('    <model name="rough_patch">')
    print('      <static>true</static>')
    print('      <pose>0 0 0 0 0 0</pose>')
    for i in range(GRID):
        for j in range(GRID):
            h = heights[i, j]
            x = x0 + step * (i + 0.5)
            y = y0 + step * (j + 0.5)
            z = h / 2
            link = f"bump_{i}_{j}"
            print(f'      <link name="{link}"><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>')
            print(f'        <collision name="collision"><geometry><box><size>{step:.3f} {step:.3f} {h:.3f}</size></box></geometry>')
            print('          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>')
            print(f'        <visual name="visual"><geometry><box><size>{step:.3f} {step:.3f} {h:.3f}</size></box></geometry>')
            print('          <material><ambient>0.65 0.6 0.5 1</ambient><diffuse>0.65 0.6 0.5 1</diffuse></material></visual></link>')
    print('    </model>')


if __name__ == "__main__":
    main()
