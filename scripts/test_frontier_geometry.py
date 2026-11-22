"""
Checks for frontier_geometry.py (the pure logic behind
frontier_explorer_go2.py). No ROS/Gazebo needed:

    python3 scripts/test_frontier_geometry.py
"""

import math
import sys

import numpy as np

from frontier_geometry import (
    cell_is_clear,
    cluster_frontier_cells,
    find_frontier_goal,
    near_excluded_goal,
    near_moving_obstacle,
    pull_back_goal,
    raycast_blocked,
    score_clusters,
    select_best_candidate,
    snap_to_free_cell,
    wavefront_frontier_cells,
    world_to_cell,
)

FAILURES = []


def check(name, cond):
    status = "OK" if cond else "FAIL"
    if not cond:
        FAILURES.append(name)
    print(f"[{status}] {name}")


RES = 0.1
OX, OY = -1.0, -1.0


def cx(i):
    """World x of cell column i's center -- real robot poses/goals are
    never exactly on a cell boundary, so tests use centers too rather than
    tripping over float noise at a boundary (e.g. (ox + res - ox) / res can
    land a hair under 1.0 and truncate the wrong way)."""
    return OX + (i + 0.5) * RES


def cy(i):
    return OY + (i + 0.5) * RES


# -- world/cell conversion -----------------------------------------------

check("world_to_cell at a cell center", world_to_cell(cx(0), cy(0), OX, OY, RES) == (0, 0))
check("world_to_cell one cell over", world_to_cell(cx(1), cy(0), OX, OY, RES) == (0, 1))


# -- a small hand-built map -----------------------------------------------
# 10x10 grid, all free (0) except a wall of occupied (100) cells splitting
# it in two, a gap in the wall, and an unexplored (-1) region beyond column
# 6 -- enough to exercise wavefront reachability, frontier framing, and
# clustering. The gap means columns 0-4 and column 6 are one connected
# reachable region (via row 8), so the only real frontier boundary is at
# column 6, against the unknown column 7.
W = H = 10
grid = np.zeros((H, W), dtype=np.int16)
grid[:, 5] = 100          # wall down column 5
grid[8, 5] = 0            # gap in the wall at row 8
grid[:, 7:] = -1          # unknown beyond column 7

ROBOT_X, ROBOT_Y = cx(1), cy(1)  # cell (1, 1), well inside the free side


def run_wavefront():
    known = grid >= 0
    free = known & (grid < 40)
    rr, rc = world_to_cell(ROBOT_X, ROBOT_Y, OX, OY, RES)
    snapped = snap_to_free_cell(free, rr, rc, W, H)
    check("robot cell already free (no snap needed)", snapped == (rr, rc))
    return wavefront_frontier_cells(grid, free, rr, rc, W, H, occupied_min=65)


frontier_cells = run_wavefront()
check("frontier found on the reachable side", len(frontier_cells) > 0)
check("frontier cells sit on the col-6/unknown boundary",
      all(c == 6 for _, c in frontier_cells))
check("gap cell (8, 5) is reachable but not itself a frontier",
      (8, 5) not in frontier_cells)

clusters = cluster_frontier_cells(frontier_cells)
check("at least one frontier cluster", len(clusters) > 0)

candidates = score_clusters(clusters, OX, OY, RES, ROBOT_X, ROBOT_Y, min_frontier_size=1)
check("scored candidates non-empty", len(candidates) > 0)
check("all candidate scores are non-negative", all(s >= 0 for _, _, s in candidates))


# -- snap_to_free_cell ------------------------------------------------

occupied_free = np.zeros((5, 5), dtype=bool)
occupied_free[2, 2] = True
check("snap finds the only free cell nearby", snap_to_free_cell(occupied_free, 0, 0, 5, 5) == (2, 2))
check("snap gives up beyond max_radius",
      snap_to_free_cell(occupied_free, 0, 0, 5, 5, max_radius=1) is None)


# -- clearance / pullback -------------------------------------------------

clear_grid = np.zeros((10, 10), dtype=np.int16)
clear_grid[5, 8] = 100  # a lone occupied cell near the far edge

check("cell far from the occupied cell is clear",
      cell_is_clear(clear_grid, 10, 10, RES, OX, OY, cx(1), cy(1),
                     clearance_standoff=0.2, occupied_min=65))
check("cell right next to the occupied cell is not clear",
      not cell_is_clear(clear_grid, 10, 10, RES, OX, OY, cx(8), cy(5),
                         clearance_standoff=0.2, occupied_min=65))

pulled = pull_back_goal(clear_grid, 10, 10, RES, OX, OY,
                         robot_x=cx(1), robot_y=cy(1),
                         centroid=(cx(9), cy(1)),
                         clearance_standoff=0.3, occupied_min=65)
dist_orig = math.hypot(cx(9) - cx(1), 0)
dist_pulled = math.hypot(pulled[0] - cx(1), pulled[1] - cy(1))
check("pullback moves a distant goal closer to the robot", dist_pulled < dist_orig)

near_centroid = pull_back_goal(clear_grid, 10, 10, RES, OX, OY,
                                robot_x=cx(1), robot_y=cy(1),
                                centroid=(cx(1) + 0.2 * RES, cy(1)),
                                clearance_standoff=0.3, occupied_min=65)
check("pullback is a no-op when already within the standoff",
      near_centroid == (cx(1) + 0.2 * RES, cy(1)))


# -- exclusion / obstacle avoidance ---------------------------------------

check("near_excluded_goal true within radius", near_excluded_goal(0.0, 0.0, [(0.1, 0.0)], radius=0.5))
check("near_excluded_goal false outside radius", not near_excluded_goal(0.0, 0.0, [(5.0, 0.0)], radius=0.5))

moving_track = {"x": 1.0, "y": 0.0, "length": 0.4, "width": 0.4}
check("near_moving_obstacle true within margin+extent",
      near_moving_obstacle(1.1, 0.0, [moving_track], margin=0.3))
check("near_moving_obstacle false far away",
      not near_moving_obstacle(10.0, 0.0, [moving_track], margin=0.3))

candidates_2 = [(0.0, 0.0, 5.0), (10.0, 10.0, 1.0)]
check("select_best_candidate picks the higher score when nothing excludes it",
      select_best_candidate(candidates_2, [], 0.5, [], 0.5) == (0.0, 0.0))
check("select_best_candidate skips an excluded top candidate",
      select_best_candidate(candidates_2, [(0.0, 0.0)], 0.5, [], 0.5) == (10.0, 10.0))
check("select_best_candidate skips a candidate near a moving obstacle",
      select_best_candidate([(1.0, 0.0, 5.0), (10.0, 10.0, 1.0)], [], 0.5,
                             [moving_track], 0.5) == (10.0, 10.0))
check("select_best_candidate returns None when everything is filtered out",
      select_best_candidate([(0.0, 0.0, 5.0)], [(0.0, 0.0)], 0.5, [], 0.5) is None)


# -- end-to-end find_frontier_goal ----------------------------------------

goal = find_frontier_goal(
    grid, W, H, RES, OX, OY, ROBOT_X, ROBOT_Y, excluded_goals=[], tracks=[],
    free_max=40, occupied_min=65, min_frontier_size=1,
    clearance_standoff=0.15, goal_exclude_radius=0.5, obstacle_avoid_radius=0.5)
check("find_frontier_goal returns a goal on the reachable side", goal is not None)
if goal is not None:
    gr, gc = world_to_cell(goal[0], goal[1], OX, OY, RES)
    check("returned goal cell sits on the col-6 frontier boundary", gc == 6)

check("find_frontier_goal returns None when robot cell is out of map bounds",
      find_frontier_goal(grid, W, H, RES, OX, OY, robot_x=1000.0, robot_y=1000.0,
                          excluded_goals=[], tracks=[], free_max=40, occupied_min=65,
                          min_frontier_size=1, clearance_standoff=0.15,
                          goal_exclude_radius=0.5, obstacle_avoid_radius=0.5) is None)

check("find_frontier_goal returns None when the only frontier is excluded",
      find_frontier_goal(grid, W, H, RES, OX, OY, ROBOT_X, ROBOT_Y,
                          excluded_goals=[(goal[0], goal[1])] if goal else [],
                          tracks=[], free_max=40, occupied_min=65, min_frontier_size=1,
                          clearance_standoff=0.15, goal_exclude_radius=2.0,
                          obstacle_avoid_radius=0.5) is None)


# -- raycast_blocked --------------------------------------------------
# Robot at cell (1,1)'s center, facing +x. A wall at (row 1, col 3) sits
# squarely on that ray within the 0.5m lookahead.

wall_grid = np.zeros((10, 10), dtype=np.int16)
wall_grid[1, 3] = 100
check("raycast_blocked true when a wall sits on the ray",
      raycast_blocked(wall_grid, 10, 10, RES, OX, OY,
                       robot_x=cx(1), robot_y=cy(1), yaw=0.0,
                       lookahead=0.5, tracks=[], obstacle_margin=0.15, occupied_min=65))
check("raycast_blocked false on a clear ray (facing away from the wall)",
      not raycast_blocked(wall_grid, 10, 10, RES, OX, OY,
                           robot_x=cx(1), robot_y=cy(1), yaw=math.pi,
                           lookahead=0.3, tracks=[], obstacle_margin=0.15, occupied_min=65))
check("raycast_blocked true when a moving obstacle sits on the ray",
      raycast_blocked(wall_grid, 10, 10, RES, OX, OY,
                       robot_x=cx(1), robot_y=cy(1), yaw=math.pi / 2,
                       lookahead=0.5, tracks=[{"x": cx(1), "y": cy(1) + 0.3,
                                                "length": 0.1, "width": 0.1}],
                       obstacle_margin=0.15, occupied_min=65))

# -- latent hazard: standoff wider than the exclusion radius ---------------
# A stuck goal is stored pulled back by clearance_standoff, but candidates are
# raw centroids. With standoff 0.8 m > exclude radius 0.3 m the same frontier
# used to be picked again right after being excluded.

wide = np.full((60, 60), -1, dtype=np.int16)
wide[:, :40] = 0
_kw = dict(free_max=40, occupied_min=65, min_frontier_size=1, obstacle_avoid_radius=0.5,
           clearance_standoff=0.8, goal_exclude_radius=0.3)
first = find_frontier_goal(wide, 60, 60, RES, OX, OY, -0.5, 2.0, [], [], **_kw)
check("wide-standoff case finds a first goal", first is not None)
again = find_frontier_goal(wide, 60, 60, RES, OX, OY, -0.5, 2.0, [first], [], **_kw)
check("excluded goal is not re-picked when standoff exceeds the exclude radius", again is None)


def test_excluded_goal_not_repicked_with_wide_standoff():
    assert first is not None and again is None


# -- world_to_cell must floor: points just below the origin are out of bounds
check("world_to_cell floors a point half a cell below the origin to -1",
      world_to_cell(OX - 0.5 * RES, OY - 0.5 * RES, OX, OY, RES) == (-1, -1))
check("find_frontier_goal rejects a robot half a cell outside the map",
      find_frontier_goal(grid, W, H, RES, OX, OY, OX - 0.5 * RES, OY - 0.5 * RES,
                          [], [], free_max=40, occupied_min=65, min_frontier_size=1,
                          clearance_standoff=0.15, goal_exclude_radius=0.5,
                          obstacle_avoid_radius=0.5) is None)


def test_world_to_cell_floors_negative_offsets():
    assert world_to_cell(OX - 0.5 * RES, OY - 0.5 * RES, OX, OY, RES) == (-1, -1)


print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("All checks passed.")
