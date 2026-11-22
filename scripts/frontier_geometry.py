"""Pure grid/geometry logic behind frontier_explorer_go2.py.

Factored out of the ROS node so it's testable without rclpy/a running ROS
graph (see test_frontier_geometry.py) -- everything here is plain
numpy/math over an occupancy grid, robot pose, and obstacle-tracker output.
"""

import math

import numpy as np


# m, extra room beyond clearance_standoff when matching excluded goals
EXCLUDE_MARGIN = 0.1


def world_to_cell(x, y, ox, oy, res):
    # floor, not int(): int() truncates toward zero, so a point up to one cell
    # below the map origin would land in row/col 0 instead of -1 (out of bounds)
    return math.floor((y - oy) / res), math.floor((x - ox) / res)  # (row, col)


def cell_center_world(r, c, ox, oy, res):
    return ox + (c + 0.5) * res, oy + (r + 0.5) * res


def snap_to_free_cell(free, rr, rc, w, h, max_radius=5):
    """Nearest free cell to (rr, rc) within max_radius, or None.

    Needed right after spawn / a tight replan when the robot's own cell
    hasn't been marked free yet even though it obviously is one.
    """
    if free[rr, rc]:
        return rr, rc
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r2, c2 = rr + dr, rc + dc
                if 0 <= r2 < h and 0 <= c2 < w and free[r2, c2]:
                    return r2, c2
    return None


def wavefront_frontier_cells(grid, free, rr, rc, w, h, occupied_min):
    """Frontier cells reachable from (rr, rc) via free (non-inflated) cells.

    Unknown cells (-1) must NOT count as free -- excluding them is what
    makes a cell adjacent to unknown space a frontier at all; otherwise the
    search floods straight through unmapped territory and never sees a
    boundary to stop at.
    """
    reachable = np.zeros_like(free)
    stack = [(rr, rc)]
    reachable[rr, rc] = True
    frontier_cells = []
    while stack:
        r, c = stack.pop()
        is_frontier = False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            val = grid[nr, nc]
            if val < 0:  # unknown
                is_frontier = True
                continue
            if free[nr, nc] and not reachable[nr, nc] and val < occupied_min:
                reachable[nr, nc] = True
                stack.append((nr, nc))
        if is_frontier:
            frontier_cells.append((r, c))
    return frontier_cells


def cluster_frontier_cells(frontier_cells):
    """4-connected clusters of frontier cells, as a list of cell lists."""
    frontier_set = set(frontier_cells)
    visited = set()
    clusters = []
    for cell in frontier_cells:
        if cell in visited:
            continue
        cluster = []
        queue = [cell]
        visited.add(cell)
        while queue:
            r, c = queue.pop()
            cluster.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in frontier_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        clusters.append(cluster)
    return clusters


def score_clusters(clusters, ox, oy, res, robot_x, robot_y, min_frontier_size):
    """[(gx, gy, score), ...] for clusters at/above min_frontier_size.

    score = size / (1 + distance from robot): bigger unknown regions win,
    but nearer ones are cheaper to reach -- same tradeoff explore_lite uses.
    """
    candidates = []
    for cluster in clusters:
        if len(cluster) < min_frontier_size:
            continue
        cr = sum(p[0] for p in cluster) / len(cluster)
        cc = sum(p[1] for p in cluster) / len(cluster)
        gx, gy = cell_center_world(cr, cc, ox, oy, res)
        dist = math.hypot(gx - robot_x, gy - robot_y)
        score = len(cluster) / (1.0 + dist)
        candidates.append((gx, gy, score))
    return candidates


def near_excluded_goal(x, y, excluded_goals, radius):
    return any(math.hypot(x - gx, y - gy) < radius for gx, gy in excluded_goals)


def near_moving_obstacle(x, y, tracks, margin):
    for track in tracks:
        extent = max(track.get('length', 0.0), track.get('width', 0.0)) / 2.0
        if math.hypot(x - track['x'], y - track['y']) < margin + extent:
            return True
    return False


def select_best_candidate(candidates, excluded_goals, goal_exclude_radius,
                           tracks, obstacle_avoid_radius):
    """Highest-scoring candidate not near an excluded goal or moving obstacle."""
    best_score, best_centroid = -1.0, None
    for gx, gy, score in candidates:
        if near_excluded_goal(gx, gy, excluded_goals, goal_exclude_radius):
            continue
        if near_moving_obstacle(gx, gy, tracks, obstacle_avoid_radius):
            continue
        if score > best_score:
            best_score, best_centroid = score, (gx, gy)
    return best_centroid


def cell_is_clear(grid, w, h, res, ox, oy, x, y, clearance_standoff, occupied_min):
    """Known, free, and with no occupied cell within clearance_standoff."""
    r, c = world_to_cell(x, y, ox, oy, res)
    if not (0 <= r < h and 0 <= c < w) or grid[r, c] < 0 or grid[r, c] >= occupied_min:
        return False
    radius_cells = max(1, int(clearance_standoff / res))
    r0, r1 = max(0, r - radius_cells), min(h, r + radius_cells + 1)
    c0, c1 = max(0, c - radius_cells), min(w, c + radius_cells + 1)
    return bool(np.all(grid[r0:r1, c0:c1] < occupied_min))


def pull_back_goal(grid, w, h, res, ox, oy, robot_x, robot_y, centroid,
                    clearance_standoff, occupied_min):
    """Nudge a frontier centroid back toward the robot, off the free/unknown
    boundary the wavefront search necessarily leaves it on, into space
    already confirmed reachable and clear of nearby occupied cells."""
    gx, gy = centroid
    dist = math.hypot(gx - robot_x, gy - robot_y)
    if dist <= clearance_standoff:
        return centroid
    frac = 1.0 - (clearance_standoff / dist)
    px, py = robot_x + frac * (gx - robot_x), robot_y + frac * (gy - robot_y)
    if cell_is_clear(grid, w, h, res, ox, oy, px, py, clearance_standoff, occupied_min):
        return (px, py)
    return centroid  # pullback landed somewhere worse -- keep the original


def find_frontier_goal(grid, w, h, res, ox, oy, robot_x, robot_y, excluded_goals,
                        tracks, *, free_max, occupied_min, min_frontier_size,
                        clearance_standoff, goal_exclude_radius, obstacle_avoid_radius):
    """Full frontier-goal pipeline: wavefront BFS -> cluster -> score ->
    filter (excluded goals / moving obstacles) -> pick best -> pull back off
    the boundary. Returns (x, y) in the same frame as robot_x/robot_y, or
    None if there's nothing left to explore (or nothing reachable)."""
    rr, rc = world_to_cell(robot_x, robot_y, ox, oy, res)
    if not (0 <= rr < h and 0 <= rc < w):
        return None
    known = grid >= 0
    free = known & (grid < free_max)
    snapped = snap_to_free_cell(free, rr, rc, w, h)
    if snapped is None:
        return None
    rr, rc = snapped
    frontier_cells = wavefront_frontier_cells(grid, free, rr, rc, w, h, occupied_min)
    if not frontier_cells:
        return None
    clusters = cluster_frontier_cells(frontier_cells)
    candidates = score_clusters(clusters, ox, oy, res, robot_x, robot_y, min_frontier_size)
    # The excluded points are pulled-back goals, but candidates are still raw
    # centroids, which sit up to clearance_standoff away from them. Keep the
    # exclusion circle wider than that, or a stuck goal is picked again.
    effective_exclude_radius = max(goal_exclude_radius,
                                   clearance_standoff + EXCLUDE_MARGIN)
    best = select_best_candidate(
        candidates, excluded_goals, effective_exclude_radius, tracks, obstacle_avoid_radius)
    if best is None:
        return None
    return pull_back_goal(grid, w, h, res, ox, oy, robot_x, robot_y, best,
                           clearance_standoff, occupied_min)


def raycast_blocked(grid, w, h, res, ox, oy, robot_x, robot_y, yaw, lookahead,
                     tracks, obstacle_margin, occupied_min):
    """Is anything (occupied cell or moving obstacle) in the way along the
    lookahead ray? Samples the whole ray, not just its endpoint -- a single
    distant sample can miss a wall between the robot and the lookahead
    point, or a thin obstacle the endpoint happens to clear."""
    steps = max(1, int(lookahead / max(res, 0.01)))
    for i in range(1, steps + 1):
        d = lookahead * i / steps
        lx = robot_x + d * math.cos(yaw)
        ly = robot_y + d * math.sin(yaw)
        if near_moving_obstacle(lx, ly, tracks, obstacle_margin):
            return True
        r, c = world_to_cell(lx, ly, ox, oy, res)
        if not (0 <= r < h and 0 <= c < w):
            continue
        if grid[r, c] >= occupied_min:
            return True
    return False
