"""Grid-based shortest path between two world points, respecting walls and the
robot's footprint clearance.

A straight Euclidean line between two stops can cut straight through a wall
(e.g. two stops in different rooms) - not a distance the robot could actually
drive. This module finds an 8-connected shortest path over the cells the
robot can actually occupy, so tour length reflects a real, drivable distance.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .map_io import FREE, OCCUPIED, OccupancyGrid

_NEIGHBORS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def traversable_mask(grid: OccupancyGrid, robot_radius_m: float) -> np.ndarray:
    """Boolean (height, width) mask of cells where the robot's footprint is
    fully clear of walls - the same test `is_stop_valid` applies to a single
    point, vectorized over every cell so path search can use it as an edge
    cost. Mirrors `is_stop_valid`'s convention: out-of-grid is never
    considered occupied, so it doesn't block clearance near the map edge."""
    occ = grid.data == OCCUPIED
    radius_px = int(np.ceil(robot_radius_m / grid.resolution))
    h, w = occ.shape
    padded = np.pad(occ, radius_px, mode="constant", constant_values=False)
    near_wall = np.zeros((h, w), dtype=bool)
    for dr in range(-radius_px, radius_px + 1):
        for dc in range(-radius_px, radius_px + 1):
            if dr * dr + dc * dc > radius_px * radius_px:
                continue
            near_wall |= padded[radius_px + dr: radius_px + dr + h, radius_px + dc: radius_px + dc + w]
    return (grid.data == FREE) & ~near_wall


def shortest_path(grid: OccupancyGrid, start_xy: tuple[float, float], goal_xy: tuple[float, float],
                   traversable: np.ndarray) -> tuple[float, list[tuple[int, int]] | None]:
    """A* over `traversable` cells (8-connected, diagonal cost sqrt(2)).
    Returns (length_m, path_as_row_col_list). If either endpoint isn't itself
    traversable (e.g. an invalid stop) or no path exists, falls back to the
    straight-line distance with `path=None` (caller draws a straight segment)."""
    start = grid.world_to_pixel(*start_xy)
    goal = grid.world_to_pixel(*goal_xy)
    straight_line_m = math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])

    if start == goal:
        return 0.0, [start]
    if not (grid.in_bounds(*start) and grid.in_bounds(*goal)
            and traversable[start] and traversable[goal]):
        return straight_line_m, None

    h, w = traversable.shape

    def heuristic(node: tuple[int, int]) -> float:
        return math.hypot(node[0] - goal[0], node[1] - goal[1])

    open_heap: list[tuple[float, tuple[int, int]]] = [(heuristic(start), start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_cost = {start: 0.0}

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node == goal:
            path = [node]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return best_cost[node] * grid.resolution, path

        cost = best_cost[node]
        r, c = node
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not traversable[nr, nc]:
                continue
            step = math.hypot(dr, dc)
            new_cost = cost + step
            neighbor = (nr, nc)
            if new_cost < best_cost.get(neighbor, math.inf):
                best_cost[neighbor] = new_cost
                came_from[neighbor] = node
                heapq.heappush(open_heap, (new_cost + heuristic(neighbor), neighbor))

    return straight_line_m, None  # disconnected free-space regions - shouldn't happen on these maps


def multi_target_shortest_paths(grid: OccupancyGrid, start_xy: tuple[float, float],
                                 goal_xys: list[tuple[float, float]],
                                 traversable: np.ndarray) -> list[tuple[float, list[tuple[int, int]] | None]]:
    """Dijkstra from one start to many goals in a single traversal, instead of
    a separate `shortest_path` search per goal. Settles nearer goals first and
    stops as soon as every goal has been reached, so building a full pairwise
    distance matrix over N stops costs O(N) searches here instead of O(N^2)
    single-target ones. Returns one (length_m, path) per entry in `goal_xys`,
    in the same order, with the same straight-line fallback as `shortest_path`
    for a goal that isn't itself traversable or isn't reachable."""
    start = grid.world_to_pixel(*start_xy)
    h, w = traversable.shape

    results: list[tuple[float, list[tuple[int, int]] | None] | None] = [None] * len(goal_xys)
    pending: dict[tuple[int, int], list[int]] = {}
    for i, xy in enumerate(goal_xys):
        goal = grid.world_to_pixel(*xy)
        if goal == start:
            results[i] = (0.0, [start])
            continue
        pending.setdefault(goal, []).append(i)

    def straight_line(i: int) -> float:
        return math.hypot(goal_xys[i][0] - start_xy[0], goal_xys[i][1] - start_xy[1])

    if not (grid.in_bounds(*start) and traversable[start]):
        for goal, idxs in pending.items():
            for i in idxs:
                results[i] = (straight_line(i), None)
        return results  # type: ignore[return-value]

    for goal in list(pending):
        if not (grid.in_bounds(*goal) and traversable[goal]):
            for i in pending.pop(goal):
                results[i] = (straight_line(i), None)

    best_cost = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]

    while open_heap and pending:
        cost, node = heapq.heappop(open_heap)
        if cost > best_cost.get(node, math.inf):
            continue
        if node in pending:
            path = [node]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            for i in pending.pop(node):
                results[i] = (cost * grid.resolution, path)
            if not pending:
                break

        r, c = node
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not traversable[nr, nc]:
                continue
            new_cost = cost + math.hypot(dr, dc)
            neighbor = (nr, nc)
            if new_cost < best_cost.get(neighbor, math.inf):
                best_cost[neighbor] = new_cost
                came_from[neighbor] = node
                heapq.heappush(open_heap, (new_cost, neighbor))

    for goal, idxs in pending.items():  # unreachable - disconnected free-space region
        for i in idxs:
            results[i] = (straight_line(i), None)
    return results  # type: ignore[return-value]
