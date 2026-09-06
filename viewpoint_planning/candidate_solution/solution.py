"""Connected-room viewpoint planning."""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, eye, hstack

from sim.map_io import OccupancyGrid
from sim.pathing import multi_target_shortest_paths, traversable_mask
from sim.visibility import (
    SensorModel,
    is_stop_valid,
    observable_wall_cells,
    scan_from_stop,
)

ROBOT_RADIUS_M = 0.2
CANDIDATE_SPACING_M = 0.8
MIN_GAIN_PERCENT = 0.5


def _room_component(grid: OccupancyGrid) -> np.ndarray:
    """Return an enclosed drivable component when the map has one.

    Free space connected to an image edge is outside space. A closed room is
    instead a component that does not touch any image edge, so border
    connectivity is a stronger inside/outside test than component size.
    """
    traversable = traversable_mask(grid, ROBOT_RADIUS_M)
    labels, component_count = ndimage.label(
        traversable, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if component_count == 0:
        return traversable

    occupied = grid.data == 1
    wall_contacts: Counter[int] = Counter()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.zeros_like(labels)
            source_rows = slice(max(0, dr), min(grid.height, grid.height + dr))
            source_cols = slice(max(0, dc), min(grid.width, grid.width + dc))
            target_rows = slice(max(0, -dr), min(grid.height, grid.height - dr))
            target_cols = slice(max(0, -dc), min(grid.width, grid.width - dc))
            shifted[target_rows, target_cols] = labels[source_rows, source_cols]
            for component_id in np.unique(shifted[occupied]):
                if component_id:
                    wall_contacts[int(component_id)] += 1

    component_sizes = np.bincount(labels.ravel())
    border_ids = set(np.unique(np.concatenate((
        labels[0], labels[-1], labels[:, 0], labels[:, -1]
    ))))
    enclosed_ids = [
        component_id for component_id in range(1, component_count + 1)
        if component_id not in border_ids
    ]
    candidate_ids = enclosed_ids or list(range(1, component_count + 1))
    selected_id = max(
        candidate_ids,
        key=lambda component_id: (
            wall_contacts[component_id],
            int(component_sizes[component_id]),
            -component_id,
        ),
    )
    return labels == selected_id


def _candidates(grid: OccupancyGrid, room: np.ndarray) -> list[tuple[float, float]]:
    step_px = max(1, int(round(CANDIDATE_SPACING_M / grid.resolution)))
    candidates: list[tuple[float, float]] = []
    for row in range(0, grid.height, step_px):
        for col in range(0, grid.width, step_px):
            if not room[row, col]:
                continue
            point = grid.pixel_to_world(row, col)
            if is_stop_valid(grid, point, ROBOT_RADIUS_M):
                candidates.append(point)
    return candidates


def _select_stops(
    grid: OccupancyGrid,
    candidates: list[tuple[float, float]],
    sensor: SensorModel,
) -> list[tuple[float, float]]:
    observable = set(map(tuple, np.argwhere(observable_wall_cells(grid))))
    coverage: dict[tuple[float, float], set[int]] = {}
    candidate_cells: dict[tuple[float, float], set[tuple[int, int]]] = {}
    for candidate in candidates:
        observed = scan_from_stop(grid, candidate, sensor)
        cells = {
            cell for cell, quality in observed.items()
            if quality >= sensor.min_quality and cell in observable
        }
        if cells:
            candidate_cells[candidate] = cells

    # A wall can be observable in principle yet impossible for this finite
    # candidate set to scan at the required quality. Exclude such cells from
    # the optimization model so the solver cannot spend effort representing
    # impossible coverage.
    target = sorted(set().union(*candidate_cells.values())) if candidate_cells else []
    target_index = {cell: index for index, cell in enumerate(target)}
    for candidate, cells in candidate_cells.items():
        coverage[candidate] = {target_index[cell] for cell in cells}

    if not coverage:
        return []

    candidate_list = list(coverage)
    row_indices: list[int] = []
    column_indices: list[int] = []
    for candidate_index, candidate in enumerate(candidate_list):
        for target_cell_index in coverage[candidate]:
            row_indices.append(target_cell_index)
            column_indices.append(candidate_index)

    wall_candidate_matrix = coo_matrix(
        (np.ones(len(row_indices)), (row_indices, column_indices)),
        shape=(len(target), len(candidate_list)),
    ).tocsr()
    variable_count = len(candidate_list) + len(target)

    # y_j can be one only when at least one selected stop sees wall j.
    constraints = hstack(
        (wall_candidate_matrix, -eye(len(target), format="csr")),
        format="csr",
    )
    stop_penalty = 1.0 / (len(candidate_list) + 1)
    objective = np.concatenate((
        np.full(len(candidate_list), stop_penalty),
        -np.ones(len(target)),
    ))

    try:
        result = milp(
            c=objective,
            integrality=np.ones(variable_count),
            bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
            constraints=LinearConstraint(
                constraints, np.zeros(len(target)), np.full(len(target), np.inf)
            ),
            options={"time_limit": 30.0},
        )
    except (ImportError, RuntimeError, ValueError):
        result = None

    if result is not None and result.x is not None:
        selected = [
            candidate
            for index, candidate in enumerate(candidate_list)
            if result.x[index] >= 0.5
        ]
        if selected:
            return selected

    # Deterministic fallback if the MILP solver is unavailable or times out
    # before producing an incumbent solution.
    selected: list[tuple[float, float]] = []
    covered: set[int] = set()
    minimum_gain = max(1, len(target) * MIN_GAIN_PERCENT / 100.0)
    remaining = dict(coverage)
    while remaining:
        best = max(
            remaining,
            key=lambda candidate: (
                len(remaining[candidate] - covered),
                -candidate[0],
                -candidate[1],
            ),
        )
        gain = remaining[best] - covered
        if len(gain) < minimum_gain:
            break
        selected.append(best)
        covered.update(gain)
        del remaining[best]
    return selected


def _order_connected(
    grid: OccupancyGrid, points: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    traversable = traversable_mask(grid, ROBOT_RADIUS_M)
    count = len(points)
    distances = [[0.0] * count for _ in range(count)]
    for start_index, start in enumerate(points):
        goal_indices = [index for index in range(count) if index != start_index]
        goals = [points[index] for index in goal_indices]
        results = multi_target_shortest_paths(
            grid, start, goals, traversable
        )
        for goal_index, (distance, _path) in zip(goal_indices, results):
            distances[start_index][goal_index] = distance

    # Deterministic nearest-neighbor initial tour.
    unvisited = set(range(1, count))
    tour = [0]
    while unvisited:
        current = tour[-1]
        next_index = min(
            unvisited,
            key=lambda index: (distances[current][index], index),
        )
        tour.append(next_index)
        unvisited.remove(next_index)

    # 2-opt for an open tour: reverse each improving internal segment.
    improved = True
    while improved:
        improved = False
        for first in range(1, count - 1):
            for last in range(first + 1, count):
                before = tour[first - 1]
                after = tour[last]
                old_cost = distances[before][tour[first]]
                new_cost = distances[before][after]
                if last + 1 < count:
                    old_cost += distances[tour[last]][tour[last + 1]]
                    new_cost += distances[tour[first]][tour[last + 1]]
                if new_cost < old_cost - 1e-9:
                    tour[first:last + 1] = reversed(tour[first:last + 1])
                    improved = True

    return [points[index] for index in tour]


def plan_viewpoints(grid: OccupancyGrid, sensor: SensorModel) -> list[tuple[float, float]]:
    room = _room_component(grid)
    candidates = _candidates(grid, room)
    if not candidates:
        return []
    stops = _select_stops(grid, candidates, sensor)
    return _order_connected(grid, stops or [candidates[0]])