"""Connected-room viewpoint planning."""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, eye, hstack, vstack

from sim.map_io import OccupancyGrid
from sim.pathing import multi_target_shortest_paths, traversable_mask
from sim.visibility import (
    SensorModel,
    observable_wall_cells,
    scan_from_stop,
)

ROBOT_RADIUS_M = 0.2
CANDIDATE_SPACING_M = 0.05


def _room_component(
    grid: OccupancyGrid, traversable: np.ndarray | None = None
) -> np.ndarray:
    """Return an enclosed drivable component when the map has one.

    Free space connected to an image edge is outside space. A closed room is
    instead a component that does not touch any image edge, so border
    connectivity is a stronger inside/outside test than component size.
    """
    if traversable is None:
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
            # `room` is already the footprint-aware traversable mask.
            if room[row, col]:
                candidates.append(point)
    return candidates


def _select_stops(
    grid: OccupancyGrid,
    candidates: list[tuple[float, float]],
    sensor: SensorModel,
    observable: set[tuple[int, int]],
) -> list[tuple[float, float]]:
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
    coverage_objective = np.concatenate((
        np.zeros(len(candidate_list)),
        -np.ones(len(target)),
    ))

    try:
        coverage_result = milp(
            c=coverage_objective,
            integrality=np.ones(variable_count),
            bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
            constraints=LinearConstraint(
                constraints,
                np.zeros(len(target)),
                np.full(len(target), np.inf),
            ),
            options={"time_limit": 120.0},
        )
        result = coverage_result
        if coverage_result.x is not None:
            best_coverage = int(np.rint(coverage_result.x[len(candidate_list):].sum()))
            coverage_floor = hstack(
                (
                    coo_matrix((1, len(candidate_list))),
                    np.ones((1, len(target))),
                ),
                format="csr",
            )
            second_constraints = vstack((constraints, coverage_floor), format="csr")
            second_lower = np.concatenate((
                np.zeros(len(target)),
                [best_coverage],
            ))
            second_upper = np.full(len(target) + 1, np.inf)
            stop_objective = np.concatenate((
                np.ones(len(candidate_list)),
                np.zeros(len(target)),
            ))
            stop_result = milp(
                c=stop_objective,
                integrality=np.ones(variable_count),
                bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
                constraints=LinearConstraint(
                    second_constraints, second_lower, second_upper
                ),
                options={"time_limit": 30.0},
            )
            if stop_result.x is not None:
                result = stop_result
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

    # Deterministic fallback if the MILP solver is unavailable or times out.
    selected: list[tuple[float, float]] = []
    covered: set[int] = set()
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
        if not gain:
            break
        selected.append(best)
        covered.update(gain)
        del remaining[best]
    return selected


def _order_connected(
    grid: OccupancyGrid,
    points: list[tuple[float, float]],
    traversable: np.ndarray,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
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

    def tour_cost(tour: list[int]) -> float:
        return sum(
            distances[tour[index]][tour[index + 1]]
            for index in range(len(tour) - 1)
        )

    def improve_tour(tour: list[int]) -> list[int]:
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
        return tour

    # Try deterministic starts. Every candidate stop remains in the tour;
    # only visit order changes, so coverage and stop count are unaffected.
    start_indices = sorted(
        {0, min(range(count), key=lambda i: (points[i][0], points[i][1]))},
        key=lambda i: i,
    )
    best_tour: list[int] | None = None
    best_cost = float("inf")
    for start in start_indices:
        unvisited = set(range(count))
        unvisited.remove(start)
        tour = [start]
        while unvisited:
            current = tour[-1]
            next_index = min(
                unvisited,
                key=lambda index: (distances[current][index], index),
            )
            tour.append(next_index)
            unvisited.remove(next_index)
        tour = improve_tour(tour)
        cost = tour_cost(tour)
        if cost < best_cost - 1e-9 or (
            abs(cost - best_cost) <= 1e-9 and tuple(tour) < tuple(best_tour or [])
        ):
            best_tour = tour
            best_cost = cost

    return [points[index] for index in best_tour or []]


def plan_viewpoints(grid: OccupancyGrid, sensor: SensorModel) -> list[tuple[float, float]]:
    traversable = traversable_mask(grid, ROBOT_RADIUS_M)
    room = _room_component(grid, traversable)
    candidates = _candidates(grid, room)
    if not candidates:
        return []
    observable = set(map(tuple, np.argwhere(observable_wall_cells(grid))))
    stops = _select_stops(grid, candidates, sensor, observable)
    return _order_connected(grid, stops or [candidates[0]], traversable)