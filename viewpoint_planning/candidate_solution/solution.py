"""Connected-room viewpoint planning."""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage

from sim.map_io import OccupancyGrid
from sim.pathing import traversable_mask
from sim.visibility import SensorModel, is_stop_valid, scan_from_stop

ROBOT_RADIUS_M = 0.2
CANDIDATE_SPACING_M = 0.05


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
    target = set(map(tuple, np.argwhere(grid.data == 1)))
    coverage: dict[tuple[float, float], set[tuple[int, int]]] = {}
    for candidate in candidates:
        observed = scan_from_stop(grid, candidate, sensor)
        cells = {
            cell for cell, quality in observed.items()
            if quality >= sensor.min_quality and cell in target
        }
        if cells:
            coverage[candidate] = cells

    selected: list[tuple[float, float]] = []
    covered: set[tuple[int, int]] = set()
    while coverage:
        best = max(
            coverage,
            key=lambda candidate: (
                len(coverage[candidate] - covered),
                -candidate[0],
                -candidate[1],
            ),
        )
        gain = coverage[best] - covered
        if not gain:
            break
        selected.append(best)
        covered.update(gain)
        del coverage[best]
    return selected


def _order_connected(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    remaining = set(points[1:])
    ordered = [points[0]]
    while remaining:
        current_x, current_y = ordered[-1]
        next_point = min(
            remaining,
            key=lambda point: (
                (point[0] - current_x) ** 2 + (point[1] - current_y) ** 2,
                point,
            ),
        )
        ordered.append(next_point)
        remaining.remove(next_point)
    return ordered


def plan_viewpoints(grid: OccupancyGrid, sensor: SensorModel) -> list[tuple[float, float]]:
    room = _room_component(grid)
    candidates = _candidates(grid, room)
    if not candidates:
        return []
    stops = _select_stops(grid, candidates, sensor)
    return _order_connected(stops or [candidates[0]])