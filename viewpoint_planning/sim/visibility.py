"""2D raycasting visibility for the accurate-scan sensor model.

Mirrors what a real rotating high-accuracy 2D laser would see from a fixed stop:
cast a dense ring of rays, walk each one cell-by-cell (DDA) until it hits an
occupied (wall) cell or exceeds max_range, and record that wall cell as
"observed" from this stop, with a quality score that falls off with range.

Deliberately not modeled: incidence angle relative to the wall's local surface
normal (a real scanner sees a wall obliquely-struck less well than one hit
head-on). That's a legitimate axis for a strong candidate to add — the hook is
`quality_at_range()` below.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .map_io import FREE, OCCUPIED, OccupancyGrid


@dataclass
class SensorModel:
    max_range_m: float = 16
    num_rays: int = 720  # 0.5 degree angular resolution
    min_quality: float = 0.5  # a wall cell only counts as "scanned" if observed at >= this quality


def quality_at_range(
    range_m: float | np.ndarray,
    max_range_m: float,
    incidence_cosine: float | np.ndarray = 1.0,
) -> float | np.ndarray:
    """Combine range falloff with wall incidence.

    ``incidence_cosine`` is 1 for a head-on hit and 0 for a grazing hit.
    Grazing returns are less accurate because the beam travels nearly along
    the wall, making the hit position more sensitive to range and map noise.
    A 0.75 floor keeps grazing hits useful while still penalizing them below
    head-on measurements.
    """
    range_quality = np.maximum(0.0, 1.0 - (np.asarray(range_m) / max_range_m) ** 2)
    incidence = np.clip(np.asarray(incidence_cosine), 0.0, 1.0)
    incidence_quality = 0.75 + 0.25 * incidence
    return range_quality * incidence_quality


@lru_cache(maxsize=16)
def _ray_directions(num_rays: int) -> tuple[np.ndarray, np.ndarray]:
    angles = 2 * np.pi * np.arange(num_rays) / num_rays
    return -np.sin(angles), np.cos(angles)


def _incidence_cosine(
    grid: OccupancyGrid,
    rows: np.ndarray,
    cols: np.ndarray,
    origin_row: int,
    origin_col: int,
) -> np.ndarray:
    """Estimate wall incidence from the hit cell's free-space neighbors."""
    free = grid.data == FREE
    normal_rows = np.zeros(len(rows), dtype=np.float64)
    normal_cols = np.zeros(len(cols), dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor_rows = rows + dr
            neighbor_cols = cols + dc
            valid = (
                (neighbor_rows >= 0) & (neighbor_rows < grid.height)
                & (neighbor_cols >= 0) & (neighbor_cols < grid.width)
            )
            free_neighbor = np.zeros(len(rows), dtype=bool)
            free_neighbor[valid] = free[
                neighbor_rows[valid], neighbor_cols[valid]
            ]
            normal_rows[free_neighbor] += dr
            normal_cols[free_neighbor] += dc

    normal_length = np.hypot(normal_rows, normal_cols)
    ray_rows = rows - origin_row
    ray_cols = cols - origin_col
    ray_length = np.hypot(ray_rows, ray_cols)
    denominator = normal_length * ray_length
    cosine = np.zeros(len(rows), dtype=np.float64)
    valid = denominator > 0.0
    cosine[valid] = np.abs(
        normal_rows[valid] * ray_rows[valid]
        + normal_cols[valid] * ray_cols[valid]
    ) / denominator[valid]
    return np.clip(cosine, 0.0, 1.0)


def scan_from_stop(grid: OccupancyGrid, stop_xy: tuple[float, float],
                    sensor: SensorModel) -> dict[tuple[int, int], float]:
    """Simulate one full rotate-and-scan from a stop. Returns {(row, col):
    quality} for every wall cell seen at quality > 0, keyed by the pixel it
    hit (best quality wins if a cell is somehow hit by more than one ray).

    All of a stop's rays are marched together as numpy arrays (DDA, one step
    per iteration across every ray at once) rather than one Python-level ray
    at a time - this is called once per candidate stop a planner considers,
    so its cost adds up quickly across a large candidate set on a large map."""
    origin_row, origin_col = grid.world_to_pixel(*stop_xy)
    max_range_px = sensor.max_range_m / grid.resolution
    steps = int(max_range_px)
    h, w = grid.data.shape
    occ_mask = grid.data == OCCUPIED

    # These directions depend only on ray count, so reuse them across stops.
    dr, dc = _ray_directions(sensor.num_rays)
    r = np.full(sensor.num_rays, float(origin_row))
    c = np.full(sensor.num_rays, float(origin_col))
    active = np.ones(sensor.num_rays, dtype=bool)

    seen: dict[tuple[int, int], float] = {}
    for _ in range(steps):
        if not active.any():
            break
        r += dr
        c += dc
        ir = r.astype(int)
        ic = c.astype(int)
        in_bounds = active & (ir >= 0) & (ir < h) & (ic >= 0) & (ic < w)
        active &= in_bounds

        occ_now = np.zeros(sensor.num_rays, dtype=bool)
        occ_now[active] = occ_mask[ir[active], ic[active]]
        hit_now = active & occ_now
        if hit_now.any():
            hit_rows = ir[hit_now]
            hit_cols = ic[hit_now]
            range_px = np.hypot(hit_rows - origin_row, hit_cols - origin_col)
            incidence = _incidence_cosine(
                grid, hit_rows, hit_cols, origin_row, origin_col
            )
            quality = quality_at_range(
                range_px * grid.resolution,
                sensor.max_range_m,
                incidence,
            )
            for row, col, q in zip(hit_rows, hit_cols, quality):
                cell = (int(row), int(col))
                if q > seen.get(cell, 0.0):
                    seen[cell] = q
            active &= ~hit_now
    return seen


def observable_wall_cells(grid: OccupancyGrid) -> np.ndarray:
    """Boolean (height, width) mask of wall cells that have at least one FREE
    8-neighbor inside the grid.

    A ray in `scan_from_stop` always stops at the *first* occupied cell it hits, so
    a cell buried inside a thick wall (e.g. a multi-pixel-deep exterior wall)
    can never be the hit cell for any ray from any free-space stop, no matter
    how the stops are placed. Counting those interior cells toward a coverage
    denominator makes 100% coverage geometrically impossible.

    Neighbor check is against FREE specifically, not merely non-occupied:
    UNKNOWN cells are never a valid stop (`is_stop_valid` requires FREE), and
    a wall cell facing only UNKNOWN space (e.g. the outer face of an exterior
    wall ring, with the unmapped/unreachable area beyond it) is not actually
    reachable either - any ray from inside the free interior hits the wall's
    free-facing side first, exactly like an interior cell would be. A cell
    directly adjacent to a FREE cell is always reachable, at minimum by a ray
    cast from that neighbor itself."""
    occ = grid.data == OCCUPIED
    free = grid.data == FREE
    padded = np.pad(free, 1, mode="constant", constant_values=False)
    neighbor_free = np.zeros_like(occ, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor_free |= padded[1 + dr: 1 + dr + occ.shape[0], 1 + dc: 1 + dc + occ.shape[1]]
    return occ & neighbor_free


def is_stop_valid(grid: OccupancyGrid, stop_xy: tuple[float, float], robot_radius_m: float = 0.2) -> bool:
    """A stop is only physically realizable if it's in free space with room for
    the robot's footprint around it."""
    row, col = grid.world_to_pixel(*stop_xy)
    if not grid.is_free(row, col):
        return False
    radius_px = int(np.ceil(robot_radius_m / grid.resolution))
    for dr in range(-radius_px, radius_px + 1):
        for dc in range(-radius_px, radius_px + 1):
            if dr * dr + dc * dc > radius_px * radius_px:
                continue
            r, c = row + dr, col + dc
            if grid.in_bounds(r, c) and grid.is_occupied(r, c):
                return False
    return True
