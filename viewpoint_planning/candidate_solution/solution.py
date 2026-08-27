"""Your task: implement plan_viewpoints().

You are given the ground-truth occupancy grid (this is the ONLY input you get —
there is no live sensor loop in this challenge, the map is already known) and
the sensor model the accurate scanner uses. Return a list of (x, y) world-frame
stop poses, IN VISIT ORDER, such that a 360-degree rotate-and-scan from each
stop, together, observes as much of the wall/boundary as possible, using as
few stops and as little travel as you can.

See README.md for the full brief, scoring rubric, and how to run
`eval.py` against your solution.
"""
from __future__ import annotations

from sim.map_io import OccupancyGrid
from sim.visibility import SensorModel


def plan_viewpoints(grid: OccupancyGrid, sensor: SensorModel) -> list[tuple[float, float]]:
    """Replace this. The placeholder below is intentionally bad — a single
    stop at the map's centroid — so you can see the scorer and visualization
    working end-to-end before you touch the algorithm.
    """
    free_cells = grid.free_cells()
    center_row, center_col = free_cells.mean(axis=0)
    x, y = grid.pixel_to_world(int(center_row), int(center_col))
    return [(x, y)]
