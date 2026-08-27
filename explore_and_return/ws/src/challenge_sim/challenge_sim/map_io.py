"""Load/save ROS map_server-style occupancy grids (room.pgm + room.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

FREE = 0
OCCUPIED = 1
UNKNOWN = 2


@dataclass
class OccupancyGrid:
    data: np.ndarray  # (height, width) uint8: FREE/OCCUPIED/UNKNOWN. Row 0 = top of image = max world y.
    resolution: float  # meters/pixel
    origin: tuple[float, float, float]  # world pose (x, y, yaw) of pixel (row=height-1, col=0)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self.origin[0]) / self.resolution)
        row_from_bottom = int((y - self.origin[1]) / self.resolution)
        row = self.height - 1 - row_from_bottom
        return row, col

    def pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self.origin[0] + (col + 0.5) * self.resolution
        row_from_bottom = self.height - 1 - row
        y = self.origin[1] + (row_from_bottom + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.data[row, col] == FREE

    def is_occupied(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.data[row, col] == OCCUPIED

    def free_cells(self) -> np.ndarray:
        return np.argwhere(self.data == FREE)


def footprint_clear(grid: OccupancyGrid, x: float, y: float, robot_radius_m: float) -> bool:
    """True if a circular footprint of robot_radius_m centered at (x, y) contains
    no occupied cell and stays in bounds."""
    row, col = grid.world_to_pixel(x, y)
    radius_px = int(np.ceil(robot_radius_m / grid.resolution))
    for dr in range(-radius_px, radius_px + 1):
        for dc in range(-radius_px, radius_px + 1):
            if dr * dr + dc * dc > radius_px * radius_px:
                continue
            r, c = row + dr, col + dc
            if not grid.in_bounds(r, c) or grid.is_occupied(r, c):
                return False
    return True


def footprint_clearance_mask(grid: OccupancyGrid, robot_radius_m: float) -> np.ndarray:
    """Vectorized equivalent of calling footprint_clear() at every cell:
    mask[row, col] is True iff a circular footprint of robot_radius_m
    centered there contains no occupied cell and stays in bounds."""
    radius_px = int(np.ceil(robot_radius_m / grid.resolution))
    height, width = grid.data.shape

    # pad with "occupied" so a footprint extending past the edge counts as blocked
    padded_occupied = np.ones((height + 2 * radius_px, width + 2 * radius_px), dtype=bool)
    padded_occupied[radius_px:radius_px + height, radius_px:radius_px + width] = grid.data == OCCUPIED

    blocked = np.zeros((height, width), dtype=bool)
    for dr in range(-radius_px, radius_px + 1):
        dc_max = int(np.floor(np.sqrt(radius_px * radius_px - dr * dr)))
        for dc in range(-dc_max, dc_max + 1):
            r0, c0 = radius_px + dr, radius_px + dc
            blocked |= padded_occupied[r0:r0 + height, c0:c0 + width]
    return ~blocked


def sample_free_pose(grid: OccupancyGrid, robot_radius_m: float,
                      rng: np.random.Generator, max_attempts: int = 2000) -> tuple[float, float]:
    """Uniformly sample a random world (x, y) in free space with room for the
    robot's footprint. Raises if none found in max_attempts (e.g. a map with
    no room bigger than the robot)."""
    free = grid.free_cells()
    if len(free) == 0:
        raise ValueError("map has no free space")
    for _ in range(max_attempts):
        row, col = free[rng.integers(0, len(free))]
        x, y = grid.pixel_to_world(int(row), int(col))
        if footprint_clear(grid, x, y, robot_radius_m):
            return x, y
    raise ValueError(f"could not find a valid spawn pose in {max_attempts} attempts")


def load_occupancy_grid(yaml_path: str | Path) -> OccupancyGrid:
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    image_path = yaml_path.parent / meta["image"]
    img = Image.open(image_path).convert("L")
    pixel = np.array(img, dtype=np.float64)

    if meta.get("negate", 0):
        occ = pixel / 255.0
    else:
        occ = (255.0 - pixel) / 255.0

    occupied_thresh = meta.get("occupied_thresh", 0.65)
    free_thresh = meta.get("free_thresh", 0.196)

    data = np.full(occ.shape, UNKNOWN, dtype=np.uint8)
    data[occ >= occupied_thresh] = OCCUPIED
    data[occ <= free_thresh] = FREE

    origin = tuple(meta.get("origin", (0.0, 0.0, 0.0)))
    return OccupancyGrid(data=data, resolution=float(meta["resolution"]), origin=origin)


def save_occupancy_grid(grid: OccupancyGrid, yaml_path: str | Path, image_name: str | None = None) -> None:
    yaml_path = Path(yaml_path)
    image_name = image_name or (yaml_path.stem + ".pgm")
    image_path = yaml_path.parent / image_name

    pixel = np.full(grid.data.shape, 205, dtype=np.uint8)
    pixel[grid.data == FREE] = 254
    pixel[grid.data == OCCUPIED] = 0
    Image.fromarray(pixel, mode="L").save(image_path)

    meta = {
        "image": image_name,
        "resolution": grid.resolution,
        "origin": list(grid.origin),
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=False)
