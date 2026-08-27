"""Load ROS map_server-style occupancy grids (the same room.pgm + room.yaml pairs
produced by finish_manual_survey.sh / run_overview_stage.sh).

Format reference: https://wiki.ros.org/map_server (image + resolution + origin +
negate + occupied_thresh + free_thresh).
"""
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
    data: np.ndarray  # (height, width) uint8, values FREE/OCCUPIED/UNKNOWN. Row 0 = top of image = max world y.
    resolution: float  # meters/pixel
    origin: tuple[float, float, float]  # world pose (x, y, yaw) of pixel (row=height-1, col=0), i.e. bottom-left

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """World (x, y) meters -> (row, col) pixel indices."""
        col = int((x - self.origin[0]) / self.resolution)
        row_from_bottom = int((y - self.origin[1]) / self.resolution)
        row = self.height - 1 - row_from_bottom
        return row, col

    def pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        """(row, col) pixel indices -> world (x, y) meters, at the pixel center."""
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
        """(N, 2) array of (row, col) for every free cell."""
        return np.argwhere(self.data == FREE)

    def occupied_cells(self) -> np.ndarray:
        """(N, 2) array of (row, col) for every occupied (wall) cell — the scan targets."""
        return np.argwhere(self.data == OCCUPIED)


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
    """Inverse of load_occupancy_grid — used by the synthetic map generator and by
    the scorer when it dumps a coverage-annotated PNG."""
    yaml_path = Path(yaml_path)
    image_name = image_name or (yaml_path.stem + ".pgm")
    image_path = yaml_path.parent / image_name

    pixel = np.full(grid.data.shape, 205, dtype=np.uint8)  # ROS convention: 205 = unknown gray
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
