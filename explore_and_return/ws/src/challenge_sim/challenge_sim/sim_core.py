"""Pure-Python simulation core: unicycle kinematics, wall collision, 2D laser
raycasting, and coverage tracking. No rclpy dependency, so this is directly
unit-testable without a ROS runtime — see test_sim_core.py.

sim_node.py is the thin rclpy wrapper around this.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .map_io import OccupancyGrid, footprint_clear, footprint_clearance_mask


@dataclass
class SimConfig:
    robot_radius_m: float = 0.2
    max_linear_vel: float = 0.3
    max_angular_vel: float = 1.2
    collision_substeps: int = 5
    laser_max_range_m: float = 6.0
    laser_num_rays: int = 360
    laser_range_noise_std: float = 0.01
    odom_linear_noise_std: float = 0.01  # fraction of commanded v
    odom_angular_noise_std: float = 0.01  # fraction of commanded w
    seed: int | None = None


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class LaserScan:
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: list[float]


class SimCore:
    """Owns ground truth. Nothing here is exposed on a ROS topic by sim_node —
    only derived, possibly-noisy odometry and laser data are."""

    def __init__(self, grid: OccupancyGrid, spawn_world: tuple[float, float], config: SimConfig):
        self.grid = grid
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.spawn_world = spawn_world  # true (x, y) — also the "home" the robot must return to
        self.true_pose = Pose2D(x=spawn_world[0], y=spawn_world[1], theta=0.0)
        # Odometry is relative to spawn, so odom-frame (0,0,0) == home.
        self.odom_pose = Pose2D(0.0, 0.0, 0.0)

        self.collision_count = 0
        self.reachable_free: set[tuple[int, int]] = self._flood_fill_reachable()

        # Boolean masks mirroring the sets above, for the vectorized scan() hot loop.
        self._observed_free_mask = np.zeros(grid.data.shape, dtype=bool)
        self._reachable_mask = np.zeros(grid.data.shape, dtype=bool)
        if self.reachable_free:
            rows, cols = zip(*self.reachable_free)
            self._reachable_mask[np.array(rows), np.array(cols)] = True

    # ---------------------------------------------------------------- setup

    def _flood_fill_reachable(self) -> set[tuple[int, int]]:
        """Cells a footprint of radius robot_radius_m can actually occupy,
        reachable from spawn. Plain is_free (single-pixel) reachability would
        credit the robot with "reaching" gaps and wall-hugging strips its body
        can never fit into, inflating the coverage_fraction denominator."""
        start = self.grid.world_to_pixel(*self.spawn_world)
        if not self.grid.is_free(*start):
            raise ValueError("spawn point is not in free space")
        clearance = footprint_clearance_mask(self.grid, self.config.robot_radius_m)
        seen = {start}
        q = deque([start])
        while q:
            row, col = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (row + dr, col + dc)
                if nb in seen or not self.grid.is_free(*nb):
                    continue
                if clearance[nb]:
                    seen.add(nb)
                    q.append(nb)
        return seen

    # ------------------------------------------------------------- dynamics

    def step(self, cmd_v: float, cmd_w: float, dt: float) -> None:
        """Advance the simulation by dt seconds given a commanded (v, w)."""
        cmd_v = max(-self.config.max_linear_vel, min(self.config.max_linear_vel, cmd_v))
        cmd_w = max(-self.config.max_angular_vel, min(self.config.max_angular_vel, cmd_w))

        noisy_v = cmd_v * (1.0 + self.rng.normal(0.0, self.config.odom_linear_noise_std))
        noisy_w = cmd_w * (1.0 + self.rng.normal(0.0, self.config.odom_angular_noise_std))

        # collision_substeps is a floor: scale up so no substep moves more than
        # half a grid cell (otherwise a large dt could tunnel through a wall).
        max_step_dist = self.config.max_linear_vel * dt
        needed_substeps = math.ceil(max_step_dist / (self.grid.resolution * 0.5)) if max_step_dist > 0 else 1
        substeps = max(self.config.collision_substeps, needed_substeps)

        sub_dt = dt / substeps
        for _ in range(substeps):
            blocked = self._substep_true(cmd_v, cmd_w, sub_dt)
            self._substep_odom(noisy_v, noisy_w, sub_dt, blocked)

    def _substep_true(self, v: float, w: float, dt: float) -> bool:
        """Advance true_pose. Returns whether translation was blocked this
        substep (used to keep odom physically consistent — see _substep_odom)."""
        pose = self.true_pose
        new_theta = pose.theta + w * dt
        new_x = pose.x + v * math.cos(pose.theta) * dt
        new_y = pose.y + v * math.sin(pose.theta) * dt

        if footprint_clear(self.grid, new_x, new_y, self.config.robot_radius_m):
            pose.x, pose.y, pose.theta = new_x, new_y, new_theta
            return False

        # blocked translation; rotation still allowed (slide-and-turn, not a hard freeze)
        if pose.theta != new_theta:
            pose.theta = new_theta
        else:
            self.collision_count += 1
        return True

    def _substep_odom(self, v: float, w: float, dt: float, blocked: bool) -> None:
        """Advance odom_pose. Not collision-checked against the map directly
        (a real wheel encoder doesn't know where the walls are), but when the
        robot is physically blocked its translation stalls too, matching a
        real robot's wheels stalling against an obstacle."""
        pose = self.odom_pose
        new_theta = pose.theta + w * dt
        if not blocked:
            pose.x += v * math.cos(pose.theta) * dt
            pose.y += v * math.sin(pose.theta) * dt
        pose.theta = new_theta

    # --------------------------------------------------------------- sensor

    def scan(self) -> LaserScan:
        """Vectorized DDA raycast: every (ray, step) position is computed in
        one shot as a (num_rays, max_steps) array."""
        cfg = self.config
        grid = self.grid
        height, width = grid.data.shape

        origin_row, origin_col = grid.world_to_pixel(self.true_pose.x, self.true_pose.y)
        max_steps = int(cfg.laser_max_range_m / grid.resolution)

        angles = self.true_pose.theta + 2 * np.pi * np.arange(cfg.laser_num_rays) / cfg.laser_num_rays
        dr = -np.sin(angles)
        dc = np.cos(angles)
        steps = np.arange(1, max_steps + 1)

        # (num_rays, max_steps): position of every ray at every step, all at once
        all_r = origin_row + dr[:, None] * steps[None, :]
        all_c = origin_col + dc[:, None] * steps[None, :]
        row = all_r.astype(np.int64)
        col = all_c.astype(np.int64)

        in_bounds = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        row_c = np.clip(row, 0, height - 1)
        col_c = np.clip(col, 0, width - 1)
        occ = in_bounds & (grid.data[row_c, col_c] == 1)  # OCCUPIED == 1

        any_hit = occ.any(axis=1)
        # first True per row = nearest occupied cell along that ray
        first_hit_step = occ.argmax(axis=1)
        hit_range = np.where(any_hit, (first_hit_step + 1) * grid.resolution, cfg.laser_max_range_m)

        # mark every in-bounds cell before the hit (or the whole ray, if no hit) as observed-free
        limit = np.where(any_hit, first_hit_step, max_steps)[:, None]  # (num_rays, 1)
        free_mask_2d = in_bounds & (np.arange(max_steps)[None, :] < limit)
        self._observed_free_mask[row_c[free_mask_2d], col_c[free_mask_2d]] = True

        if cfg.laser_range_noise_std > 0:
            hit_range = hit_range + self.rng.normal(0.0, cfg.laser_range_noise_std, size=cfg.laser_num_rays)
            hit_range = np.clip(hit_range, 0.0, None)

        ranges = np.where(any_hit, hit_range, np.inf)

        # angle_min/max are relative to base_link, not world — heading is
        # applied separately via the odom->base_link TF.
        return LaserScan(
            angle_min=0.0,
            angle_max=2 * math.pi * (cfg.laser_num_rays - 1) / cfg.laser_num_rays,
            angle_increment=2 * math.pi / cfg.laser_num_rays,
            range_min=0.05,
            range_max=cfg.laser_max_range_m,
            ranges=ranges.tolist(),
        )

    # --------------------------------------------------------------- report

    @property
    def coverage_fraction(self) -> float:
        total = self._reachable_mask.sum()
        if total == 0:
            return 0.0
        return float((self._observed_free_mask & self._reachable_mask).sum()) / float(total)

    @property
    def distance_to_home(self) -> float:
        return math.hypot(self.true_pose.x - self.spawn_world[0], self.true_pose.y - self.spawn_world[1])
