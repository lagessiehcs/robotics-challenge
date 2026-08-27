"""Regression tests for the pure-physics sim core (no rclpy required).
Run with: python3 -m pytest test/test_sim_core.py -v
(or plain `python3 test/test_sim_core.py` from the challenge_sim package root)
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from challenge_sim.map_io import FREE, OCCUPIED, OccupancyGrid, sample_free_pose
from challenge_sim.sim_core import SimCore, SimConfig


@pytest.fixture
def grid() -> OccupancyGrid:
    """8x6m room with an internal wall + doorway connecting two halves."""
    res = 0.05
    h, w = 120, 160
    data = np.full((h, w), FREE, dtype=np.uint8)
    data[:3, :] = OCCUPIED
    data[-3:, :] = OCCUPIED
    data[:, :3] = OCCUPIED
    data[:, -3:] = OCCUPIED
    data[:, 78:82] = OCCUPIED
    data[50:70, 78:82] = FREE  # doorway
    return OccupancyGrid(data=data, resolution=res, origin=(0.0, 0.0, 0.0))


def _turn(sim: SimCore, theta: float, dt: float = 0.1) -> None:
    """Command a pure rotation via .step() so true_pose and odom_pose stay
    in sync, the way a real cmd_vel sequence would."""
    w = sim.config.max_angular_vel if theta >= 0 else -sim.config.max_angular_vel
    steps = int(abs(theta) / (sim.config.max_angular_vel * dt)) + 2
    for _ in range(steps):
        sim.step(cmd_v=0.0, cmd_w=w, dt=dt)


def test_reachable_flood_fill_spans_both_rooms(grid):
    sim = SimCore(grid, grid.pixel_to_world(60, 40), SimConfig(seed=42))
    assert len(sim.reachable_free) > 5000


def test_unicycle_kinematics(grid):
    sim = SimCore(grid, grid.pixel_to_world(60, 40), SimConfig(seed=42))
    x0 = sim.true_pose.x
    sim.step(cmd_v=0.2, cmd_w=0.0, dt=1.0)
    assert abs((sim.true_pose.x - x0) - 0.2) < 0.02


def test_collision_stops_translation_without_tunneling(grid):
    cfg = SimConfig(seed=42)
    sim = SimCore(grid, grid.pixel_to_world(60, 10), cfg)
    _turn(sim, math.pi)
    for _ in range(80):
        sim.step(cmd_v=0.3, cmd_w=0.0, dt=0.2)
    row, col = grid.world_to_pixel(sim.true_pose.x, sim.true_pose.y)
    assert not grid.is_occupied(row, col)
    assert sim.collision_count > 0


def test_odom_stalls_with_true_pose_when_blocked(grid):
    """Odom must stall along with the true pose when the robot is blocked,
    not keep integrating translation while physically stuck against a wall."""
    cfg = SimConfig(seed=42)
    sim = SimCore(grid, grid.pixel_to_world(60, 40), cfg)
    _turn(sim, math.pi)
    for _ in range(80):
        sim.step(cmd_v=0.3, cmd_w=0.0, dt=0.2)
    true_dx = sim.true_pose.x - sim.spawn_world[0]
    assert abs(sim.odom_pose.x - true_dx) < 0.1


def test_scan_sees_walls_and_grows_coverage(grid):
    sim = SimCore(grid, grid.pixel_to_world(60, 40), SimConfig(seed=42))
    scan = sim.scan()
    finite = [r for r in scan.ranges if math.isfinite(r)]
    assert len(finite) > 0
    assert sim.coverage_fraction > 0.0


def test_coverage_is_monotonic_while_exploring(grid):
    sim = SimCore(grid, grid.pixel_to_world(60, 40), SimConfig(seed=42))
    prev = 0.0
    for _ in range(30):
        sim.step(cmd_v=0.15, cmd_w=0.3, dt=0.5)
        sim.scan()
        cov = sim.coverage_fraction
        assert cov >= prev - 1e-9
        prev = cov


def test_velocity_is_clamped(grid):
    cfg = SimConfig(seed=42)
    sim = SimCore(grid, grid.pixel_to_world(60, 40), cfg)
    x0 = sim.true_pose.x
    sim.step(cmd_v=999.0, cmd_w=0.0, dt=1.0)
    assert abs((sim.true_pose.x - x0) - cfg.max_linear_vel) < 0.05


def test_distance_to_home(grid):
    sim = SimCore(grid, grid.pixel_to_world(60, 40), SimConfig(seed=42))
    assert sim.distance_to_home < 1e-9
    sim.step(cmd_v=0.3, cmd_w=0.0, dt=1.0)
    assert sim.distance_to_home > 0.2


def test_scan_hits_land_on_real_walls_when_rotated(grid):
    """Reconstruct each hit's world position the way a real consumer
    (RViz/slam_toolbox/Nav2) does — heading + angle_min + i*increment —
    and check it lands on an actual occupied cell."""
    sim = SimCore(grid, grid.pixel_to_world(60, 40), SimConfig(seed=42))
    _turn(sim, math.pi / 2)  # rotate so heading isn't masked at theta=0
    scan = sim.scan()

    checked = 0
    for i, r in enumerate(scan.ranges):
        if not math.isfinite(r):
            continue
        world_angle = sim.true_pose.theta + scan.angle_min + i * scan.angle_increment
        hit_x = sim.true_pose.x + r * math.cos(world_angle)
        hit_y = sim.true_pose.y + r * math.sin(world_angle)
        row, col = grid.world_to_pixel(hit_x, hit_y)
        # allow the immediate neighbourhood: pixel rounding at a grazing hit
        # can land one cell short of the wall it actually grazed
        neighborhood_occupied = any(
            grid.is_occupied(row + dr, col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        )
        assert neighborhood_occupied, f"ray {i}: reconstructed hit ({hit_x:.2f},{hit_y:.2f}) is not near a wall"
        checked += 1
    assert checked > 50  # sanity: the room isn't so open that nothing was hit


def test_sample_free_pose_avoids_walls_and_narrow_gaps(grid):
    rng = np.random.default_rng(1)
    for _ in range(50):
        x, y = sample_free_pose(grid, robot_radius_m=0.2, rng=rng)
        row, col = grid.world_to_pixel(x, y)
        assert grid.is_free(row, col)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
