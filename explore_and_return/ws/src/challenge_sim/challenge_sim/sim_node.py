"""rclpy wrapper around sim_core.SimCore. This node IS the "robot": it publishes
/scan and /odom, broadcasts odom->base_link, consumes /cmd_vel, and exposes the
/finish_exploration service a candidate's node calls when it believes the
survey (explore + return home) is complete.

It's also the sole /clock authority for the whole graph — every other node
reads "now" from /clock, so scaling it speeds up the entire stack together.
sim_node itself stays on the wall clock (see main()), since it's the thing
computing what "sim time" even is.

Ground truth (the real map, the true pose) never appears on any topic — only
the report written at the end of the session reveals it.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Quaternion, Twist, TransformStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

# dynamic_typing so a launch override like time_limit_s:=300 (parsed as int)
# doesn't crash a param declared with a float default.
_NUMERIC = ParameterDescriptor(dynamic_typing=True)

from .map_io import FREE, OCCUPIED, UNKNOWN, load_occupancy_grid, sample_free_pose, save_occupancy_grid
from .map_io import OccupancyGrid as MapGrid
from .sim_core import SimConfig, SimCore


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def seconds_to_time_msg(seconds: float) -> TimeMsg:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    return TimeMsg(sec=sec, nanosec=nanosec)


class SimNode(Node):
    def __init__(self) -> None:
        super().__init__("challenge_sim_node")

        self.declare_parameter("map_yaml", "")
        self.declare_parameter("robot_radius_m", 0.2, _NUMERIC)
        self.declare_parameter("max_linear_vel", 0.3, _NUMERIC)
        self.declare_parameter("max_angular_vel", 1.2, _NUMERIC)
        self.declare_parameter("laser_max_range_m", 6.0, _NUMERIC)
        self.declare_parameter("laser_num_rays", 360, _NUMERIC)
        self.declare_parameter("laser_range_noise_std", 0.01, _NUMERIC)
        self.declare_parameter("odom_linear_noise_std", 0.01, _NUMERIC)
        self.declare_parameter("odom_angular_noise_std", 0.01, _NUMERIC)
        self.declare_parameter("sim_period_s", 0.05, _NUMERIC)
        self.declare_parameter("scan_period_s", 0.1, _NUMERIC)
        self.declare_parameter("cmd_vel_timeout_s", 0.5, _NUMERIC)
        self.declare_parameter("time_limit_s", 5400.0, _NUMERIC)
        self.declare_parameter("time_scale", 1.0, _NUMERIC)
        self.declare_parameter("return_tolerance_m", 0.3, _NUMERIC)
        self.declare_parameter("min_coverage_required", 0.8, _NUMERIC)
        self.declare_parameter("results_path", "results/report.yaml")
        self.declare_parameter("seed", 0, _NUMERIC)

        map_yaml = self.get_parameter("map_yaml").value
        if not map_yaml:
            raise RuntimeError("map_yaml parameter is required")

        grid = load_occupancy_grid(map_yaml)
        seed = int(self.get_parameter("seed").value)
        self.seed = seed
        spawn_rng = np.random.default_rng(seed if seed else None)
        robot_radius = float(self.get_parameter("robot_radius_m").value)
        try:
            spawn = sample_free_pose(grid, robot_radius, spawn_rng)
        except ValueError as exc:
            self.get_logger().fatal(f"Cannot spawn on {map_yaml}: {exc}")
            raise

        config = SimConfig(
            robot_radius_m=robot_radius,
            max_linear_vel=float(self.get_parameter("max_linear_vel").value),
            max_angular_vel=float(self.get_parameter("max_angular_vel").value),
            laser_max_range_m=float(self.get_parameter("laser_max_range_m").value),
            laser_num_rays=int(self.get_parameter("laser_num_rays").value),
            laser_range_noise_std=float(self.get_parameter("laser_range_noise_std").value),
            odom_linear_noise_std=float(self.get_parameter("odom_linear_noise_std").value),
            odom_angular_noise_std=float(self.get_parameter("odom_angular_noise_std").value),
            seed=seed if seed else None,
        )
        self.sim = SimCore(grid, spawn, config)

        self.time_scale = float(self.get_parameter("time_scale").value)
        if self.time_scale <= 0:
            raise RuntimeError("time_scale must be > 0")
        self.sim_time_s = 0.0

        self.get_logger().info(
            f"Loaded {map_yaml} — spawned at world ({spawn[0]:.2f}, {spawn[1]:.2f}); "
            f"{len(self.sim.reachable_free)} reachable free cells. time_scale={self.time_scale}x"
        )

        self.last_cmd = (0.0, 0.0)
        self.last_cmd_time = time.monotonic()
        self.wall_start_time = time.monotonic()
        self.finished = False

        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.tf_broadcaster = TransformBroadcaster(self)

        # sim_period_s / scan_period_s are sim-time intervals, fixed regardless
        # of time_scale — only how often each tick fires in real time changes.
        self.sim_period_s = float(self.get_parameter("sim_period_s").value)
        self.scan_period_s = float(self.get_parameter("scan_period_s").value)
        self.sim_timer = self.create_timer(self.sim_period_s / self.time_scale, self._on_sim_tick)
        self.scan_timer = self.create_timer(self.scan_period_s / self.time_scale, self._on_scan_tick)

        self.finish_srv = self.create_service(Trigger, "/finish_exploration", self._on_finish)

        # Subscribed for the whole session so the latest /map is already in
        # hand at finalize time. QoS matches slam_toolbox's publish QoS
        # (TRANSIENT_LOCAL) so a late subscriber still gets the last map.
        self.latest_map_msg: OccupancyGridMsg | None = None
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_sub = self.create_subscription(OccupancyGridMsg, "/map", self._on_map, map_qos)

    # ------------------------------------------------------------- callbacks

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.last_cmd = (msg.linear.x, msg.angular.z)
        self.last_cmd_time = time.monotonic()

    def _on_sim_tick(self) -> None:
        if self.finished:
            return

        v, w = self.last_cmd
        # real-time check (not sim-time): detects the candidate's own process going unresponsive
        if time.monotonic() - self.last_cmd_time > float(self.get_parameter("cmd_vel_timeout_s").value):
            v, w = 0.0, 0.0

        self.sim.step(v, w, self.sim_period_s)
        self.sim_time_s += self.sim_period_s

        time_limit_s = float(self.get_parameter("time_limit_s").value)
        if self.sim_time_s >= time_limit_s:
            self.get_logger().warn(
                f"Time limit reached ({time_limit_s:.0f}s of sim time) without a call to /finish_exploration."
            )
            self._finalize(timed_out=True)
            return

        self.clock_pub.publish(Clock(clock=seconds_to_time_msg(self.sim_time_s)))
        self._publish_odom_and_tf()

    def _publish_odom_and_tf(self) -> None:
        now = seconds_to_time_msg(self.sim_time_s)
        pose = self.sim.odom_pose

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = pose.x
        odom.pose.pose.position.y = pose.y
        odom.pose.pose.orientation = yaw_to_quaternion(pose.theta)
        odom.twist.twist.linear.x = self.last_cmd[0]
        odom.twist.twist.angular.z = self.last_cmd[1]
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = pose.x
        tf.transform.translation.y = pose.y
        tf.transform.rotation = yaw_to_quaternion(pose.theta)
        self.tf_broadcaster.sendTransform(tf)

    def _on_scan_tick(self) -> None:
        if self.finished:
            return
        scan = self.sim.scan()
        msg = LaserScan()
        msg.header.stamp = seconds_to_time_msg(self.sim_time_s)
        msg.header.frame_id = "base_link"  # laser co-located with base_link, zero offset
        msg.angle_min = scan.angle_min
        msg.angle_max = scan.angle_max
        msg.angle_increment = scan.angle_increment
        msg.range_min = scan.range_min
        msg.range_max = scan.range_max
        msg.ranges = [float(r) for r in scan.ranges]
        self.scan_pub.publish(msg)

    def _on_map(self, msg: OccupancyGridMsg) -> None:
        self.latest_map_msg = msg

    def _on_finish(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.finished:
            response.success = False
            response.message = "Session already finished."
            return response
        report = self._finalize(timed_out=False)
        response.success = bool(report["success"])
        response.message = (
            f"coverage={report['coverage_fraction']:.1%} "
            f"distance_to_home={report['distance_to_home_m']:.2f}m "
            f"elapsed_sim={report['elapsed_time_s']:.1f}s "
            f"elapsed_wall={report['elapsed_wall_time_s']:.1f}s "
            f"collisions={report['collision_count']} "
            f"success={report['success']} "
            f"report_written_to={report['results_path']}"
        )
        return response

    # ---------------------------------------------------------------- report

    @staticmethod
    def _grid_from_map_msg(msg: OccupancyGridMsg) -> MapGrid:
        """Convert a live nav_msgs/OccupancyGrid (slam_toolbox's /map) into
        map_io's OccupancyGrid, for reuse with save_occupancy_grid(). Mirrors
        nav2_map_server's default free/occupied thresholds (25/65%)."""
        width, height = msg.info.width, msg.info.height
        raw = np.array(msg.data, dtype=np.int16).reshape((height, width))
        data = np.full((height, width), UNKNOWN, dtype=np.uint8)
        data[(raw >= 0) & (raw <= 25)] = FREE
        data[raw >= 65] = OCCUPIED
        # nav_msgs/OccupancyGrid: row 0 = min world y. map_io.OccupancyGrid: row 0 = max world y. Flip to match.
        data = np.flipud(data)
        origin_pos = msg.info.origin.position
        q = msg.info.origin.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return MapGrid(data=data, resolution=msg.info.resolution, origin=(origin_pos.x, origin_pos.y, yaw))

    def _finalize(self, timed_out: bool) -> dict:
        self.finished = True
        coverage = self.sim.coverage_fraction
        distance_home = self.sim.distance_to_home
        min_coverage = float(self.get_parameter("min_coverage_required").value)
        tolerance = float(self.get_parameter("return_tolerance_m").value)
        success = (not timed_out) and coverage >= min_coverage and distance_home <= tolerance

        report = {
            "map_yaml": str(self.get_parameter("map_yaml").value),
            # 0 means "random spawn, different each run" (see README) — not
            # reproducible after the fact; any other value is the exact seed
            # that produced this run's spawn point.
            "seed": self.seed,
            "coverage_fraction": float(coverage),
            "distance_to_home_m": float(distance_home),
            "elapsed_time_s": float(self.sim_time_s),
            "elapsed_wall_time_s": float(time.monotonic() - self.wall_start_time),
            "time_scale": self.time_scale,
            "collision_count": int(self.sim.collision_count),
            "timed_out": bool(timed_out),
            "min_coverage_required": min_coverage,
            "return_tolerance_m": tolerance,
            "success": bool(success),
        }

        results_path = Path(str(self.get_parameter("results_path").value)).resolve()
        report["results_path"] = str(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # save the candidate's own SLAM-built map (not ground truth) next to the report
        map_base_path = results_path.parent / "map"
        if self.latest_map_msg is not None:
            save_occupancy_grid(self._grid_from_map_msg(self.latest_map_msg), map_base_path.with_suffix(".yaml"))
            report["map_output_yaml"] = str(map_base_path.with_suffix(".yaml"))
        else:
            self.get_logger().warn("No /map message received yet — skipping map.pgm/map.yaml export")
            report["map_output_yaml"] = None

        with open(results_path, "w") as f:
            yaml.safe_dump(report, f, default_flow_style=False)

        self.get_logger().info(f"SESSION FINISHED — wrote {results_path}\n{yaml.safe_dump(report)}")
        return report


def main() -> None:
    rclpy.init()
    node = SimNode()
    # This node is the /clock publisher, so it must not run on sim time itself.
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, False)])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
