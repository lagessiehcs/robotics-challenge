"""Your task: replace the EXPLORING state's body (marked below) with real
exploration + coverage logic. Everything else in this file is infrastructure
you're free to use as-is, restructure, or throw away entirely — the only
external contract that matters is:

  - subscribe /map (nav_msgs/OccupancyGrid) — published live by slam_toolbox
  - send goals via the /navigate_to_pose action (nav2_msgs/action/NavigateToPose)
  - call the /finish_exploration service (std_srvs/srv/Trigger) when done

See README.md for the full brief and scoring rubric.

Note: "home" is the odom frame's origin (see get_home_pose_in_map_frame
below) — re-derive its pose in the map frame every time you need it, don't
cache it from t=0. slam_toolbox corrects the map->odom transform as it
refines the pose graph, so where home sits in the map frame can drift over a
run even though the robot itself never moves in the odom frame.
"""
from __future__ import annotations

import math
from collections import deque

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import Trigger


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class ExplorerNode(Node):
    def __init__(self) -> None:
        super().__init__("candidate_explorer")

        self.no_frontier_since = None
        self.no_frontier_timeout = 10.0

        # Minimum passage/entrance width that the robot is willing to enter.
        self.min_passage_width = 0.5  # metres

        # Number of consecutive detections required before blacklisting.
        self.narrow_entrance_required = 2
        self.narrow_entrance_count = 0

        # Store the active Nav2 goal handle so we can cancel it.
        self._goal_handle = None

        # Minimum distance a frontier goal must have from an obstacle.
        self.obstacle_clearance = 0.1  # metres
        self.latest_map: OccupancyGrid | None = None
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self._on_map,
            10
        )

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose"
        )

        self.finish_client = self.create_client(
            Trigger,
            "/finish_exploration"
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.state = "WAITING_FOR_MAP"
        self._goal_in_progress = False

        # ------------------------------------------------------------
        # Frontier progress tracking.
        #
        # This prevents repeatedly sending the exact same frontier goal
        # when Nav2 reports success but the map has not changed enough.
        # ------------------------------------------------------------

        self.current_frontier = None
        self.current_frontier_distance = float("inf")

        self.last_progress_time = self.get_clock().now()

        # Simulated seconds without meaningful progress before we
        # abandon the current frontier.
        self.progress_timeout = 2.0

        # Frontiers that repeatedly fail or make no progress.
        self.frontier_blacklist = []

        self.timer = self.create_timer(1.0, self._tick)

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def get_local_passage_width(
        self,
        robot_x: float,
        robot_y: float,
        direction_yaw: float,
    ) -> float | None:
        """
        Estimate the free-space width around the robot perpendicular
        to its direction of travel.

        Returns the width in metres, or None if it cannot be determined.
        """

        if self.latest_map is None:
            return None

        msg = self.latest_map

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        data = msg.data

        # Direction perpendicular to travel.
        perp_x = -math.sin(direction_yaw)
        perp_y = math.cos(direction_yaw)

        # Search left/right until we hit an obstacle.
        max_search = 2.0
        step = resolution

        left_distance = max_search
        right_distance = max_search

        # ------------------------------------------------------------
        # Search left.
        # ------------------------------------------------------------

        d = 0.0

        while d <= max_search:

            x = robot_x + perp_x * d
            y = robot_y + perp_y * d

            mx = int((x - origin_x) / resolution)
            my = int((y - origin_y) / resolution)

            if (
                mx < 0 or mx >= width
                or my < 0 or my >= height
            ):
                left_distance = d
                break

            value = data[my * width + mx]

            # Occupied.
            if value > 20:
                left_distance = d
                break

            # Unknown space is treated as a boundary too.
            if value < 0:
                left_distance = d
                break

            d += step

        # ------------------------------------------------------------
        # Search right.
        # ------------------------------------------------------------

        d = 0.0

        while d <= max_search:

            x = robot_x - perp_x * d
            y = robot_y - perp_y * d

            mx = int((x - origin_x) / resolution)
            my = int((y - origin_y) / resolution)

            if (
                mx < 0 or mx >= width
                or my < 0 or my >= height
            ):
                right_distance = d
                break

            value = data[my * width + mx]

            if value > 20:
                right_distance = d
                break

            if value < 0:
                right_distance = d
                break

            d += step

        return left_distance + right_distance

    def get_home_pose_in_map_frame(self) -> PoseStamped | None:
        try:
            t = self.tf_buffer.lookup_transform(
                "map",
                "odom",
                rclpy.time.Time()
            )
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(
                f"no map->odom transform yet: {ex}"
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.orientation = t.transform.rotation

        return pose

    def send_nav_goal(self, pose: PoseStamped, on_done) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                "/navigate_to_pose action server not available"
            )
            return

        goal = NavigateToPose.Goal()
        goal.pose = pose

        self._goal_in_progress = True

        send_future = self.nav_client.send_goal_async(goal)

        def _on_goal_response(fut):
            handle = fut.result()

            if not handle.accepted:
                self.get_logger().warn("goal rejected")
                self._goal_handle = None
                self._goal_in_progress = False
                on_done(False)
                return

            self._goal_handle = handle

            result_future = handle.get_result_async()

            def _on_result(fut2):
                status = fut2.result().status

                self._goal_handle = None
                self._goal_in_progress = False

                on_done(
                    status == GoalStatus.STATUS_SUCCEEDED
                )

            result_future.add_done_callback(_on_result)

        send_future.add_done_callback(_on_goal_response)

    def call_finish_exploration(self) -> None:
        if not self.finish_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                "/finish_exploration service not available"
            )
            return

        future = self.finish_client.call_async(
            Trigger.Request()
        )

        def _on_response(fut):
            res = fut.result()
            self.get_logger().info(
                f"Session result: {res.message}"
            )

        future.add_done_callback(_on_response)

    def frontier_is_blacklisted(self, frontier) -> bool:
        """Return True if this frontier is already known to be bad."""

        for old_frontier in self.frontier_blacklist:
            distance = math.hypot(
                frontier[0] - old_frontier[0],
                frontier[1] - old_frontier[1]
            )

            # Blacklist radius of 0.5 m.
            if distance < 0.8:
                return True

        return False

    def frontier_is_too_close_to_obstacle(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        resolution: float,
        data,
    ) -> bool:
        """
        Return True if the frontier cell is too close to an occupied cell.

        Occupied cells are those with occupancy > 20.
        Unknown cells are ignored here.
        """

        clearance_cells = int(
            math.ceil(self.obstacle_clearance / resolution)
        )

        for dy in range(-clearance_cells, clearance_cells + 1):
            for dx in range(-clearance_cells, clearance_cells + 1):

                # Don't use a square clearance region; use a circle.
                distance = math.hypot(
                    dx * resolution,
                    dy * resolution
                )

                if distance > self.obstacle_clearance:
                    continue

                nx = x + dx
                ny = y + dy

                # Outside map.
                if (
                    nx < 0
                    or nx >= width
                    or ny < 0
                    or ny >= height
                ):
                    continue

                i = ny * width + nx

                # Occupied.
                if data[i] > 20:
                    return True

        return False

    def find_nearest_frontier(self):
        """Find the best frontier cluster and return an actual goal point."""

        if self.latest_map is None:
            return None

        msg = self.latest_map

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        data = msg.data

        # ------------------------------------------------------------
        # Current robot position.
        # ------------------------------------------------------------

        try:
            tf = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )
        except tf2_ros.TransformException:
            return None

        robot_x = tf.transform.translation.x
        robot_y = tf.transform.translation.y

        # ------------------------------------------------------------
        # 1. Find all frontier cells.
        # ------------------------------------------------------------

        frontier_cells = set()

        for y in range(1, height - 1):
            for x in range(1, width - 1):

                i = y * width + x

                # Known free cell.
                if data[i] < 0 or data[i] > 20:
                    continue

                neighbors = (
                    i - 1,
                    i + 1,
                    i - width,
                    i + width,
                )

                # Must touch unknown space.
                if not any(data[n] == -1 for n in neighbors):
                    continue

                # ------------------------------------------------------------
                # Reject frontiers that are too close to an obstacle.
                # ------------------------------------------------------------

                if self.frontier_is_too_close_to_obstacle(
                    x,
                    y,
                    width,
                    height,
                    resolution,
                    data,
                ):
                    continue

                frontier_cells.add((x, y))

        if not frontier_cells:
            self.get_logger().warn(
                "No frontier cells detected!"
            )
            return None

        self.get_logger().info(
            f"Detected {len(frontier_cells)} frontier cells"
        )

        # ------------------------------------------------------------
        # 2. Cluster frontier cells.
        # ------------------------------------------------------------

        clusters = []
        visited = set()

        for start in frontier_cells:

            if start in visited:
                continue

            cluster = []
            queue = deque([start])
            visited.add(start)

            while queue:

                x, y = queue.popleft()
                cluster.append((x, y))

                # 8-connected clustering.
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):

                        if dx == 0 and dy == 0:
                            continue

                        neighbor = (x + dx, y + dy)

                        if (
                            neighbor in frontier_cells
                            and neighbor not in visited
                        ):
                            visited.add(neighbor)
                            queue.append(neighbor)

            clusters.append(cluster)

        # ------------------------------------------------------------
        # 3. Remove tiny clusters.
        # ------------------------------------------------------------

        MIN_CLUSTER_SIZE = 0.5

        clusters = [
            cluster
            for cluster in clusters
            if len(cluster) >= MIN_CLUSTER_SIZE
        ]

        self.get_logger().info(
            f"Found {len(clusters)} frontier clusters "
            f"after filtering"
        )

        if not clusters:
            return None

        # ------------------------------------------------------------
        # 4. Find best cluster.
        # ------------------------------------------------------------

        best_cluster = None
        best_goal = None
        best_score = float("inf")

        for cluster in clusters:

            closest_frontier = None
            closest_distance = float("inf")

            # --------------------------------------------------------
            # Find the closest usable frontier CELL.
            # --------------------------------------------------------

            for x, y in cluster:

                wx = origin_x + (x + 0.5) * resolution
                wy = origin_y + (y + 0.5) * resolution

                distance = math.hypot(
                    wx - robot_x,
                    wy - robot_y
                )

                # Don't choose a frontier we're already on.
                if distance < 0.5:
                    continue

                # Ignore blacklisted frontier cells.
                if self.frontier_is_blacklisted(
                    (wx, wy)
                ):
                    continue

                if distance < closest_distance:
                    closest_distance = distance
                    closest_frontier = (wx, wy)

            # No usable cells in this cluster.
            if closest_frontier is None:
                continue

            # --------------------------------------------------------
            # Cluster score.
            #
            # Smaller distance = better
            # Larger cluster = better
            # --------------------------------------------------------

            score = (
                closest_distance
                / math.sqrt(len(cluster))
            )

            self.get_logger().debug(
                f"Cluster size={len(cluster)}, "
                f"distance={closest_distance:.2f}m, "
                f"score={score:.3f}"
            )

            if score < best_score:

                best_score = score
                best_cluster = cluster
                best_goal = closest_frontier

        # ------------------------------------------------------------
        # 5. No usable cluster.
        # ------------------------------------------------------------

        if best_goal is None:
            self.get_logger().warn(
                "Frontier clusters exist, but none have "
                "a usable goal point."
            )
            return None

        self.get_logger().info(
            f"Selected frontier cluster: "
            f"size={len(best_cluster)}, "
            f"goal=({best_goal[0]:.2f}, "
            f"{best_goal[1]:.2f}), "
            f"distance={best_score:.2f}"
        )

        return best_goal

    def _tick(self) -> None:

        # ============================================================
        # WAITING FOR MAP
        # ============================================================

        if self.state == "WAITING_FOR_MAP":

            if self.latest_map is not None:
                self.get_logger().info(
                    "Got first /map."
                )

                self.state = "EXPLORING"

            return

        # ============================================================
        # EXPLORING
        # ============================================================

        if self.state == "EXPLORING":

            # Don't do anything while Nav2 is executing a goal.
            if self._goal_in_progress:

                # ------------------------------------------------------------
                # Check whether we are approaching a passage that is too narrow.
                # ------------------------------------------------------------

                if self.current_frontier is not None:

                    try:
                        tf = self.tf_buffer.lookup_transform(
                            "map",
                            "base_link",
                            rclpy.time.Time()
                        )
                    except tf2_ros.TransformException:
                        return

                    robot_x = tf.transform.translation.x
                    robot_y = tf.transform.translation.y

                    frontier_x = self.current_frontier[0]
                    frontier_y = self.current_frontier[1]

                    direction_yaw = math.atan2(
                        frontier_y - robot_y,
                        frontier_x - robot_x
                    )

                    passage_width = self.get_local_passage_width(
                        robot_x,
                        robot_y,
                        direction_yaw
                    )

                    if passage_width is not None:

                        self.get_logger().debug(
                            f"Estimated passage width: "
                            f"{passage_width:.2f} m"
                        )

                        if passage_width < self.min_passage_width:

                            self.narrow_entrance_count += 1

                            self.get_logger().warn(
                                f"Narrow entrance detected: "
                                f"{passage_width:.2f} m "
                                f"(minimum="
                                f"{self.min_passage_width:.2f} m), "
                                f"count="
                                f"{self.narrow_entrance_count}"
                            )

                        else:
                            self.narrow_entrance_count = 0

                        # ----------------------------------------------------
                        # Require multiple detections so a single bad map cell
                        # doesn't blacklist a frontier accidentally.
                        # ----------------------------------------------------

                        if (
                            self.narrow_entrance_count
                            >= self.narrow_entrance_required
                        ):

                            frontier = self.current_frontier

                            self.get_logger().warn(
                                f"Blacklisting frontier "
                                f"({frontier[0]:.2f}, "
                                f"{frontier[1]:.2f}) because the "
                                f"entrance is too narrow."
                            )

                            self.frontier_blacklist.append(
                                frontier
                            )

                            self.current_frontier = None
                            self.current_frontier_distance = float("inf")
                            self.narrow_entrance_count = 0

                            # Cancel the active Nav2 goal.
                            if self._goal_handle is not None:
                                cancel_future = (
                                    self._goal_handle.cancel_goal_async()
                                )

                            return

                return

            # --------------------------------------------------------
            # Find the nearest usable frontier.
            # --------------------------------------------------------

            frontier = self.find_nearest_frontier()

            if frontier is not None:
                self.no_frontier_since = None

            if frontier is None:

                now = self.get_clock().now()

                # Start the "no frontier" timer.
                if self.no_frontier_since is None:
                    self.no_frontier_since = now

                    self.get_logger().info(
                        "No usable frontier found. "
                        "Continuing exploration for up to 10s."
                    )

                    return

                elapsed = (
                    now - self.no_frontier_since
                ).nanoseconds / 1e9

                # Still within the grace period.
                if elapsed < self.no_frontier_timeout:
                    self.get_logger().debug(
                        f"No frontier for {elapsed:.1f}s "
                        f"(timeout={self.no_frontier_timeout:.1f}s)"
                    )
                    return

                # No frontier for 10 seconds.
                self.get_logger().info(
                    "No usable frontier found for "
                    f"{self.no_frontier_timeout:.1f}s. "
                    "Exploration complete; returning home."
                )

                self.state = "RETURNING"
                self.no_frontier_since = None

                return

            x, y = frontier

            # --------------------------------------------------------
            # Get current robot position.
            # --------------------------------------------------------

            try:
                tf = self.tf_buffer.lookup_transform(
                    "map",
                    "base_link",
                    rclpy.time.Time()
                )
            except tf2_ros.TransformException:
                return

            robot_x = tf.transform.translation.x
            robot_y = tf.transform.translation.y

            distance = math.hypot(
                x - robot_x,
                y - robot_y
            )

            now = self.get_clock().now()

            # --------------------------------------------------------
            # Check whether this is the same frontier we were already
            # trying to reach.
            # --------------------------------------------------------

            same_frontier = False

            if self.current_frontier is not None:

                same_frontier = (
                    math.hypot(
                        x - self.current_frontier[0],
                        y - self.current_frontier[1]
                    ) < 0.5
                )

            # --------------------------------------------------------
            # NEW FRONTIER
            # --------------------------------------------------------

            if not same_frontier:

                self.current_frontier = frontier
                self.current_frontier_distance = distance
                self.last_progress_time = now

            # --------------------------------------------------------
            # SAME FRONTIER
            # --------------------------------------------------------

            else:

                # The robot has moved meaningfully closer.
                if distance < self.current_frontier_distance - 0.10:

                    self.current_frontier_distance = distance
                    self.last_progress_time = now

                    self.get_logger().info(
                        f"Progress toward frontier: "
                        f"{distance:.2f} m"
                    )

                # ----------------------------------------------------
                # No meaningful progress for too long.
                # ----------------------------------------------------

                elif (
                    (
                        now - self.last_progress_time
                    ).nanoseconds / 1e9
                    > self.progress_timeout
                ):

                    self.get_logger().warn(
                        f"No progress toward frontier "
                        f"({x:.2f}, {y:.2f}) for "
                        f"{self.progress_timeout:.0f}s. "
                        f"Blacklisting it."
                    )

                    self.frontier_blacklist.append(
                        frontier
                    )

                    self.current_frontier = None
                    self.current_frontier_distance = float("inf")
                    self.last_progress_time = now

                    return

                # ----------------------------------------------------
                # IMPORTANT:
                #
                # If Nav2 already succeeded at this frontier, do NOT
                # send the exact same goal again.
                #
                # Wait for the frontier to disappear/change or for
                # the progress timeout to blacklist it.
                # ----------------------------------------------------

                else:

                    self.get_logger().debug(
                        f"Waiting for map update at frontier "
                        f"({x:.2f}, {y:.2f})"
                    )

                    return

            # --------------------------------------------------------
            # Send goal.
            # --------------------------------------------------------

            pose = PoseStamped()

            pose.header.frame_id = "map"
            pose.header.stamp = now.to_msg()

            pose.pose.position.x = x
            pose.pose.position.y = y

            # Face toward the frontier.
            yaw = math.atan2(
                y - robot_y,
                x - robot_x
            )

            pose.pose.orientation = yaw_to_quaternion(yaw)

            self.get_logger().info(
                f"Navigating to frontier "
                f"({x:.2f}, {y:.2f}), "
                f"distance={distance:.2f}m"
            )

            def _on_done(success: bool, goal=frontier) -> None:

                self.get_logger().info(
                    f"frontier goal finished, "
                    f"success={success}"
                )

                # ----------------------------------------------------
                # If Nav2 failed, this frontier is probably unreachable.
                # Don't keep trying it forever.
                # ----------------------------------------------------

                if not success:

                    self.get_logger().warn(
                        f"Blacklisting failed frontier "
                        f"({goal[0]:.2f}, {goal[1]:.2f})"
                    )

                    self.frontier_blacklist.append(
                        goal
                    )

                    self.current_frontier = None
                    self.current_frontier_distance = float("inf")

                # If successful, DON'T reset current_frontier.
                #
                # We want _tick() to notice that this is still the
                # same frontier and wait for the map to change.
                #
                # If it never changes, the progress timeout will
                # blacklist it.

                self.state = "EXPLORING"

            self.send_nav_goal(
                pose,
                _on_done
            )

            return

        # ============================================================
        # RETURNING HOME
        # ============================================================

        if self.state == "RETURNING":

            if self._goal_in_progress:
                return

            # IMPORTANT:
            #
            # Recalculate home every time.
            # slam_toolbox can change map -> odom while exploring.
            #
            home = self.get_home_pose_in_map_frame()

            if home is None:
                return

            def _on_done(success: bool) -> None:

                self.get_logger().info(
                    f"return-home goal finished, "
                    f"success={success}"
                )

                self.state = "FINISHING"

            self.send_nav_goal(
                home,
                _on_done
            )

            return

        # ============================================================
        # FINISHING
        # ============================================================

        if self.state == "FINISHING":

            self.state = "DONE"

            self.call_finish_exploration()

            return

        # ============================================================
        # DONE
        # ============================================================

        # DONE: nothing left to do.


def main() -> None:
    rclpy.init()

    node = ExplorerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()