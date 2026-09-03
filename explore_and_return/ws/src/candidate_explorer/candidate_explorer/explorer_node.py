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
from nav2_msgs.action import NavigateToPose, ComputePathToPose
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
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

        self.obstacle_distance_map = None
        self.obstacle_distance_map_version = 0
        self.map_version = 0

        self._path_check_in_progress = False

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
        self.obstacle_clearance = 0.4  # metres
        self.latest_map: OccupancyGrid | None = None
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self._on_map,
            10
        )


        self.path_client = ActionClient(
            self,
            ComputePathToPose,
            "/compute_path_to_pose"
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
        self.map_version += 1

    def build_obstacle_distance_map(self):
        if self.latest_map is None:
            return None

        msg = self.latest_map

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        data = msg.data

        clearance_cells = math.ceil(
            self.obstacle_clearance / resolution
        )

        distance = [-1] * (width * height)
        queue = deque()

        # Multi-source BFS:
        # every occupied cell is a source with distance 0.
        for i, value in enumerate(data):
            if value > 20:
                distance[i] = 0
                queue.append(i)

        while queue:

            i = queue.popleft()
            d = distance[i]

            # We don't care about distances beyond the required
            # obstacle clearance.
            if d >= clearance_cells:
                continue

            next_d = d + 1

            x = i % width
            y = i // width

            # Left
            if x > 0:
                n = i - 1

                if distance[n] == -1:
                    distance[n] = next_d
                    queue.append(n)

            # Right
            if x < width - 1:
                n = i + 1

                if distance[n] == -1:
                    distance[n] = next_d
                    queue.append(n)

            # Down
            if y > 0:
                n = i - width

                if distance[n] == -1:
                    distance[n] = next_d
                    queue.append(n)

            # Up
            if y < height - 1:
                n = i + width

                if distance[n] == -1:
                    distance[n] = next_d
                    queue.append(n)

        return distance

    def get_home_pose_in_map_frame(self) -> PoseStamped | None:
        try:
            t = self.tf_buffer.lookup_transform("map", "odom", rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"no map->odom transform yet: {ex}")
            return None
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.orientation = t.transform.rotation
        return pose


    def compute_global_path_async(
        self,
        goal_pose: PoseStamped,
        on_done
    ) -> None:

        if not self.path_client.server_is_ready():
            self.get_logger().warn(
                "/compute_path_to_pose action server not ready"
            )
            on_done(None)
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(
                f"Could not get robot pose for path planning: {exc}"
            )
            on_done(None)
            return

        goal = ComputePathToPose.Goal()

        goal.goal = goal_pose
        goal.start = PoseStamped()
        goal.start.header.frame_id = "map"
        goal.start.header.stamp = self.get_clock().now().to_msg()

        goal.start.pose.position.x = (
            tf.transform.translation.x
        )
        goal.start.pose.position.y = (
            tf.transform.translation.y
        )
        goal.start.pose.orientation = (
            tf.transform.rotation
        )

        goal.planner_id = "GridBased"
        goal.use_start = True

        send_future = self.path_client.send_goal_async(goal)

        def _goal_response(future):

            try:
                handle = future.result()
            except Exception as exc:
                self.get_logger().error(
                    f"ComputePathToPose failed: {exc}"
                )
                on_done(None)
                return

            if not handle.accepted:
                self.get_logger().warn(
                    "ComputePathToPose goal rejected"
                )
                on_done(None)
                return

            result_future = handle.get_result_async()

            def _result_response(fut):

                try:
                    result = fut.result()
                except Exception as exc:
                    self.get_logger().error(
                        f"Path computation result failed: {exc}"
                    )
                    on_done(None)
                    return

                if result.status != GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().warn(
                        f"Path computation failed, "
                        f"status={result.status}"
                    )
                    on_done(None)
                    return

                path = result.result.path

                if path is None or not path.poses:
                    self.get_logger().warn(
                        "Nav2 returned an empty path"
                    )
                    on_done(None)
                    return

                self.get_logger().debug(
                    f"Computed global path with "
                    f"{len(path.poses)} poses"
                )

                on_done(path)

            result_future.add_done_callback(_result_response)

        send_future.add_done_callback(_goal_response)

    def path_has_narrow_passage(self, path) -> bool:
        if self.latest_map is None:
            return False

        poses = path.poses

        # Check every few poses instead of every single one.
        for i in range(1, len(poses) - 1, 3):

            p = poses[i].pose.position
            prev = poses[i - 1].pose.position
            nxt = poses[i + 1].pose.position

            yaw = math.atan2(
                nxt.y - prev.y,
                nxt.x - prev.x
            )

            width = self.get_cross_section_width(
                p.x,
                p.y,
                yaw
            )

            if (
                width is not None
                and width < self.min_passage_width
            ):
                self.get_logger().warn(
                    f"Narrow passage detected: {width:.2f} m"
                )
                return True

        return False

    def get_cross_section_width(
        self,
        x: float,
        y: float,
        yaw: float
    ):
        if self.latest_map is None:
            return None

        msg = self.latest_map

        res = msg.info.resolution
        width = msg.info.width
        height = msg.info.height
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        data = msg.data

        perp_x = -math.sin(yaw)
        perp_y = math.cos(yaw)

        max_scan = 1.0

        def scan(sign):

            d = res

            while d <= max_scan:

                wx = x + sign * perp_x * d
                wy = y + sign * perp_y * d

                mx = int((wx - ox) / res)
                my = int((wy - oy) / res)

                if (
                    mx < 0 or mx >= width
                    or my < 0 or my >= height
                ):
                    # Outside the known map is not considered
                    # a narrow wall.
                    return max_scan

                value = data[my * width + mx]

                if value > 20:
                    return d

                d += res

            return max_scan

        left = scan(+1)
        right = scan(-1)

        return left + right

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

        if (
            self.obstacle_distance_map is None
            or self.obstacle_distance_map_version != self.map_version
        ):
            self.get_logger().debug(
                "Rebuilding obstacle distance map..."
            )

            self.obstacle_distance_map = (
                self.build_obstacle_distance_map()
            )

            self.obstacle_distance_map_version = self.map_version

        distance_map = self.obstacle_distance_map

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


                clearance_cells = math.ceil(
                    self.obstacle_clearance / resolution
                )

                if (
                    distance_map[i] >= 0
                    and distance_map[i] < clearance_cells
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
                if distance <= 0.45:
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
            f"distance={closest_distance:.2f}"
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

            # --------------------------------------------------------
            # A navigation goal is currently running.
            #
            # We deliberately do NOT perform local narrow-passage
            # detection here anymore.
            #
            # The path was checked BEFORE NavigateToPose was sent.
            # --------------------------------------------------------

            if self._goal_in_progress:
                return

            # --------------------------------------------------------
            # A ComputePathToPose request is currently running.
            #
            # Wait for its callback.
            # --------------------------------------------------------

            if self._path_check_in_progress:
                return

            # --------------------------------------------------------
            # Find the best frontier.
            # --------------------------------------------------------

            frontier = self.find_nearest_frontier()

            if frontier is not None:
                self.no_frontier_since = None

            # --------------------------------------------------------
            # No frontier found.
            # --------------------------------------------------------

            if frontier is None:

                now = self.get_clock().now()

                if self.no_frontier_since is None:

                    self.no_frontier_since = now

                    self.get_logger().info(
                        "No usable frontier found. "
                        f"Waiting up to "
                        f"{self.no_frontier_timeout:.1f}s "
                        "for the map to update."
                    )

                    return

                elapsed = (
                    now - self.no_frontier_since
                ).nanoseconds / 1e9

                if elapsed < self.no_frontier_timeout:

                    self.get_logger().debug(
                        f"No frontier for {elapsed:.1f}s "
                        f"(timeout="
                        f"{self.no_frontier_timeout:.1f}s)"
                    )

                    return

                self.get_logger().info(
                    "No usable frontier found for "
                    f"{self.no_frontier_timeout:.1f}s. "
                    "Exploration complete; returning home."
                )

                self.state = "RETURNING"
                self.no_frontier_since = None

                return

            # --------------------------------------------------------
            # Frontier coordinates.
            # --------------------------------------------------------

            x, y = frontier

            # --------------------------------------------------------
            # Get current robot pose.
            # --------------------------------------------------------

            try:

                tf = self.tf_buffer.lookup_transform(
                    "map",
                    "base_link",
                    rclpy.time.Time()
                )

            except tf2_ros.TransformException as exc:

                self.get_logger().debug(
                    f"Waiting for map -> base_link TF: {exc}"
                )

                return

            robot_x = tf.transform.translation.x
            robot_y = tf.transform.translation.y

            distance = math.hypot(
                x - robot_x,
                y - robot_y
            )

            # --------------------------------------------------------
            # Ignore frontiers that are essentially under the robot.
            # --------------------------------------------------------

            if distance <= 0.45:

                self.get_logger().info(
                    f"Skipping frontier at "
                    f"({x:.2f}, {y:.2f}) because it is "
                    f"too close to the robot: {distance:.2f}m"
                )

                return

            now = self.get_clock().now()

            # --------------------------------------------------------
            # Check whether this is the same frontier we were
            # previously working on.
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

                self.narrow_entrance_count = 0

            # --------------------------------------------------------
            # SAME FRONTIER
            # --------------------------------------------------------

            else:

                # ----------------------------------------------------
                # Meaningful progress.
                # ----------------------------------------------------

                if (
                    distance
                    < self.current_frontier_distance - 0.10
                ):

                    self.current_frontier_distance = distance
                    self.last_progress_time = now

                    self.get_logger().info(
                        f"Progress toward frontier: "
                        f"{distance:.2f} m"
                    )

                # ----------------------------------------------------
                # No meaningful progress.
                #
                # Use a reasonably generous timeout because SLAM may
                # need time to publish a changed map.
                # ----------------------------------------------------

                else:

                    elapsed = (
                        now - self.last_progress_time
                    ).nanoseconds / 1e9

                    if elapsed > self.progress_timeout:

                        self.get_logger().warn(
                            f"No progress toward frontier "
                            f"({x:.2f}, {y:.2f}) for "
                            f"{elapsed:.1f}s. "
                            "Blacklisting it."
                        )

                        self.frontier_blacklist.append(
                            frontier
                        )

                        self.current_frontier = None
                        self.current_frontier_distance = float("inf")
                        self.last_progress_time = now

                        return

                    # ------------------------------------------------
                    # We already attempted this frontier and are
                    # waiting for the map/frontier structure to change.
                    # ------------------------------------------------

                    self.get_logger().debug(
                        f"Waiting for map update around frontier "
                        f"({x:.2f}, {y:.2f})"
                    )

                    return

            # --------------------------------------------------------
            # Construct NavigateToPose goal.
            # --------------------------------------------------------

            pose = PoseStamped()

            pose.header.frame_id = "map"
            pose.header.stamp = now.to_msg()

            pose.pose.position.x = x
            pose.pose.position.y = y

            # Face toward the unexplored region.
            yaw = math.atan2(
                y - robot_y,
                x - robot_x
            )

            pose.pose.orientation = yaw_to_quaternion(yaw)

            # --------------------------------------------------------
            # Remember exactly which frontier this path belongs to.
            #
            # This prevents an old asynchronous callback from sending
            # a navigation goal for a frontier that is no longer valid.
            # --------------------------------------------------------

            frontier_being_checked = frontier

            self._path_check_in_progress = True

            self.get_logger().info(
                f"Checking global path to frontier "
                f"({x:.2f}, {y:.2f}), "
                f"distance={distance:.2f}m"
            )

            # --------------------------------------------------------
            # Path result callback.
            # --------------------------------------------------------

            def _on_path_ready(path):

                self._path_check_in_progress = False

                # ----------------------------------------------------
                # Make sure this callback still belongs to the current
                # frontier.
                # ----------------------------------------------------

                if self.current_frontier is None:

                    self.get_logger().debug(
                        "Path result arrived after frontier was cleared."
                    )

                    return

                if (
                    math.hypot(
                        frontier_being_checked[0]
                        - self.current_frontier[0],

                        frontier_being_checked[1]
                        - self.current_frontier[1]
                    ) > 0.5
                ):

                    self.get_logger().debug(
                        "Discarding stale path result."
                    )

                    return

                # ----------------------------------------------------
                # Nav2 could not find a path.
                # ----------------------------------------------------

                if path is None:

                    self.get_logger().warn(
                        f"No global path to frontier "
                        f"({frontier_being_checked[0]:.2f}, "
                        f"{frontier_being_checked[1]:.2f}). "
                        "Blacklisting."
                    )

                    self.frontier_blacklist.append(
                        frontier_being_checked
                    )

                    self.current_frontier = None
                    self.current_frontier_distance = float("inf")

                    return

                # ----------------------------------------------------
                # Inspect the actual Nav2 path.
                # ----------------------------------------------------

                if self.path_has_narrow_passage(path):

                    self.get_logger().warn(
                        f"Rejecting frontier "
                        f"({frontier_being_checked[0]:.2f}, "
                        f"{frontier_being_checked[1]:.2f}): "
                        "global path contains a narrow passage."
                    )

                    self.frontier_blacklist.append(
                        frontier_being_checked
                    )

                    self.current_frontier = None
                    self.current_frontier_distance = float("inf")

                    return

                # ----------------------------------------------------
                # Path is acceptable.
                #
                # Now send NavigateToPose.
                # ----------------------------------------------------

                self.get_logger().info(
                    f"Global path accepted. "
                    f"Navigating to frontier "
                    f"({frontier_being_checked[0]:.2f}, "
                    f"{frontier_being_checked[1]:.2f})"
                )

                def _on_done(
                    success: bool,
                    goal=frontier_being_checked
                ):

                    self.get_logger().info(
                        f"Frontier goal finished: "
                        f"success={success}"
                    )

                    # ------------------------------------------------
                    # Failed navigation.
                    # ------------------------------------------------

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
                        self.last_progress_time = (
                            self.get_clock().now()
                        )

                        return

                    # ------------------------------------------------
                    # Navigation succeeded.
                    #
                    # Keep the frontier temporarily so that we don't
                    # immediately repeat it.
                    #
                    # The next find_nearest_frontier() call will normally
                    # select a newly exposed frontier after SLAM updates.
                    # ------------------------------------------------

                    self.last_progress_time = (
                        self.get_clock().now()
                    )

                    self.current_frontier_distance = 0.0

                    self.get_logger().info(
                        "Reached frontier. "
                        "Waiting for the map to expose new frontiers."
                    )

                self.send_nav_goal(
                    pose,
                    _on_done
                )

            # --------------------------------------------------------
            # Start asynchronous path computation.
            # --------------------------------------------------------

            self.compute_global_path_async(
                pose,
                _on_path_ready
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