"""Launches the provided infrastructure for the explore-and-return challenge:
the sim (challenge_sim), live SLAM (slam_toolbox), and navigation (Nav2's
navigation_launch.py — not localization, since slam_toolbox does that live).

The candidate's own node (candidate_explorer) is launched separately — see
README.md. Nav2's behavior tree and costmap config are candidate-editable:
they're installed from candidate_explorer/config — see those files.

Usage:
    ros2 launch challenge_sim challenge.launch.py map_yaml:=/path/to/room.yaml
"""
import os
import tempfile
from datetime import datetime

import launch
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


class MergeTopLevelYaml(launch.Substitution):
    """Overlay a YAML file's top-level keys onto a base YAML file.

    Unlike RewrittenYaml (which rewrites individual leaf values), this
    replaces whole top-level blocks wholesale — lets costmap_params.yaml
    freely restructure local_costmap/global_costmap instead of just
    swapping scalar values. Writes the merged result to a temp file and
    returns its path.
    """

    def __init__(self, base_file, overlay_file):
        super().__init__()
        from launch.utilities import normalize_to_list_of_substitutions

        self.__base_file = normalize_to_list_of_substitutions(base_file)
        self.__overlay_file = normalize_to_list_of_substitutions(overlay_file)

    def describe(self):
        return ""

    def perform(self, context):
        base_path = launch.utilities.perform_substitutions(context, self.__base_file)
        overlay_path = launch.utilities.perform_substitutions(context, self.__overlay_file)
        with open(base_path) as f:
            data = yaml.safe_load(f)
        with open(overlay_path) as f:
            overlay = yaml.safe_load(f) or {}
        data.update(overlay)
        merged = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml")
        yaml.dump(data, merged)
        merged.close()
        return merged.name


def generate_launch_description():
    pkg_share = get_package_share_directory("challenge_sim")
    nav2_bringup_share = get_package_share_directory("nav2_bringup")
    slam_toolbox_share = get_package_share_directory("slam_toolbox")
    candidate_share = get_package_share_directory("candidate_explorer")

    map_yaml = LaunchConfiguration("map_yaml")
    seed = LaunchConfiguration("seed")
    time_limit_s = LaunchConfiguration("time_limit_s")
    time_scale = LaunchConfiguration("time_scale")
    results_path = LaunchConfiguration("results_path")
    rviz = LaunchConfiguration("rviz")
    bt_xml = LaunchConfiguration("bt_xml")
    costmap_params = LaunchConfiguration("costmap_params")

    declare_map_yaml = DeclareLaunchArgument(
        "map_yaml", description="Path to a room.yaml (the ground-truth map, hidden from the candidate)"
    )
    declare_seed = DeclareLaunchArgument("seed", default_value="0", description="0 = random spawn each run")
    declare_time_limit = DeclareLaunchArgument(
        "time_limit_s", default_value="5400.0", description="SIM-time budget in seconds (default: 1.5h)"
    )
    declare_time_scale = DeclareLaunchArgument(
        "time_scale",
        default_value="4.0",
        description="How much faster than real time the whole stack runs. 1.0 = real time.",
    )
    # timestamped by default so repeat runs don't clobber each other
    declare_results_path = DeclareLaunchArgument(
        "results_path",
        default_value=f"results/{datetime.now():%Y%m%d_%H%M%S}/report.yaml",
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz", default_value="false", description="Launch RViz2 to watch the robot/map/coverage live"
    )
    declare_bt_xml = DeclareLaunchArgument(
        "bt_xml",
        default_value=os.path.join(candidate_share, "config", "navigate_to_pose.xml"),
        description="Nav2 behavior tree — a candidate-editable copy of Nav2's stock default",
    )
    declare_costmap_params = DeclareLaunchArgument(
        "costmap_params",
        default_value=os.path.join(candidate_share, "config", "costmap_params.yaml"),
        description=(
            "local_costmap/global_costmap params — a candidate-editable file, merged "
            "on top of nav2_params.yaml at launch time (see costmap_params.yaml)"
        ),
    )

    sim_node = Node(
        package="challenge_sim",
        executable="sim_node",
        name="challenge_sim_node",
        output="screen",
        parameters=[
            {
                "map_yaml": map_yaml,
                "seed": seed,
                "time_limit_s": time_limit_s,
                "time_scale": time_scale,
                "results_path": results_path,
            }
        ],
    )

    # Layer the candidate-editable costmap blocks onto the static
    # nav2_params.yaml, then the candidate-editable BT path on top of that.
    nav2_params_with_costmap = MergeTopLevelYaml(
        base_file=os.path.join(pkg_share, "config", "nav2_params.yaml"),
        overlay_file=costmap_params,
    )
    nav2_params_with_bt = RewrittenYaml(
        source_file=nav2_params_with_costmap,
        root_key="",
        param_rewrites={"default_nav_to_pose_bt_xml": bt_xml},
        convert_types=False,  # this is a file path, not a number/bool
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_share, "launch", "online_async_launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "slam_params_file": os.path.join(pkg_share, "config", "slam_toolbox_params.yaml"),
        }.items(),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_share, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "params_file": nav2_params_with_bt,
            "autostart": "true",
        }.items(),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=["-d", os.path.join(pkg_share, "config", "challenge.rviz")],
        condition=IfCondition(rviz),
    )

    return LaunchDescription(
        [
            declare_map_yaml,
            declare_seed,
            declare_time_limit,
            declare_time_scale,
            declare_results_path,
            declare_rviz,
            declare_bt_xml,
            declare_costmap_params,
            sim_node,
            slam,
            nav2,
            rviz_node,
        ]
    )
