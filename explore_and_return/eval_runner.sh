#!/usr/bin/env bash
# Convenience launcher: brings up the provided infra (sim + slam_toolbox +
# Nav2) for one map, waits for it to settle, then runs your explorer node.
# Ctrl-C stops both. The final report is printed once /finish_exploration
# is called (or the time limit is hit).
#
# Usage: ./eval_runner.sh <path/to/room.yaml> [seed] [time_limit_s] [time_scale]
# Set RVIZ=true to also open RViz and watch the robot/map/coverage live.
set -euo pipefail

MAP_YAML="${1:?Usage: eval_runner.sh <room.yaml> [seed] [time_limit_s] [time_scale]}"
SEED="${2:-0}"
TIME_LIMIT="${3:-5400.0}"
TIME_SCALE="${4:-4.0}"
RVIZ="${RVIZ:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$SCRIPT_DIR/ws"
RESULTS_DIR="$SCRIPT_DIR/results/$(date +%Y%m%d_%H%M%S)"
RESULTS_PATH="$RESULTS_DIR/report.yaml"
mkdir -p "$RESULTS_DIR"

# ROS 2's ament setup scripts aren't nounset-safe, so relax -u just for sourcing them.
set +u
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

# Recursively SIGKILL a PID and everything descended from it. ros2 launch's
# own graceful shutdown can't be trusted here (/clock stops the instant
# /finish_exploration runs, so sim-time-dependent nodes can stall shutting
# down), and pkill -P alone only reaches direct children — this walks the
# whole tree so nothing (e.g. RViz) survives as an orphan.
kill_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_tree "$child"
  done
  kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
  echo "Shutting down..."
  kill_tree "$INFRA_PID"
  kill_tree "$CANDIDATE_PID"
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Launching sim + slam_toolbox + Nav2 (map: $MAP_YAML, seed: $SEED, time_scale: ${TIME_SCALE}x)..."
ros2 launch challenge_sim challenge.launch.py \
  map_yaml:="$MAP_YAML" seed:="$SEED" time_limit_s:="$TIME_LIMIT" time_scale:="$TIME_SCALE" \
  results_path:="$RESULTS_PATH" rviz:="$RVIZ" &
INFRA_PID=$!

echo "Waiting for bringup..."
sleep 30

echo "Starting your explorer node..."
ros2 run candidate_explorer explorer_node --ros-args -p use_sim_time:=true &
CANDIDATE_PID=$!

echo "Running. Waiting for $RESULTS_PATH to appear (or Ctrl-C to stop)..."
until [ -f "$RESULTS_PATH" ]; do
  sleep 2
done

echo ""
echo "=== REPORT ==="
cat "$RESULTS_PATH"
