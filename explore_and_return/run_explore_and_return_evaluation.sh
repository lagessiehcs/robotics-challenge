#!/usr/bin/env bash
# Run the explore-and-return evaluation on every supplied map. By default each
# trial begins inside the building and completed results go to
# results/batch_random_indoor.
#
# Usage: ./run_explore_and_return_evaluation.sh [trials_per_map]
# Example: ./run_explore_and_return_evaluation.sh 10
#
# Optional environment variables:
#   MAP_IDS="1 2 3 4 5"  Maps to run (default: all five maps)
#   TIME_LIMIT_S=5400.0   Simulated time limit passed to eval_runner.sh
#   TIME_SCALE=4.0        Simulation speed multiplier passed to eval_runner.sh
#   SPAWN_MODE=indoor     "indoor" (default) or "random"; random results go
#                          to results/batch_random
#   SPAWN_SEED_BASE=100000 Base for deterministic indoor-spawn seed search
set -euo pipefail

TRIALS_PER_MAP="${1:-10}"
MAP_IDS="${MAP_IDS:-1 2 3 4 5}"
TIME_LIMIT_S="${TIME_LIMIT_S:-5400.0}"
TIME_SCALE="${TIME_SCALE:-4.0}"
SPAWN_MODE="${SPAWN_MODE:-indoor}"
SPAWN_SEED_BASE="${SPAWN_SEED_BASE:-100000}"

if ! [[ "$TRIALS_PER_MAP" =~ ^[1-9][0-9]*$ ]]; then
  echo "trials_per_map must be a positive integer; got: $TRIALS_PER_MAP" >&2
  exit 2
fi
if ! [[ "$SPAWN_SEED_BASE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPAWN_SEED_BASE must be a positive integer; got: $SPAWN_SEED_BASE" >&2
  exit 2
fi
if [[ "$SPAWN_MODE" != "indoor" && "$SPAWN_MODE" != "random" ]]; then
  echo "SPAWN_MODE must be 'indoor' or 'random'; got: $SPAWN_MODE" >&2
  exit 2
fi

CHALLENGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_RUNNER="$CHALLENGE_DIR/eval_runner.sh"
if [[ "$SPAWN_MODE" == "indoor" ]]; then
  OUTPUT_ROOT="$CHALLENGE_DIR/results/batch_random_indoor"
else
  OUTPUT_ROOT="$CHALLENGE_DIR/results/batch_random"
fi
SOURCE_RESULTS_ROOT="$CHALLENGE_DIR/results"
if [[ -d "$CHALLENGE_DIR/maps" ]]; then
  MAPS_DIR="$CHALLENGE_DIR/maps"
else
  # On the host, maps is a sibling of explore_and_return. In Docker it is
  # mounted at /challenge/maps, which is handled by the branch above.
  MAPS_DIR="$CHALLENGE_DIR/../maps"
fi
MARKER_FILE="$(mktemp)"
FAILED_RUNS=0

cleanup_marker() {
  rm -f "$MARKER_FILE"
}
trap cleanup_marker EXIT

# The simulator samples from all free map cells, including exterior space.
# Without relying on any pre-made spawn map, classify a candidate location by
# its ground-truth initial laser geometry: an indoor location is tightly
# enclosed by walls, so at least 95% of 360 rays must hit an occupied cell
# within the simulator's 6 m laser range and the mean hit range must be no
# more than 2.5 m.  The seed is deterministic per map/trial, which also makes
# a resumed batch reproduce the same start pose.
find_indoor_seed() {
  local map_yaml="$1"
  local first_seed="$2"

  PYTHONPATH="$CHALLENGE_DIR/ws/src/challenge_sim${PYTHONPATH:+:$PYTHONPATH}" \
    python3 - "$map_yaml" "$first_seed" <<'PY'
import math
import sys

import numpy as np

from challenge_sim.map_io import load_occupancy_grid, sample_free_pose

map_yaml, first_seed = sys.argv[1], int(sys.argv[2])
grid = load_occupancy_grid(map_yaml)


def is_enclosed_by_walls(x: float, y: float) -> bool:
    """Apply the simulator's occupied-cell raycast without laser noise."""
    ray_count = 360
    max_range_m = 6.0
    height, width = grid.data.shape
    row0, col0 = grid.world_to_pixel(x, y)
    steps = np.arange(1, int(max_range_m / grid.resolution) + 1)
    angles = 2 * math.pi * np.arange(ray_count) / ray_count
    rows = (row0 - np.sin(angles)[:, None] * steps[None, :]).astype(np.int64)
    cols = (col0 + np.cos(angles)[:, None] * steps[None, :]).astype(np.int64)
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    occupied = in_bounds & (
        grid.data[np.clip(rows, 0, height - 1), np.clip(cols, 0, width - 1)] == 1
    )
    hit = occupied.any(axis=1)
    ranges = np.where(hit, (occupied.argmax(axis=1) + 1) * grid.resolution, max_range_m)
    return hit.mean() >= 0.95 and ranges.mean() <= 2.5

for seed in range(first_seed, first_seed + 100000):
    # A non-zero seed is required: zero means a fresh nondeterministic spawn.
    spawn = sample_free_pose(grid, 0.2, np.random.default_rng(seed))
    if is_enclosed_by_walls(*spawn):
        print(seed)
        break
else:
    raise SystemExit("could not find an enclosed indoor spawn seed after 100000 attempts")
PY
}

for map_id in $MAP_IDS; do
  map_yaml="$MAPS_DIR/$map_id/room.yaml"
  if [[ ! -f "$map_yaml" ]]; then
    echo "Map $map_id does not exist: $map_yaml" >&2
    exit 2
  fi

  for ((trial = 1; trial <= TRIALS_PER_MAP; trial++)); do
    trial_name="$(printf 'trial_%02d' "$trial")"
    trial_dir="$OUTPUT_ROOT/$map_id/$trial_name"
    report_path="$trial_dir/report.yaml"
    log_path="$trial_dir/console.log"

    # A finished report makes the batch safely resumable. Do not overwrite an
    # incomplete directory: its log is useful when diagnosing a failed run.
    if [[ -f "$report_path" ]]; then
      echo "Skipping Map $map_id, $trial_name: report already exists."
      continue
    fi
    if [[ -e "$trial_dir" ]]; then
      echo "Refusing to reuse incomplete trial directory: $trial_dir" >&2
      echo "Inspect or rename it, then rerun this script to resume." >&2
      exit 1
    fi

    if [[ "$SPAWN_MODE" == "indoor" ]]; then
      first_seed=$((SPAWN_SEED_BASE + map_id * 1000000 + trial * 100000))
      if ! seed="$(find_indoor_seed "$map_yaml" "$first_seed")"; then
        echo "Could not choose an indoor spawn for Map $map_id, $trial_name." >&2
        FAILED_RUNS=$((FAILED_RUNS + 1))
        continue
      fi
      spawn_description="indoor seed $seed"
    else
      seed=0
      spawn_description="random seed 0"
    fi
    mkdir -p "$trial_dir"
    echo "=== Map $map_id, $trial_name ($trial/$TRIALS_PER_MAP), $spawn_description ==="
    # eval_runner.sh owns its timestamped output path. Record a timestamp just
    # before launch, then copy its newly completed result directory here.
    touch "$MARKER_FILE"
    if "$EVAL_RUNNER" "$map_yaml" "$seed" \
        "$TIME_LIMIT_S" "$TIME_SCALE" 2>&1 | tee "$log_path"; then
      mapfile -t source_reports < <(
        find "$SOURCE_RESULTS_ROOT" -mindepth 2 -maxdepth 2 -name report.yaml \
          -newer "$MARKER_FILE" -print | sort
      )
      if (( ${#source_reports[@]} == 1 )); then
        source_dir="$(dirname "${source_reports[0]}")"
        cp -a "$source_dir/." "$trial_dir/"
        echo "Completed: $report_path"
      elif (( ${#source_reports[@]} == 0 )); then
        echo "Run exited without creating a report: $trial_dir" >&2
        FAILED_RUNS=$((FAILED_RUNS + 1))
      else
        echo "Found multiple new reports; refusing to guess which one belongs to this trial." >&2
        FAILED_RUNS=$((FAILED_RUNS + 1))
      fi
    else
      echo "Run failed; log retained at: $log_path" >&2
      FAILED_RUNS=$((FAILED_RUNS + 1))
    fi
  done
done

if ((FAILED_RUNS > 0)); then
  echo "$FAILED_RUNS run(s) failed or produced no report." >&2
  exit 1
fi

echo "Evaluation complete: $OUTPUT_ROOT"
