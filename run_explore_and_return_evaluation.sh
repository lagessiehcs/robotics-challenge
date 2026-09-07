#!/usr/bin/env bash
# Run the explore-and-return evaluation on every supplied map. Each trial uses
# seed 0 (a new random spawn) and copies completed results to
# results/explore_and_return.
#
# Usage: ./run_explore_and_return_evaluation.sh [trials_per_map]
# Example: ./run_explore_and_return_evaluation.sh 10
#
# Optional environment variables:
#   MAP_IDS="1 2 3 4 5"  Maps to run (default: all five maps)
#   TIME_LIMIT_S=5400.0   Simulated time limit passed to eval_runner.sh
#   TIME_SCALE=4.0        Simulation speed multiplier passed to eval_runner.sh
set -euo pipefail

TRIALS_PER_MAP="${1:-10}"
MAP_IDS="${MAP_IDS:-1 2 3 4 5}"
TIME_LIMIT_S="${TIME_LIMIT_S:-5400.0}"
TIME_SCALE="${TIME_SCALE:-4.0}"

if ! [[ "$TRIALS_PER_MAP" =~ ^[1-9][0-9]*$ ]]; then
  echo "trials_per_map must be a positive integer; got: $TRIALS_PER_MAP" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_RUNNER="$REPO_DIR/explore_and_return/eval_runner.sh"
OUTPUT_ROOT="$REPO_DIR/results/explore_and_return"
SOURCE_RESULTS_ROOT="$REPO_DIR/explore_and_return/results"
MARKER_FILE="$(mktemp)"
FAILED_RUNS=0

cleanup_marker() {
  rm -f "$MARKER_FILE"
}
trap cleanup_marker EXIT

for map_id in $MAP_IDS; do
  map_yaml="$REPO_DIR/maps/$map_id/room.yaml"
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

    mkdir -p "$trial_dir"
    echo "=== Map $map_id, $trial_name ($trial/$TRIALS_PER_MAP), seed 0 ==="
    # eval_runner.sh owns its timestamped output path. Record a timestamp just
    # before launch, then copy its newly completed result directory here.
    touch "$MARKER_FILE"
    if "$EVAL_RUNNER" "$map_yaml" 0 \
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
