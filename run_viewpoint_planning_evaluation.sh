#!/usr/bin/env bash
# Evaluate the current viewpoint planner once on every selected map and write
# stable, per-map result paths under the repository-level results directory.
# Build the viewpoint-planning image first; see viewpoint_planning/README.md.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$REPO_DIR/viewpoint_planning/eval_runner.sh"
MAP_IDS="${MAP_IDS:-1 2 3 4 5}"
OUTPUT_ROOT="$REPO_DIR/results/viewpoint_planning"
FORCE="${FORCE:-false}"

if [[ "$FORCE" != "true" && "$FORCE" != "false" ]]; then
  echo "FORCE must be true or false; got: $FORCE" >&2
  exit 2
fi

for map_id in $MAP_IDS; do
  map_yaml="$REPO_DIR/maps/$map_id/room.yaml"
  output_png="$OUTPUT_ROOT/$map_id/coverage_report.png"
  output_json="${output_png%.png}.json"
  if [[ ! -f "$map_yaml" ]]; then
    echo "Map $map_id does not exist: $map_yaml" >&2
    exit 2
  fi
  if [[ "$FORCE" == "false" && -f "$output_json" ]]; then
    echo "Skipping Map $map_id: $output_json already exists (set FORCE=true to rerun)."
    continue
  fi

  echo "=== Evaluating viewpoint planner on Map $map_id ==="
  "$RUNNER" --map "maps/$map_id/room.yaml" --out "/repo/results/viewpoint_planning/$map_id/coverage_report.png"
done

python3 "$REPO_DIR/summarize_viewpoint_planning.py" --input "$OUTPUT_ROOT"
