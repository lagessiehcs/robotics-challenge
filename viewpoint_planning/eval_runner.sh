#!/usr/bin/env bash
# Run the challenge in the already-built `viewpoint-planning` image. The
# repository root is bind-mounted at /repo, so the evaluator reads the current
# planner and shared maps and can write canonical outputs under
# /repo/results/viewpoint_planning/.
#
# Build the image first (see README.md), and rebuild it whenever
# Dockerfile/requirements.txt change:
#   docker build -t viewpoint-planning .
#
# Usage:
#   ./eval_runner.sh --map maps/1/room.yaml
#   ./eval_runner.sh --map maps/2/room.yaml
set -euo pipefail

IMAGE=viewpoint-planning
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DOCKER_ARGS=(-v "$REPO_DIR":/repo -w /repo/viewpoint_planning -u "$(id -u)":"$(id -g)")
if [[ -d "$REPO_DIR/maps" ]]; then
  # Preserve the documented `maps/<id>/room.yaml` path inside the planner's
  # working directory while the repository root remains available at /repo
  # for canonical result output.
  DOCKER_ARGS+=(-v "$REPO_DIR/maps":/repo/viewpoint_planning/maps:ro)
fi

docker run --rm "${DOCKER_ARGS[@]}" "$IMAGE" "$@"
