#!/usr/bin/env bash
# Run the challenge in the already-built `viewpoint-planning` image,
# bind-mounted to this repo so eval.py reads the repo's code/maps
# and writes results/<timestamp>/ straight back to the host - no separate
# volume needed, it's all the one bind mount.
#
# Build the image first (see README.md), and rebuild it whenever
# Dockerfile/requirements.txt change:
#   docker build -t viewpoint-planning .
#
# The map set that lives alongside this repo (../maps, sibling to
# viewpoint_planning/) is bind-mounted read-only too, if present, over
# /app/maps - so it's read live from the host, never copied into the image.
#
# Usage:
#   ./eval_runner.sh --map maps/1/room.yaml
#   ./eval_runner.sh --map maps/2/room.yaml
set -euo pipefail

IMAGE=viewpoint-planning
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_MAPS_DIR="$SCRIPT_DIR/../maps"

DOCKER_ARGS=(-v "$SCRIPT_DIR":/app -w /app -u "$(id -u)":"$(id -g)")
if [ -d "$EXTERNAL_MAPS_DIR" ]; then
  DOCKER_ARGS+=(-v "$EXTERNAL_MAPS_DIR":/app/maps:ro)
fi

docker run --rm "${DOCKER_ARGS[@]}" "$IMAGE" "$@"
