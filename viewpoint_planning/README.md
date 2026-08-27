# Challenge: Minimum-Stop Viewpoint Planning for Accurate Scanning

## The problem

You're given a floor plan (already known — this is not an exploration task) as
a standard occupancy grid: `room.pgm` + `room.yaml`, the same format
`nav_msgs/OccupancyGrid` and ROS's `map_server` use. Black = wall, white =
free space, gray = unknown. `room.yaml` gives you the resolution (meters/pixel)
and the origin.

You have a robot which you want to use to make accurate measurements of the space,
for that, the idea is to make the robot stop at key points where using a laser sensor,
a 360-degree sweep takes place. from a stop, everything in
line-of-sight within the sensor's range gets measured, with accuracy falling
off with distance. Anything occluded by a wall isn't seen at all.

## Your task

Implement `plan_viewpoints()` in `candidate_solution/solution.py`:

```python
def plan_viewpoints(grid: OccupancyGrid, sensor: SensorModel) -> list[tuple[float, float]]:
    ...
```

Given the map and the sensor model, return a list of `(x, y)` world-frame stop
poses, **in visit order**. That's it — you don't need to emit orientations,
the scan is a full rotation.

Constraints your stops must satisfy (the scorer enforces these — see
`sim/visibility.py::is_stop_valid`):
- Must be in free space.
- Must have room for the robot's footprint (default 0.2 m radius) clear of
  any wall.

We value reproducible results: given the same map and sensor model,
`plan_viewpoints()` should return the same stops in the same order every
time it's run. If your approach relies on randomness anywhere (random
restarts, stochastic sampling, etc.), seed it explicitly and mention the seed
in your write-up.

## Scoring

Everything runs in a container — no local Python setup needed. Build the
image once, from this directory:

```bash
docker build -t viewpoint-planning .
```

Rebuild it any time `Dockerfile` or `requirements.txt` change; otherwise you
only need this once. Then, for every run:

```bash
./eval_runner.sh --map maps/1/room.yaml
```

This bind-mounts this repo into the container so `eval.py` runs
your code straight off disk, and writes results back to the host — no volume
juggling. `maps/` here is a shared sample-map directory at the root of this
repo (sibling to `viewpoint_planning/`), auto-mounted read-only by the
script if present; try any of `maps/1/room.yaml` through `maps/5/room.yaml`,
each a `room.pgm` + `room.yaml` pair in the format described above.

If you'd rather run outside Docker: `pip install -r requirements.txt`, then
call `python eval.py --map maps/1/room.yaml` directly — same
script, same scoring, just your local interpreter.

Either way, this scores your solution and writes `coverage_report.png` to
`results/<timestamp>/` — green wall segments were scanned at sufficient
quality, red were missed or never in range, and numbered markers show your
stops in visit order. It also writes `coverage_report.json` alongside it
with the same metrics in machine-readable form (coverage fraction, stop
count, tour length, invalid stops). You're evaluated on three axes, in this
priority order:

1. **Wall coverage %** — fraction of wall cells observed at or above
   `sensor.min_quality`. This is the primary metric; a plan that covers 60% of
   the space with 2 stops loses to one that covers 98% with 5.
2. **Stop count** — fewer is better, at a given coverage level.
3. **Tour length** — shorter travel between stops is better, at a given
   coverage level and stop count.

There's a real tradeoff between all three — say what you optimized for and
why in your write-up. A solution that dominates on all three axes against a
naive baseline (see the placeholder in `candidate_solution/solution.py` — a
single stop at the map's centroid) is a strong submission on its own; you
don't need a fully general/optimal solver.

The exact scan/quality model lives in `sim/visibility.py` — it's not hidden
from you, so you can reason about it precisely (e.g. exploit `quality_at_range`
being a known linear falloff) rather than treating the sensor as a black box.

**One things worth knowing about how the metrics are actually computed:**

- **Tour length is a real, drivable distance, not a straight line.** A
  straight line between two stops can cut straight through a wall (e.g. two
  stops in different rooms) — not something the robot could actually drive.
  `tour_length_m` is instead an 8-connected shortest path through free space,
  clear of walls by the robot's footprint (`sim/pathing.py::shortest_path`,
  A* over `sim/pathing.py::traversable_mask`). The report PNG draws this
  actual routed path (solid blue) rather than a dashed straight line between
  markers; a dashed red line means no free-space route could be found between
  that pair (e.g. one of them is an invalid stop), and the straight-line
  distance is used as a fallback for that segment only.


## What "good" looks like

A baseline solution: pick candidate stop locations (e.g. on a grid over free
space), greedily select the one that covers the most currently-uncovered
wall, repeat until coverage plateaus, then order the chosen stops with a
simple nearest-neighbor tour. This is a legitimate, gradeable submission.

Stronger submissions might do one or more of:
- A real set-cover formulation instead of pure greedy (e.g. an ILP or a
  weighted greedy with a proven approximation bound).
- A proper tour optimization (2-opt, or better) instead of nearest-neighbor.
- Model incidence angle — a wall struck at a grazing angle scans worse than
  one struck head-on — as an addition to `quality_at_range`, and justify why
  that matters for a real accurate scan.
- Handle the case where full coverage is geometrically impossible (occluded
  nooks smaller than the sensor's minimum range, etc.) gracefully rather than
  looping forever trying to cover them.

## Write-Up

In [write-ups](../write-ups/) document your approach and reasoning behind it, what you'd do with more time,
and where you expect it to break. you can also use the template located in the directory.
