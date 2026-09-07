# Write-Up: Autonomous Explore-and-Return

## Approach

I implemented a map-driven frontier explorer on top of the provided SLAM and
Nav2 stack. The node runs a small state machine:

```text
WAITING_FOR_MAP -> EXPLORING -> RETURNING -> FINISHING -> DONE
```

During exploration it:

1. Treats a known-free occupancy-grid cell adjacent to an unknown cell as a
   frontier cell.
2. Rejects cells near obstacles, cells already effectively at the robot, and
   cells close to previously failed goals.
3. Groups remaining cells into 8-connected clusters and ignores very small
   clusters, which are commonly mapping noise.
4. Selects one reachable goal per cluster and scores clusters using distance,
   heading change, and estimated information gain. The heading term reduces
   unnecessary reversals; information gain is a bounded bonus rather than a
   reason to cross the map for a distant frontier.
5. Requests a Nav2 global path before committing to the navigation action.
   It rejects unavailable, zero-length, excessively indirect, or narrow-path
   routes. This avoids repeatedly spending time on frontiers that look valid
   in the occupancy grid but are poor navigation targets.

While a goal is active, the node monitors progress and marks a stalled or
failed frontier as temporarily unusable. It also watches for narrow passages
ahead and cancels goals that would enter one. A large `map -> odom` correction
from SLAM invalidates the current map-frame plan, so the node cancels that
goal, allows the costmaps to settle, and replans. However, most of the time  A large `map -> odom` correction means map is corrupted.

If frontier clusters exist but none yields a usable goal, the robot tries a
short, directional recovery move before checking again. Only after a full
recovery sweep and a 10-second no-frontier window does it consider exploration
complete.

For the return phase, home is never cached in the `map` frame. Home is the
`odom` origin, so the node transforms `(0, 0, 0)` from `odom` into `map`
immediately before every return attempt. This remains correct after SLAM
pose-graph corrections. A separate progress watchdog cancels a stuck
return-home goal, makes a short recovery move, and retries home. On a
successful arrival, the node calls `/finish_exploration`.

## Design Decisions & Tradeoffs

The primary objective was dependable coverage with a reliable return, rather
than pursuing every last mapped boundary. In particular:

- A 0.5 m obstacle-clearance filter and a 0.55 m minimum passage width trade
  some potentially observable space for fewer collisions and stuck episodes.
- Frontiers must be at least 0.60 m away. This avoids Nav2 accepting a goal
  inside its 0.5 m tolerance without meaningful robot motion.
- The frontier score favors continuity of heading and only modestly rewards
  information gain. This reduces branch-to-branch oscillation and repeated
  travel through explored corridors, at the cost of occasionally delaying a
  large frontier behind the robot.
- A route whose Nav2 path exceeds 2.5x its straight-line distance is recorded
  as expensive and deferred once. Its measured detour becomes a score penalty,
  so cheaper frontiers are preferred without permanently excluding a reachable
  region.
- Failed goals are blacklisted within 0.8 m. This prevents retry loops but can
  discard a frontier that becomes practical after a later map update.

The candidate costmap configuration was changed from the original 5 m × 5 m,
0.02 m-resolution local costmap to a 2 m × 2 m, 0.05 m-resolution rolling
local costmap. Both local and global costmaps changed from 0.1 m inflation
with a cost-scaling factor of 3.0 to 0.5 m inflation with a factor of 5.0.
This makes the robot more conservative around mapped obstacles and reduces
costmap computation, at the cost of treating more narrow space as unsuitable
for navigation.

The behavior tree retains Nav2's 1 Hz replanning and six top-level recovery
attempts, but changes the two context recovery nodes from one retry to zero.
Its recovery order changes from clear-costmaps → spin → wait → short backup
to clear-costmaps → 0.5 m backup → positive spin → negative spin. This favors
immediately creating clearance and changing heading over waiting in place.
The explorer's checks complement Nav2: Nav2 handles driving and local
recovery, while the explorer decides whether a proposed exploration target is
worth attempting.

## Conclusions

### Performance

The repository contains completed reports in `explore_and_return/results/`.
They are historical development runs rather than a controlled benchmark: the
reports do not record a code revision, configuration snapshot, or an automatic
SLAM-map-integrity label. The figures below therefore describe the observed
results, not a general performance guarantee.

Of the 81 reports containing all final metrics, 38 achieved at least 95%
coverage and returned within the 0.30 m requirement. In those runs, coverage
was 96.1–100.0% (mean 99.2%), final distance to home was 0.116–0.299 m (mean
0.212 m), and simulated completion time was 108–2,124 s (mean 540 s). Thus,
when SLAM produces a usable map, the explorer has repeatedly achieved roughly
95–100% coverage and returned within tolerance.

| Map | Completed reports | Successful reports | Reports with >=95% coverage and <=0.30 m home |
|---|---:|---:|---:|
| 1 | 25 | 18 | 16 |
| 2 | 15 | 1 | 1 |
| 3 | 8 | 3 | 3 |
| 4 | 17 | 10 | 10 |
| 5 | 16 | 11 | 8 |

The dominant observed failure mode is a corrupted or otherwise unusable SLAM
map. This is diagnosed from the saved maps and run behaviour, not from a field
in `report.yaml`; possible causes include odometry drift and an incorrect or
failed loop closure. Failures are concentrated on Maps 2 and 3, though they
also occur on other maps. Establishing the root cause requires retaining the
corresponding SLAM/Nav2 logs and saved map for each failed run.

For a reproducible evaluation, include one row per run with the following
data: map ID/path, run ID (the result-directory timestamp), requested and
reported seed, code commit, explorer/SLAM/Nav2 configuration, time scale,
`coverage_fraction`, `distance_to_home_m`, `elapsed_time_s`,
`collision_count`, `timed_out`, `success`, and a map-integrity classification
(`usable`, `corrupted`, or `uncertain`) supported by the saved `map.yaml`/
`map.pgm` and logs. Summarize those rows per map and across several nonzero
seeds, reporting both all-run results and the usable-map subset; do not omit
corrupted-map runs from the overall success rate.

Run the 10-trial-per-map indoor evaluation in Docker from the repository root:

```bash
docker compose -f explore_and_return/docker/docker-compose.yml run --rm challenge \
  bash -lc 'cd /challenge && ./run_explore_and_return_evaluation.sh 10'
```

To run the unrestricted random-spawn baseline instead:

```bash
docker compose -f explore_and_return/docker/docker-compose.yml run --rm challenge \
  bash -lc 'cd /challenge && SPAWN_MODE=random ./run_explore_and_return_evaluation.sh 10'
```

For a smaller or tuned batch, pass environment variables before the script;
for example, this runs five trials only on Maps 1 and 3 at a 1,800-second
simulated-time limit:

```bash
docker compose -f explore_and_return/docker/docker-compose.yml run --rm challenge \
  bash -lc 'cd /challenge && MAP_IDS="1 3" TIME_LIMIT_S=1800 ./run_explore_and_return_evaluation.sh 5'
```

The batch evaluator is my contribution. By default it selects reproducible
nonzero seeds whose initial location is tightly enclosed by walls in a
simulated 360° lidar scan of `room.pgm`; it does not need a pre-made spawn
mask. This keeps the indoor evaluation set separate at
`explore_and_return/results/batch_random_indoor/<map>/trial_XX/`. To retain
an unrestricted baseline, set `SPAWN_MODE=random`; those random-spawn results
are written to `explore_and_return/results/batch_random/<map>/trial_XX/`.

Each completed trial includes the simulator report, saved map, and console
log. The launcher is resumable: it skips a trial with an existing `report.yaml`
and preserves an incomplete trial folder for inspection rather than
overwriting it.

### What I'd Do With More Time

- Replace Euclidean frontier ranking with path-cost ranking for every viable
  cluster, rather than using path planning only as a final filter.
- Estimate information gain by ray-casting from candidate viewpoints, which
  would better reflect what the laser can actually observe.
- Make clearance, detour, timeout, and score weights map-resolution-aware and
  tune them from a multi-seed evaluation set.
- Expire or revalidate blacklist entries after meaningful map changes, so a
  SLAM correction or new view can reopen previously rejected space.
- Add explicit coverage and return-margin budgeting, allowing the node to
  leave low-value frontier remnants when the estimated return cost is high.

### Known Limitations / Where I Expect This to Break

- Occupancy-grid frontiers are sensitive to incomplete or noisy SLAM maps;
  filtering helps, but can remove narrow legitimate entrances.
- The distance-transform and cross-section tests use the current map rather
  than the exact robot footprint and future local-costmap state. A route can
  still become blocked after it is accepted.
- The 10-second no-frontier timeout may stop too early if SLAM publishes a
  delayed map update, while a longer timeout wastes simulated time in a truly
  complete map.
- Recovery moves are open-loop Nav2 goals in fixed relative directions. They
  are practical for escaping local dead ends but are not a general local
  exploration planner.
- Exact repeatability is not expected for evaluation runs with random spawn
  seed `0`; fixed nonzero seeds make the simulator spawn reproducible, though
  asynchronous SLAM and navigation timing can still introduce small variation.

## Modified Files or Packages

- `candidate_explorer/candidate_explorer/explorer_node.py`: frontier
  detection, target scoring, path validation, recovery, map-correction
  handling, and return-home logic.
- `candidate_explorer/config/costmap_params.yaml`: candidate costmap overlay
  with laser obstacle layers and inflation settings.
- `candidate_explorer/config/navigate_to_pose.xml`: editable Nav2 replanning
  and recovery behavior tree.
