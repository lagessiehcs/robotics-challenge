#!/usr/bin/env python3
"""Summarize an explore-and-return indoor batch and create write-up plots.

Usage:
    python3 summarize_indoor_batch.py
    python3 summarize_indoor_batch.py --input results/explore_and_return/batch_random_indoor

The default input is the root-level results directory requested by the
challenge workflow. Output files are written to <input>/summary by default.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


DEFAULT_INPUT = Path("results/explore_and_return/batch_random_indoor")
PLOT_DPI = 220
CORRECTION_PATTERN = re.compile(
    r"Large map->odom correction detected \(translation=([0-9.]+)m, "
    r"yaw=([0-9.]+)deg\)"
)


def outcome(row: dict) -> str:
    """Classify a run using the two criteria that determine success."""
    coverage_ok = row["coverage_fraction"] >= 0.8
    home_ok = row["distance_to_home_m"] <= 0.3
    if coverage_ok and home_ok:
        return "Success"
    if coverage_ok:
        return "Coverage pass, return miss"
    if home_ok:
        return "Coverage miss, home pass"
    return "Both thresholds missed"


def read_reports(input_dir: Path) -> list[dict]:
    rows = []
    for report_path in sorted(input_dir.glob("*/trial_*/report.yaml")):
        with report_path.open() as report_file:
            report = yaml.safe_load(report_file) or {}
        log_path = report_path.with_name("console.log")
        correction_events = []
        if log_path.is_file():
            correction_events = CORRECTION_PATTERN.findall(log_path.read_text())
        try:
            map_id = int(report_path.parent.parent.name)
            trial_id = int(report_path.parent.name.removeprefix("trial_"))
            rows.append(
                {
                    "map_id": map_id,
                    "trial_id": trial_id,
                    "seed": report.get("seed"),
                    "coverage_fraction": float(report["coverage_fraction"]),
                    "distance_to_home_m": float(report["distance_to_home_m"]),
                    "elapsed_time_s": float(report["elapsed_time_s"]),
                    "elapsed_wall_time_s": float(report["elapsed_wall_time_s"]),
                    "collision_count": int(report["collision_count"]),
                    "success": bool(report["success"]),
                    "timed_out": bool(report["timed_out"]),
                    "large_map_correction_count": len(correction_events),
                    "largest_map_correction_m": max(
                        (float(event[0]) for event in correction_events), default=0.0
                    ),
                    "largest_map_correction_deg": max(
                        (float(event[1]) for event in correction_events), default=0.0
                    ),
                    "outcome": "",  # populated below for CSV and plots
                    "report_path": str(report_path),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed report: {report_path}") from exc
    for row in rows:
        row["outcome"] = outcome(row)
    return rows


def write_csv(rows: list[dict], output_dir: Path) -> None:
    fieldnames = list(rows[0])
    with (output_dir / "trial_metrics.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


OUTCOME_ORDER = [
    "Success",
    "Coverage pass, return miss",
    "Coverage miss, home pass",
    "Both thresholds missed",
]
OUTCOME_COLORS = ["#54A24B", "#F58518", "#4C78A8", "#E45756"]


def save_session_performance_plot(rows: list[dict], title: str, output_path: Path) -> None:
    """Show every trial against the coverage and return-home success criteria."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4), layout="constrained")
    color_by_outcome = dict(zip(OUTCOME_ORDER, OUTCOME_COLORS))

    # The success quadrant is deliberately shaded so the plot remains readable
    # even when multiple near-identical successful sessions overlap.
    ax.fill_betweenx([0.0, 0.3], 80.0, 102.0, color="#54A24B", alpha=0.10, zorder=0)
    for category in OUTCOME_ORDER:
        category_rows = [row for row in rows if row["outcome"] == category]
        if not category_rows:
            continue
        ax.scatter(
            [100 * row["coverage_fraction"] for row in category_rows],
            [row["distance_to_home_m"] for row in category_rows],
            s=56,
            color=color_by_outcome[category],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label=category,
            zorder=3,
        )

    max_home_distance = max(row["distance_to_home_m"] for row in rows)
    ax.axvline(80, color="#D62728", linestyle="--", linewidth=1.4, zorder=2)
    ax.axhline(0.3, color="#D62728", linestyle="--", linewidth=1.4, zorder=2)
    ax.text(80.8, max_home_distance * 0.78, "80% coverage", color="#A61C1C", fontsize=9)
    ax.text(1.0, 0.34, "0.3 m home", color="#A61C1C", fontsize=9)
    ax.set(
        title=title,
        xlabel="Coverage of reachable free space (%)",
        ylabel="True final distance to home (m; symlog scale)",
        xlim=(-2, 102),
        ylim=(-0.03, max(1.0, max_home_distance * 1.25)),
    )
    # This preserves the full range (including exactly-zero home distance) while
    # keeping the 0.3 m criterion visible beside multi-metre misses.
    ax.set_yscale("symlog", linthresh=0.3, linscale=0.8)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2, frameon=False)
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Indoor batch directory")
    parser.add_argument("--output", type=Path, help="Directory for CSV and PNG outputs")
    args = parser.parse_args()

    rows = read_reports(args.input)
    if not rows:
        raise SystemExit(f"No completed reports found under: {args.input}")

    output_dir = args.output or args.input / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir)
    save_session_performance_plot(
        rows,
        f"All indoor exploration sessions (n={len(rows)})",
        output_dir / "all_indoor_sessions.png",
    )
    stable_rows = [row for row in rows if row["large_map_correction_count"] == 0]
    save_session_performance_plot(
        stable_rows,
        f"Indoor sessions with no logged large SLAM correction (n={len(stable_rows)})",
        output_dir / "indoor_sessions_no_large_slam_correction.png",
    )
    print(f"Wrote CSV and figures for {len(rows)} reports into: {output_dir}")


if __name__ == "__main__":
    main()
