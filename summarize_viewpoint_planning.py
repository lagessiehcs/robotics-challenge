#!/usr/bin/env python3
"""Summarize one canonical viewpoint-planning report per map.

Usage:
    python3 summarize_viewpoint_planning.py
    python3 summarize_viewpoint_planning.py --input results/viewpoint_planning
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("results/viewpoint_planning")
PLOT_DPI = 220


def read_reports(input_dir: Path) -> list[dict]:
    rows = []
    for report_path in sorted(input_dir.glob("*/coverage_report.json")):
        try:
            map_id = int(report_path.parent.name)
        except ValueError as exc:
            raise ValueError(f"Expected a numeric map directory: {report_path}") from exc
        report = json.loads(report_path.read_text())
        rows.append(
            {
                "map_id": map_id,
                "coverage_fraction": float(report["coverage_fraction"]),
                "covered_wall_cells": int(report["covered_wall_cells"]),
                "total_wall_cells": int(report["total_wall_cells"]),
                "unreachable_wall_cells": int(report["unreachable_wall_cells"]),
                "num_stops": int(report["num_stops"]),
                "tour_length_m": float(report["tour_length_m"]),
                "planning_time_s": float(report["planning_time_s"]),
                "invalid_stop_count": len(report["invalid_stops"]),
                "report_path": str(report_path),
            }
        )
    return rows


def write_csv(rows: list[dict], output_dir: Path) -> None:
    with (output_dir / "map_metrics.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], output_dir: Path) -> None:
    map_labels = [f"Map {row['map_id']}" for row in rows]
    coverage = [100 * row["coverage_fraction"] for row in rows]
    stops = [row["num_stops"] for row in rows]
    tour_lengths = [row["tour_length_m"] for row in rows]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained")
    metrics = [
        (coverage, "Wall coverage (%)", "Coverage by map", colors[0]),
        (stops, "Stops", "Stops selected", colors[1]),
        (tour_lengths, "Routed distance (m)", "Tour length", colors[2]),
    ]
    for axis, (values, ylabel, title, color) in zip(axes, metrics):
        bars = axis.bar(map_labels, values, color=color, alpha=0.85)
        axis.set(title=title, ylabel=ylabel)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        upper = max(values) * 1.15 if max(values) else 1.0
        axis.set_ylim(0, upper)
        for bar, value in zip(bars, values):
            label = f"{value:.1f}%" if ylabel == "Wall coverage (%)" else f"{value:.1f}" if "distance" in ylabel else str(value)
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.02,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Viewpoint-planning performance on the five maps", fontsize=15)
    fig.savefig(output_dir / "performance_by_map.png", dpi=PLOT_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = read_reports(args.input)
    if not rows:
        raise SystemExit(f"No coverage_report.json files found under: {args.input}")
    output_dir = args.output or args.input / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir)
    save_plot(rows, output_dir)
    print(f"Wrote metrics and figure for {len(rows)} reports into: {output_dir}")


if __name__ == "__main__":
    main()
