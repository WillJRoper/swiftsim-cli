"""Task-count analysis module for SWIFT simulations.

This module analyses the `engine_print_task_counts` output in SWIFT logs.

CLI integration:
  * add_task_counts_arguments(subparsers)
  * run_swift_task_counts(args)

Core functionality:
  * analyse_swift_task_counts(log_file, output_path, prefix, show_plot)

The analysis is intentionally light-weight:
  * Uses timing.log_scan.scan_log() (task-count blocks are tracked per rank)
  * Builds a time series of per-step total task counts (summed over ranks,
    using the last block each rank printed in the step)
  * Produces:
      - Scatter plot: total tasks vs simulation time
      - Cumulative plot: cumulative tasks vs simulation time
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from swiftsim_cli.modes.common_args import (
    add_labels_argument,
    add_output_arguments,
)
from swiftsim_cli.timing.log_scan import scan_log
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.plotting import save_figure


class LogData(TypedDict):
    """Type definition for log file data."""

    log_file: str
    times: NDArray[np.float64]
    totals: NDArray[np.float64]
    cumulative: NDArray[np.float64]
    marker: str
    label: str
    task_series: dict[str, NDArray[np.float64]]


__all__ = [
    "add_task_counts_arguments",
    "run_swift_task_counts",
    "analyse_swift_task_counts",
]


# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "task_counts_analysis"


# ============================================================================
# CLI ARGUMENT SETUP
# ============================================================================


def add_task_counts_arguments(subparsers) -> None:
    """Add CLI arguments for engine_print_task_counts analysis.

    Subcommand name: 'task-counts'
    """
    task_parser = subparsers.add_parser(
        "task-counts",
        help=(
            "Plot the engine_print_task_counts output of SWIFT log files "
            "per step and cumulatively."
        ),
    )

    task_parser.add_argument(
        "log_files",
        nargs="+",
        help="SWIFT log files to analyse (plotted together).",
        type=Path,
    )
    add_labels_argument(task_parser)
    add_output_arguments(task_parser, OUT_SUBDIR)

    task_parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        nargs="+",
        help="Task names to include, as written in the log (e.g. 'sort', "
        "'self', 'pair') (default: all tasks).",
        default=None,
    )


def run_swift_task_counts(args: argparse.Namespace) -> None:
    """Entry point for the 'task-counts' CLI subcommand.

    This mirrors run_swift_log_timing() and simply forwards args.
    """
    analyse_swift_task_counts(
        log_files=[str(f) for f in args.log_files],
        labels=args.labels,
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        task_filter=args.tasks,
    )


def task_count_series(
    log_file: str, task_filter: list[str] | None = None
) -> tuple[list[int], list[float], list[int], dict[str, NDArray[np.float64]]]:
    """Per-step task counts of one log.

    Uses the last block each rank printed in a step, summed over ranks.

    Args:
        log_file: The SWIFT log.
        task_filter: If given, the totals only count these tasks.

    Returns:
        (steps, simulation times, totals, per-task series); every series
        has one value per step (zero where a step did not list the task).
    """
    # Parse the log. Steps without a simulation time (an incomplete
    # last step) cannot be placed on the time axis.
    scan = scan_log(log_file)
    steps: list[int] = []
    sim_times: list[float] = []
    totals: list[int] = []
    per_step_counts: list[dict[str, int]] = []

    for step in scan.steps:
        if step.sim_time is None or not step.task_counts:
            continue
        # The last block each rank printed in this step
        snaps = list({s.rank: s for s in step.task_counts}.values())
        steps.append(step.number)
        sim_times.append(step.sim_time)

        # Task counts summed over ranks
        aggregated_counts: dict[str, int] = {}
        for snap in snaps:
            for task_name, count in snap.counts.items():
                aggregated_counts[task_name] = (
                    aggregated_counts.get(task_name, 0) + count
                )
        per_step_counts.append(aggregated_counts)

        if task_filter:
            totals.append(
                sum(aggregated_counts.get(t, 0) for t in task_filter)
            )
        else:
            totals.append(
                sum(
                    int(snap.total_tasks or snap.system_total or 0)
                    for snap in snaps
                )
            )

    # One value per step for every task type (zero where a step's
    # blocks did not list the task).
    task_names = sorted({n for counts in per_step_counts for n in counts})
    task_series: dict[str, NDArray[np.float64]] = {
        name: np.asarray(
            [counts.get(name, 0) for counts in per_step_counts],
            dtype=float,
        )
        for name in task_names
    }

    return steps, sim_times, totals, task_series


# ============================================================================
# CORE ANALYSIS
# ============================================================================


def analyse_swift_task_counts(
    log_files: list[str],
    labels: list[str] | None = None,
    output_path: str | Path | None = None,
    prefix: str | None = None,
    show_plot: bool = False,
    task_filter: list[str] | None = None,
) -> None:
    """Analyse engine_print_task_counts blocks in SWIFT log files.

    This function:
      * Accepts multiple log files for comparison
      * Scans each log once, assigning task-count blocks to steps (the
        records before step row N belong to step N).
      * Sums the last block each rank printed in a step over the ranks.
      * Optionally filters to include only specific task types.
      * Builds:
          - A scatter plot of total tasks vs simulation time.
          - A cumulative total tasks vs simulation time plot.
      * Each log file gets a different marker on the plots.

    Args:
        log_files:
            List of paths to SWIFT log files to analyse.
        labels:
            Optional list of labels for the log files. If None, filenames
            will be used. Must match the order and length of log_files.
        output_path:
            Directory whose ``task_counts_analysis`` subdirectory receives
            the plots. If None, the current directory is used.
        prefix:
            Optional prefix added to each output filename.
        show_plot:
            Whether to display plots interactively.
        task_filter:
            Optional list of task names to include. If None, all tasks
            are included.
    """
    print(
        f"Analyzing engine_print_task_counts in {len(log_files)} log file(s)"
    )

    # Validate labels if provided
    if labels is not None:
        if len(labels) != len(log_files):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match "
                f"number of log files ({len(log_files)})"
            )
    else:
        # Use filenames as labels
        labels = [Path(f).name for f in log_files]

    if task_filter:
        print(f"Filtering for specific tasks: {', '.join(task_filter)}")

    out_dir = OUT_SUBDIR

    # Define markers for different log files
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # ------------------------------------------------------------------
    # Process each log file
    # ------------------------------------------------------------------
    all_data: list[LogData] = []

    for log_idx, log_file in enumerate(log_files):
        print(f"\nLog {log_idx + 1}/{len(log_files)}: {log_file}")

        steps, sim_times, totals, task_series = task_count_series(
            log_file, task_filter
        )
        print(f"  Found task counts for {len(steps)} steps")

        if not steps:
            print("  WARNING: No usable data found in this log")
            continue

        steps_arr = np.asarray(steps, dtype=int)
        times_arr = np.asarray(sim_times, dtype=float)
        totals_arr = np.asarray(totals, dtype=float)
        cumulative_arr = np.cumsum(totals_arr)

        print(
            f"  Prepared time series: {len(steps_arr)} steps "
            f"(step {steps_arr.min()} - {steps_arr.max()})"
        )

        # Print diagnostic table for this log
        print("\n  Task Count Summary:")
        print("  " + "=" * 70)
        print(
            f"  {'Task Type':<20} {'Total':<15} {'Avg/Step':<15} "
            f"{'Max/Step':<15}"
        )
        print("  " + "-" * 70)

        # Overall totals row
        total_all = totals_arr.sum()
        avg_all = totals_arr.mean()
        max_all = totals_arr.max()
        print(
            f"  {'ALL TASKS':<20} {total_all:<15,.0f} {avg_all:<15,.2f} "
            f"{max_all:<15,.0f}"
        )
        print("  " + "-" * 70)

        # Individual task type rows (sorted by total count descending)
        task_totals = []
        for task_name in sorted(task_series.keys()):
            task_arr = task_series[task_name]
            task_total = task_arr.sum()
            task_avg = task_arr.mean()
            task_max = task_arr.max()
            task_totals.append((task_name, task_total, task_avg, task_max))

        # Sort by total descending
        task_totals.sort(key=lambda x: x[1], reverse=True)

        for task_name, task_total, task_avg, task_max in task_totals:
            print(
                f"  {task_name:<20} {task_total:<15,.0f} {task_avg:<15,.2f} "
                f"{task_max:<15,.0f}"
            )
        print("  " + "=" * 70)

        all_data.append(
            {
                "log_file": log_file,
                "times": times_arr,
                "totals": totals_arr,
                "cumulative": cumulative_arr,
                "marker": markers[log_idx % len(markers)],
                "label": labels[log_idx],
                "task_series": task_series,
            }
        )

    if not all_data:
        print("\nNo usable data found in any log files!")
        return

    # ------------------------------------------------------------------
    # Plot 1: Scatter of total tasks vs simulation time
    # ------------------------------------------------------------------
    print("Creating per-step task-count scatter plot...")

    # Build plot titles based on whether filtering is active
    if task_filter:
        task_list = ", ".join(task_filter)
        title_suffix = f" (filtered: {task_list})"
        ylabel_suffix = " - filtered tasks"
    else:
        title_suffix = ""
        ylabel_suffix = ""

    fig, ax = plt.subplots(figsize=(10, 6))
    for data in all_data:
        ax.scatter(
            data["times"],
            data["totals"],
            marker=data["marker"],
            s=20,
            alpha=0.7,
            label=data["label"],
        )
    if len(all_data) > 1:
        ax.legend()
    ax.set_xlabel("Simulation time")
    ax.set_ylabel(f"Total tasks per step{ylabel_suffix}")
    ax.set_title(f"engine_print_task_counts: per-step totals{title_suffix}")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    p1 = create_output_path(
        output_path, prefix, "task_counts_per_step.png", out_dir
    )
    save_figure(fig, p1, show=show_plot)

    # ------------------------------------------------------------------
    # Plot 2: Cumulative tasks vs simulation time
    # ------------------------------------------------------------------
    print("Creating cumulative task-count plot...")

    fig, ax = plt.subplots(figsize=(10, 6))
    for data in all_data:
        ax.plot(
            data["times"],
            data["cumulative"],
            marker=data["marker"],
            markersize=4,
            label=data["label"],
        )
    if len(all_data) > 1:
        ax.legend()
    ax.set_xlabel("Simulation time")
    ax.set_ylabel(f"Cumulative tasks{ylabel_suffix}")
    ax.set_title(
        f"engine_print_task_counts: cumulative total tasks{title_suffix}"
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    p2 = create_output_path(
        output_path, prefix, "task_counts_cumulative.png", out_dir
    )
    save_figure(fig, p2, show=show_plot)

    # ------------------------------------------------------------------
    # Plot 3: Individual task types vs simulation time
    # ------------------------------------------------------------------
    print("Creating individual task types scatter plot...")

    # Collect all unique task names across all logs
    all_task_names: set[str] = set()
    for data in all_data:
        all_task_names.update(data["task_series"].keys())

    # Filter to only include non-zero tasks or specified tasks
    tasks_to_plot: list[str]
    if task_filter:
        tasks_to_plot = [t for t in task_filter if t in all_task_names]
    else:
        # Only plot tasks that have non-zero counts
        tasks_to_plot = []
        for task_name in sorted(all_task_names):
            has_nonzero = False
            for data in all_data:
                if task_name in data["task_series"]:
                    if np.any(data["task_series"][task_name] > 0):
                        has_nonzero = True
                        break
            if has_nonzero:
                tasks_to_plot.append(task_name)

    if tasks_to_plot:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Use a colormap for different task types
        colors = matplotlib.colormaps["tab20"](
            np.linspace(0, 1, len(tasks_to_plot))
        )

        # Plot data without labels first
        for task_idx, task_name in enumerate(tasks_to_plot):
            for log_idx, data in enumerate(all_data):
                if task_name not in data["task_series"]:
                    continue

                task_counts = data["task_series"][task_name]
                # Only plot non-zero values
                nonzero_mask = task_counts > 0
                if not np.any(nonzero_mask):
                    continue

                ax.scatter(
                    data["times"][nonzero_mask],
                    task_counts[nonzero_mask],
                    marker=data["marker"],
                    s=20,
                    alpha=0.6,
                    color=colors[task_idx],
                )

        ax.set_xlabel("Simulation time")
        ax.set_ylabel("Task count per step")
        ax.set_title("Individual task types evolution")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle="--", which="both")

        # Create two separate legends
        from matplotlib.lines import Line2D

        # Legend 1: Task types (colors)
        task_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[i],
                markersize=8,
                label=task_name,
            )
            for i, task_name in enumerate(tasks_to_plot)
        ]
        legend1 = ax.legend(
            handles=task_handles,
            title="Task Types",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
        )

        # Legend 2: Log files (markers) - only if multiple logs
        if len(all_data) > 1:
            marker_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=data["marker"],
                    color="w",
                    markerfacecolor="gray",
                    markersize=8,
                    label=data["label"],
                )
                for data in all_data
            ]
            ax.legend(
                handles=marker_handles,
                title="Log Files",
                bbox_to_anchor=(1.05, 0.5),
                loc="upper left",
                fontsize=8,
            )
            # Add both legends to the plot
            ax.add_artist(legend1)

        p3 = create_output_path(
            output_path, prefix, "task_counts_by_type.png", out_dir
        )
        save_figure(fig, p3, show=show_plot)

        print("\nCreated task-count plots:")
        print(f"  - {p1}")
        print(f"  - {p2}")
        print(f"  - {p3}")
    else:
        print("\nCreated task-count plots:")
        print(f"  - {p1}")
        print(f"  - {p2}")
        print("  (No individual task plot - no non-zero tasks found)")
