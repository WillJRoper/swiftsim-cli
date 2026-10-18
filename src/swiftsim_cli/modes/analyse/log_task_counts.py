"""Task-count analysis of ``engine_print_task_counts`` output in SWIFT logs.

Per step (the last block each rank printed in it):

* the total is SWIFT's "System total" (all ranks) when printed, else the
  sum of each rank's "Total" (the full task list, skipped tasks included);
* per-type counts are the tasks that will run (SWIFT counts skipped tasks
  under ``skipped``, not their type), summed over the ranks in the log.

MPI logs often only hold rank 0's output; then the per-type counts (and
the totals, without a System total) cover rank 0 only, and a warning says
so. Plots: total tasks per step and per-type counts per step.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from swiftsim_cli.modes.common_args import (
    add_labels_argument,
    add_output_arguments,
    add_steps_argument,
    existing_file,
    resolve_labels,
)
from swiftsim_cli.timing.log_scan import scan_log
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.messages import plural, reading, warn
from swiftsim_cli.utilities.plotting import save_figure

from .task_plot_metadata import colour_map


@dataclass
class TaskCounts:
    """Per-step task counts of one log (one value per step everywhere).

    Attributes:
        steps: Step numbers.
        sim_times: Simulation times (internal units).
        totals: Total tasks (selected types when filtered).
        task_series: Per-type counts of active (unskipped) tasks.
        skipped: Skipped tasks of any type.
        n_ranks: MPI ranks of the run.
        ranks_seen: Ranks whose blocks appear in the log.
        system_total: Whether totals came from SWIFT's System total.
    """

    steps: list[int]
    sim_times: list[float]
    totals: list[int]
    task_series: dict[str, NDArray[np.float64]]
    skipped: NDArray[np.float64]
    n_ranks: int = 1
    ranks_seen: list[int] = field(default_factory=list)
    system_total: bool = False

    @property
    def partial(self) -> bool:
        """Whether only some ranks' blocks are in the log."""
        return len(self.ranks_seen) < self.n_ranks


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


def add_task_counts_arguments(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add CLI arguments for engine_print_task_counts analysis.

    Subcommand name: 'task-counts'
    """
    task_parser = subparsers.add_parser(
        "task-counts",
        help="Plot the engine_print_task_counts output of SWIFT log files "
        "per step.",
        description="Plot the engine_print_task_counts output of SWIFT log "
        "files per step: total tasks (SWIFT's System total for MPI runs) "
        "and the per-type counts of tasks that run. Requires SWIFT run with "
        "-v 1 (the counts are printed at each rebuild).",
    )
    task_parser.set_defaults(func=run_swift_task_counts)

    task_parser.add_argument(
        "log_files",
        nargs="+",
        metavar="FILE",
        help="SWIFT log files to analyse (plotted together).",
        type=existing_file,
    )
    add_labels_argument(task_parser)
    add_output_arguments(task_parser, OUT_SUBDIR)
    add_steps_argument(task_parser)

    task_parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        metavar="NAME",
        help="Task names to include, as written in the log (e.g. 'sort', "
        "'self', 'pair') (default: all tasks).",
        default=None,
    )


def run_swift_task_counts(args: argparse.Namespace) -> None:
    """Entry point for the 'task-counts' CLI subcommand."""
    analyse_swift_task_counts(
        log_files=[str(f) for f in args.log_files],
        labels=resolve_labels(args.log_files, args.labels),
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        task_filter=args.tasks,
        step_range=args.steps,
    )


def task_count_series(
    log_file: str,
    task_filter: list[str] | None = None,
    step_range: Sequence[int] | None = None,
) -> TaskCounts:
    """Per-step task counts of one log.

    Args:
        log_file: The SWIFT log.
        task_filter: If given, the totals only count these task types.
        step_range: Optional inclusive ``(start, end)`` step range.
    """
    # Steps without a simulation time (an incomplete last step) cannot
    # be placed on the time axis.
    with reading(log_file):
        scan = scan_log(log_file)
    steps: list[int] = []
    sim_times: list[float] = []
    totals: list[int] = []
    per_step_counts: list[dict[str, int]] = []
    ranks_seen: set[int] = set()
    used_system = False

    for step in scan.steps:
        if step.sim_time is None or not step.task_counts:
            continue
        if step_range is not None and not (
            step_range[0] <= step.number <= step_range[1]
        ):
            continue
        # The last block each rank printed in this step
        snaps = list({s.rank: s for s in step.task_counts}.values())
        ranks_seen.update(s.rank for s in snaps)
        steps.append(step.number)
        sim_times.append(step.sim_time)

        counts: dict[str, int] = {}
        for snap in snaps:
            for name, count in snap.counts.items():
                counts[name] = counts.get(name, 0) + count
        per_step_counts.append(counts)

        system = next((s.system_total for s in snaps if s.system_total), None)
        if task_filter:
            totals.append(sum(counts.get(t, 0) for t in task_filter))
        elif system is not None:
            used_system = True
            totals.append(system)
        else:
            totals.append(sum(int(s.total_tasks or 0) for s in snaps))

    names = sorted(
        {n for counts in per_step_counts for n in counts} - {"skipped"}
    )
    return TaskCounts(
        steps=steps,
        sim_times=sim_times,
        totals=totals,
        task_series={
            name: np.asarray(
                [c.get(name, 0) for c in per_step_counts], dtype=float
            )
            for name in names
        },
        skipped=np.asarray(
            [c.get("skipped", 0) for c in per_step_counts], dtype=float
        ),
        n_ranks=scan.n_ranks,
        ranks_seen=sorted(ranks_seen),
        system_total=used_system,
    )


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
    step_range: Sequence[int] | None = None,
) -> None:
    """Analyse engine_print_task_counts blocks in SWIFT log files.

    Prints a summary table per log and plots the total tasks per step and
    the per-type counts per step (one marker per log).

    Args:
        log_files: SWIFT logs to analyse (plotted together).
        labels: Optional labels matching ``log_files`` (default: stems).
        output_path: Directory whose ``task_counts_analysis``
            subdirectory receives the plots (default: current directory).
        prefix: Optional prefix added to each output filename.
        show_plot: Whether to display plots interactively.
        task_filter: Optional task types to include (names as printed in
            the log); unknown names are an error.
        step_range: Optional inclusive ``(start, end)`` step range.

    Logs without task counts are reported as warnings and skipped.

    Raises:
        ValueError: If no log has task counts, or for unknown task names.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    print(
        "Analysing engine_print_task_counts in "
        f"{plural(len(log_files), 'log file')}"
    )
    labels = resolve_labels(log_files, labels)
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    results: list[tuple[str, str, TaskCounts]] = []
    for index, log_file in enumerate(log_files):
        print(f"\nLog {index + 1}/{len(log_files)}: {log_file}")
        counts = task_count_series(log_file, task_filter, step_range)
        if not counts.steps:
            warn(f"no task counts found in {log_file}; skipping it")
            continue
        print(f"  Found task counts for {plural(len(counts.steps), 'step')}")
        if counts.partial:
            seen = ", ".join(map(str, counts.ranks_seen))
            warn(
                f"{log_file}: task counts from rank(s) {seen} only, of "
                f"{counts.n_ranks} ranks; per-type counts"
                + (
                    ""
                    if counts.system_total and not task_filter
                    else " and totals"
                )
                + " cover those ranks only"
            )
        results.append((log_file, labels[index], counts))

    if not results:
        raise ValueError(
            "no engine_print_task_counts output found in any log"
            + (" in the requested steps" if step_range is not None else "")
            + "; run SWIFT with -v 1"
        )

    if task_filter:
        known = set().union(*(c.task_series for _, _, c in results))
        unknown = [t for t in task_filter if t not in known]
        if unknown:
            raise ValueError(
                f"unknown task name(s) {', '.join(unknown)}; the logs list: "
                + ", ".join(sorted(known))
            )
        print(f"Filtering for specific tasks: {', '.join(task_filter)}")

    for log_file, _, counts in results:
        _print_summary(log_file, counts, task_filter)

    scope = " (rank 0 only)" if all(c.partial for *_, c in results) else ""
    if task_filter:
        total_label = f"Tasks per step ({', '.join(task_filter)}){scope}"
    else:
        total_label = "Total tasks per step (incl. skipped)" + (
            "" if all(c.system_total for *_, c in results) else scope
        )

    # Plot 1: total tasks per step
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, (_, label, counts) in enumerate(results):
        ax.scatter(
            counts.sim_times,
            counts.totals,
            marker=markers[index % len(markers)],
            s=20,
            alpha=0.7,
            label=label,
        )
    if len(results) > 1:
        ax.legend()
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(total_label)
    ax.set_title("engine_print_task_counts: tasks per step")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    outputs = [
        save_figure(
            fig,
            create_output_path(
                output_path, prefix, "task_counts_per_step.png", OUT_SUBDIR
            ),
            show=show_plot,
        )
    ]

    # Plot 2: per-type counts per step (skipped tasks are not a type)
    names = task_filter or sorted(
        name
        for name in set().union(*(c.task_series for *_, c in results))
        if any(
            np.any(c.task_series.get(name, np.zeros(1)) > 0)
            for *_, c in results
        )
    )
    if names:
        colours = colour_map(names)
        fig, ax = plt.subplots(figsize=(12, 7))
        for index, (_, _, counts) in enumerate(results):
            times = np.asarray(counts.sim_times)
            for name in names:
                values = counts.task_series.get(name)
                if values is None or not np.any(values > 0):
                    continue
                ax.scatter(
                    times[values > 0],
                    values[values > 0],
                    marker=markers[index % len(markers)],
                    s=20,
                    alpha=0.7,
                    color=colours[name],
                )
        ax.set_xlabel(TIME_LABEL)
        ax.set_ylabel(f"Active tasks per step{scope}")
        ax.set_title(
            "engine_print_task_counts: tasks per type (excl. skipped)"
        )
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle="--", which="both")
        # The log legend sits inside the axes; the type legend (the axes'
        # own legend, so bbox_inches="tight" keeps it) to the right.
        if len(results) > 1:
            ax.add_artist(
                ax.legend(
                    handles=[
                        Line2D(
                            [0],
                            [0],
                            marker=markers[i % len(markers)],
                            color="w",
                            markerfacecolor="grey",
                            markersize=7,
                            label=label,
                        )
                        for i, (_, label, _) in enumerate(results)
                    ],
                    title="Log",
                    loc="best",
                    fontsize="small",
                )
            )
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colours[n],
                    markersize=7,
                    label=n,
                )
                for n in names
            ],
            title="Task type",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            ncol=math.ceil(len(names) / 25),
            fontsize="small",
            borderaxespad=0.0,
        )
        outputs.append(
            save_figure(
                fig,
                create_output_path(
                    output_path, prefix, "task_counts_by_type.png", OUT_SUBDIR
                ),
                show=show_plot,
            )
        )

    print("\nCreated task-count plots:")
    for path in outputs:
        print(f"  - {path}")


TIME_LABEL = "Simulation time [internal units]"


def _print_summary(
    log_file: str, counts: TaskCounts, task_filter: list[str] | None
) -> None:
    """Print the per-log summary table."""
    totals = np.asarray(counts.totals, dtype=float)
    first = (
        f"SELECTED ({', '.join(task_filter)})"
        if task_filter
        else "ALL (incl. skipped)"
    )
    print(f"\n  Task count summary: {log_file}")
    print("  " + "=" * 78)
    print(
        f"  {'Task Type':<28} {'Total':<15} {'Avg/Step':<15} {'Max/Step':<15}"
    )
    print("  " + "-" * 78)
    print(
        f"  {first[:28]:<28} {totals.sum():<15,.0f} {totals.mean():<15,.2f} "
        f"{totals.max():<15,.0f}"
    )
    print("  " + "-" * 78)
    rows = sorted(
        counts.task_series.items(), key=lambda item: -float(item[1].sum())
    )
    rows.append(("skipped (any type)", counts.skipped))
    for name, values in rows:
        print(
            f"  {name:<28} {values.sum():<15,.0f} {values.mean():<15,.2f} "
            f"{values.max():<15,.0f}"
        )
    print("  " + "=" * 78)
    print(
        "  Per-type rows count tasks that run; skipped tasks are counted "
        "once under 'skipped'."
    )
