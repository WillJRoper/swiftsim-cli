"""Combined task and threadpool plotting for SWIFT analysis."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from swiftsim_cli.utilities import create_output_path

from .task_debug_data import TaskDebugData, TaskFilter, TaskRecord
from .threadpool_data import ThreadpoolData, ThreadpoolRecord

__all__ = [
    "add_task_plots_arguments",
    "run_swift_task_plots",
    "analyse_swift_task_plots",
]


@dataclass(frozen=True)
class PlotInterval:
    """A normalised interval ready for plotting."""

    lane: int
    start_ms: float
    end_ms: float
    colour: str
    label: str


def add_task_plots_arguments(subparsers) -> None:
    """Add CLI arguments for combined task/threadpool plotting."""
    parser = subparsers.add_parser(
        "task-plots",
        help=(
            "Plot SWIFT threadpool and task-debug timings on a shared "
            "wall-clock axis."
        ),
    )

    parser.add_argument(
        "task_file",
        type=Path,
        help="Task-debug file produced by SWIFT's -y output.",
    )
    parser.add_argument(
        "threadpool_file",
        type=Path,
        help="Threadpool dump produced by SWIFT's -Y output.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=(
            "Specific rank to plot for MPI task-debug files. By default all "
            "ranks in the task file are emitted."
        ),
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=0.0,
        help="Upper time limit in ms. Defaults to the data range.",
    )
    parser.add_argument(
        "--mintic",
        type=int,
        default=None,
        help="Absolute tick to use as the shared start time.",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=1,
        help="Expansion factor for each thread lane.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=16.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=10.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        default=False,
        help="Suppress legends on generated figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show the generated figures interactively.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        default=None,
        help="Where to save the analysis outputs.",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default=None,
        help="Optional filename prefix for generated outputs.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task label to include, e.g. 'pair/force'.",
    )
    parser.add_argument("--ci-type", type=int, default=None)
    parser.add_argument("--cj-type", type=int, default=None)
    parser.add_argument("--ci-subtype", type=int, default=None)
    parser.add_argument("--cj-subtype", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument(
        "--activity-plot",
        action="store_true",
        default=False,
        help="Also create task-activity heatmaps for each selected rank.",
    )
    parser.add_argument(
        "--sort-threads",
        action="store_true",
        default=False,
        help="Sort activity-plot rows by total active time.",
    )


def run_swift_task_plots(args: argparse.Namespace) -> None:
    """Entry point for the ``analyse task-plots`` subcommand."""
    analyse_swift_task_plots(
        task_file=args.task_file,
        threadpool_file=args.threadpool_file,
        output_path=args.output_path,
        prefix=args.prefix,
        rank=args.rank,
        limit=args.limit,
        mintic=args.mintic,
        expand=args.expand,
        width=args.width,
        height=args.height,
        show_plot=args.show,
        show_legend=not args.no_legend,
        task_filter=TaskFilter(
            task=args.task,
            ci_type=args.ci_type,
            cj_type=args.cj_type,
            ci_subtype=args.ci_subtype,
            cj_subtype=args.cj_subtype,
            depth=args.depth,
        ),
        activity_plot=args.activity_plot,
        sort_threads=args.sort_threads,
    )


def analyse_swift_task_plots(
    task_file: str | Path,
    threadpool_file: str | Path,
    output_path: str | Path | None = None,
    prefix: str | None = None,
    rank: int | None = None,
    limit: float = 0.0,
    mintic: int | None = None,
    expand: int = 1,
    width: float = 16.0,
    height: float = 10.0,
    show_plot: bool = False,
    show_legend: bool = True,
    task_filter: TaskFilter | None = None,
    activity_plot: bool = False,
    sort_threads: bool = False,
) -> None:
    """Create combined task/threadpool plots and optional activity heatmaps."""
    if expand < 1:
        raise ValueError("Thread expansion factor must be at least 1.")

    task_data = TaskDebugData(task_file)
    threadpool_data = ThreadpoolData(threadpool_file)
    ranks = _select_ranks(task_data, rank)
    out_dir = "task_plots_analysis"
    filter_suffix = _task_filter_suffix(task_filter)
    output_root = str(output_path) if output_path is not None else None

    for current_rank in ranks:
        task_records = task_data.get_rank_records(current_rank, task_filter)
        if not task_records:
            raise ValueError(
                "No task records matched the selected filters "
                f"for rank {current_rank}."
            )

        start_ms, window_ms = _time_window(
            task_data=task_data,
            threadpool_data=threadpool_data,
            rank=current_rank,
            limit=limit,
            mintic=mintic,
        )

        figure = _plot_combined_timeline(
            task_records=task_records,
            threadpool_records=threadpool_data.worker_records,
            background_records=threadpool_data.background_records,
            task_thread_count=task_data.thread_counts[current_rank],
            threadpool_thread_count=threadpool_data.thread_count,
            start_ms=start_ms,
            window_ms=window_ms,
            expand=expand,
            width=width,
            height=height,
            show_legend=show_legend,
        )
        filename = f"task_plots_rank{current_rank}{filter_suffix}.png"
        output = create_output_path(output_root, prefix, filename, out_dir)
        figure.savefig(output, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(figure)

        if activity_plot:
            activity_figure = _plot_task_activity(
                task_records=task_records,
                thread_count=task_data.thread_counts[current_rank],
                start_ms=start_ms,
                window_ms=window_ms,
                width=width,
                height=height,
                sort_threads=sort_threads,
            )
            activity_name = (
                f"task_activity_rank{current_rank}{filter_suffix}.png"
            )
            activity_output = create_output_path(
                output_root,
                prefix,
                activity_name,
                out_dir,
            )
            activity_figure.savefig(
                activity_output, dpi=300, bbox_inches="tight"
            )
            if show_plot:
                plt.show()
            plt.close(activity_figure)


def _select_ranks(task_data: TaskDebugData, rank: int | None) -> list[int]:
    """Select the ranks to emit for the current command invocation."""
    if rank is None:
        return task_data.ranks
    if rank not in task_data.ranks:
        raise ValueError(f"Rank {rank} not present in {task_data.file_path}")
    return [rank]


def _time_window(
    task_data: TaskDebugData,
    threadpool_data: ThreadpoolData,
    rank: int,
    limit: float,
    mintic: int | None,
) -> tuple[float, float]:
    """Compute the common time origin and x-axis span in ms."""
    task_start, task_end = task_data.rank_bounds_ms(rank)
    pool_start, pool_end = threadpool_data.bounds_ms()

    if mintic is not None:
        start_ms = min(
            task_data.mintic_to_ms(mintic),
            threadpool_data.mintic_to_ms(mintic),
        )
    else:
        start_ms = min(task_start, pool_start)

    auto_limit = max(task_end, pool_end) - start_ms
    window_ms = limit if limit > 0 else auto_limit
    if window_ms <= 0:
        raise ValueError("Computed plot window is not positive.")
    return start_ms, window_ms


def _plot_combined_timeline(
    task_records: Sequence[TaskRecord],
    threadpool_records: Sequence[ThreadpoolRecord],
    background_records: Sequence[ThreadpoolRecord],
    task_thread_count: int,
    threadpool_thread_count: int,
    start_ms: float,
    window_ms: float,
    expand: int,
    width: float,
    height: float,
    show_legend: bool,
) -> plt.Figure:
    """Render the combined top-threadpool / bottom-task timeline figure."""
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(width, height),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.15},
    )

    top_intervals = _expand_intervals(
        threadpool_records,
        threadpool_thread_count,
        expand,
        start_ms,
        window_ms,
        lambda record: record.thread,
        lambda record: record.start_ms,
        lambda record: record.end_ms,
        lambda record: record.colour,
        lambda record: record.function,
    )
    bottom_intervals = _expand_intervals(
        task_records,
        task_thread_count,
        expand,
        start_ms,
        window_ms,
        lambda record: record.thread,
        lambda record: record.start_ms,
        lambda record: record.end_ms,
        lambda record: record.colour,
        lambda record: record.label,
    )

    _plot_background_band(
        ax_top,
        background_records,
        max(threadpool_thread_count * expand, 1),
        start_ms,
        window_ms,
    )
    _draw_lane_intervals(ax_top, top_intervals)
    _draw_lane_intervals(ax_bottom, bottom_intervals)

    _format_timeline_axis(
        ax_top,
        threadpool_thread_count,
        expand,
        "Threadpool Thread ID",
    )
    _format_timeline_axis(
        ax_bottom,
        task_thread_count,
        expand,
        "Task Thread ID",
    )

    for axis in (ax_top, ax_bottom):
        axis.set_xlim(-0.01 * window_ms, 1.01 * window_ms)
        axis.axvline(0.0, color="k", linestyle="--", linewidth=1)
        axis.axvline(window_ms, color="k", linestyle="--", linewidth=1)

    ax_top.set_title("Threadpool")
    ax_bottom.set_title("Tasks")
    ax_bottom.set_xlabel("Wall clock time [ms]")

    if show_legend:
        _add_legend(ax_top, top_intervals, ncol=4)
        _add_legend(ax_bottom, bottom_intervals, ncol=8)

    return fig


def _plot_task_activity(
    task_records: Sequence[TaskRecord],
    thread_count: int,
    start_ms: float,
    window_ms: float,
    width: float,
    height: float,
    sort_threads: bool,
    xbins: int = 1000,
) -> plt.Figure:
    """Render a task-activity heatmap for a filtered task selection."""
    if thread_count < 1:
        raise ValueError("Task activity plots require at least one thread.")

    grid = np.zeros((thread_count, xbins), dtype=float)
    for record in task_records:
        start_bin = int(
            np.clip(
                (record.start_ms - start_ms) / window_ms * xbins, 0, xbins - 1
            )
        )
        end_bin = int(
            np.clip((record.end_ms - start_ms) / window_ms * xbins, 0, xbins)
        )
        if end_bin <= start_bin:
            end_bin = min(start_bin + 1, xbins)
        grid[record.thread, start_bin:end_bin] = 1.0

    if sort_threads:
        order = np.argsort(grid.sum(axis=1))[::-1]
        grid = grid[order, :]

    fig, (ax_hist, ax_grid) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(width, height),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 8], "hspace": 0.05},
    )
    x_edges = np.linspace(0.0, window_ms, xbins + 1)
    y_edges = np.arange(thread_count + 1)
    mesh = ax_grid.pcolormesh(
        x_edges, y_edges, grid, cmap="coolwarm", shading="auto"
    )
    fig.colorbar(mesh, ax=ax_grid, ticks=[0, 1], label="Activity")

    ax_grid.set_ylabel("Thread ID")
    ax_grid.set_xlabel("Wall clock time [ms]")
    ax_grid.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax_grid.axvline(window_ms, color="k", linestyle="--", linewidth=1)

    counts = grid.sum(axis=0)
    ax_hist.bar(x_edges[:-1], counts, width=np.diff(x_edges), align="edge")
    ax_hist.set_ylabel("Tasks Running")
    ax_hist.grid(True)
    ax_hist.tick_params(
        axis="x", which="both", bottom=False, labelbottom=False
    )
    return fig


def _plot_background_band(
    axis: plt.Axes,
    records: Sequence[ThreadpoolRecord],
    total_lanes: int,
    start_ms: float,
    window_ms: float,
) -> None:
    """Draw the faint background occupancy spans from the threadpool dump."""
    spans: list[tuple[float, float]] = []
    colours: list[str] = []
    for record in records:
        clipped = _clip_interval(
            record.start_ms, record.end_ms, start_ms, window_ms
        )
        if clipped is None:
            continue
        spans.append(clipped)
        colours.append(record.colour)

    if spans:
        axis.broken_barh(
            spans,
            (0.0, max(total_lanes - 0.1, 0.9)),
            facecolors=colours,
            linewidth=0,
            alpha=0.15,
        )


T = TypeVar("T")


def _expand_intervals(
    records: Sequence[T],
    thread_count: int,
    expand: int,
    start_ms: float,
    window_ms: float,
    thread_getter: Callable[[T], int],
    start_getter: Callable[[T], float],
    end_getter: Callable[[T], float],
    colour_getter: Callable[[T], str],
    label_getter: Callable[[T], str],
) -> list[PlotInterval]:
    """Expand each lane in the same round-robin style as SWIFT's plots."""
    by_thread: dict[int, list[T]] = defaultdict(list)
    for record in records:
        by_thread[thread_getter(record)].append(record)

    expanded: list[PlotInterval] = []
    for thread in range(thread_count):
        thread_records = sorted(by_thread.get(thread, []), key=start_getter)
        for index, record in enumerate(thread_records):
            clipped = _clip_interval(
                start_getter(record),
                end_getter(record),
                start_ms,
                window_ms,
            )
            if clipped is None:
                continue
            lane = thread * expand + (index % expand)
            expanded.append(
                PlotInterval(
                    lane=lane,
                    start_ms=clipped[0],
                    end_ms=clipped[0] + clipped[1],
                    colour=colour_getter(record),
                    label=label_getter(record),
                )
            )
    return expanded


def _draw_lane_intervals(
    axis: plt.Axes, intervals: Sequence[PlotInterval]
) -> None:
    """Draw lane intervals using ``broken_barh`` grouped by lane."""
    by_lane: dict[int, list[PlotInterval]] = defaultdict(list)
    for interval in intervals:
        by_lane[interval.lane].append(interval)

    for lane, lane_intervals in by_lane.items():
        spans = [
            (interval.start_ms, interval.end_ms - interval.start_ms)
            for interval in lane_intervals
        ]
        colours = [interval.colour for interval in lane_intervals]
        axis.broken_barh(
            spans, (lane + 0.05, 0.9), facecolors=colours, linewidth=0
        )


def _format_timeline_axis(
    axis: plt.Axes,
    thread_count: int,
    expand: int,
    ylabel: str,
) -> None:
    """Apply the consistent thread-lane formatting used by both panels."""
    total_lanes = max(thread_count * expand, 1)
    axis.set_ylim(0.0, total_lanes + 0.5)
    axis.set_ylabel(ylabel if expand == 1 else f"{ylabel} * {expand}")
    axis.set_axisbelow(True)
    axis.grid(True, axis="y")


def _add_legend(
    axis: plt.Axes, intervals: Sequence[PlotInterval], ncol: int
) -> None:
    """Add a legend sorted by label frequency for one panel."""
    labels_by_count = [
        label
        for label, _ in Counter(
            interval.label for interval in intervals
        ).most_common()
    ]
    colour_by_label = {
        interval.label: interval.colour for interval in intervals
    }
    handles = [
        mpatches.Patch(color=colour_by_label[label], label=label)
        for label in labels_by_count
    ]
    if handles:
        axis.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.02),
            ncol=ncol,
        )


def _clip_interval(
    start_ms: float,
    end_ms: float,
    origin_ms: float,
    window_ms: float,
) -> tuple[float, float] | None:
    """Clip one interval into the plot window and return ``(start, width)``."""
    clipped_start = max(start_ms, origin_ms)
    clipped_end = min(end_ms, origin_ms + window_ms)
    if clipped_end <= clipped_start:
        return None
    return clipped_start - origin_ms, clipped_end - clipped_start


def _task_filter_suffix(task_filter: TaskFilter | None) -> str:
    """Build a filename suffix describing the active task filter."""
    if task_filter is None:
        return ""

    suffixes: list[str] = []
    if task_filter.task is not None:
        suffixes.append(task_filter.task.replace("/", "-"))
    if task_filter.ci_type is not None:
        suffixes.append(f"ci{task_filter.ci_type}")
    if task_filter.cj_type is not None:
        suffixes.append(f"cj{task_filter.cj_type}")
    if task_filter.ci_subtype is not None:
        suffixes.append(f"cisub{task_filter.ci_subtype}")
    if task_filter.cj_subtype is not None:
        suffixes.append(f"cjsub{task_filter.cj_subtype}")
    if task_filter.depth is not None:
        suffixes.append(f"depth{task_filter.depth}")

    if not suffixes:
        return ""
    return "_" + "_".join(suffixes)
