"""Combined task and threadpool plotting for SWIFT analysis."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence, TypeVar

import numpy as np
from tqdm.auto import tqdm

from swiftsim_cli.modes.common_args import (
    add_output_arguments,
    add_rank_argument,
    existing_file,
    non_negative_int,
    positive_float,
    positive_int,
)
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.plotting import save_figure
from swiftsim_cli.utilities.tables import create_ascii_table

from .task_debug_data import TaskDebugData, TaskFilter, TaskRecord
from .task_plot_metadata import CELL_SUBTYPES, CELL_TYPES
from .threadpool_data import ThreadpoolData, ThreadpoolRecord

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "add_task_plots_arguments",
    "run_swift_task_plots",
    "analyse_swift_task_plots",
]


# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "task_plots_analysis"


@dataclass(frozen=True)
class PlotInterval:
    """A normalised interval ready for plotting."""

    lane: int
    start_ms: float
    end_ms: float
    colour: str
    label: str


def add_task_plots_arguments(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add CLI arguments for combined task/threadpool plotting."""
    parser = subparsers.add_parser(
        "task-plots",
        help=(
            "Plot SWIFT task-debug and threadpool timings on a shared "
            "wall-clock axis."
        ),
        description="Plot SWIFT task-debug and threadpool timings on a "
        "shared wall-clock axis. Requires SWIFT configured with "
        "--enable-task-debugging and --enable-threadpool-debugging and run "
        "with -y N and -Y N.",
    )
    parser.set_defaults(func=run_swift_task_plots)

    parser.add_argument(
        "task_file",
        type=existing_file,
        metavar="TASK_FILE",
        help="Task-debug file produced by SWIFT's -y output.",
    )
    parser.add_argument(
        "threadpool_file",
        type=existing_file,
        metavar="THREADPOOL_FILE",
        help=(
            "Threadpool dump produced by SWIFT's -Y output. Threadpool dumps "
            "are per rank: for multi-rank task files the dump is only "
            "overlaid on its own rank, taken from SWIFT's "
            "'threadpool_info-rank<N>-step<M>.dat' name, or from --rank if "
            "the name carries no rank. Other ranks are plotted without the "
            "threadpool panel."
        ),
    )
    add_rank_argument(
        parser,
        default="all",
        help="MPI rank to plot from multi-rank task files; 'all' gives one "
        "set of plots per rank.",
    )
    parser.add_argument(
        "--limit",
        type=positive_float,
        default=0.0,
        metavar="MS",
        help="Upper time limit in ms (default: the data range).",
    )
    parser.add_argument(
        "--mintic",
        type=non_negative_int,
        default=None,
        metavar="TICK",
        help="Absolute tick to use as the shared start time "
        "(default: the earliest tick).",
    )
    parser.add_argument(
        "--expand",
        type=positive_int,
        default=1,
        metavar="N",
        help="Expansion factor for each thread lane (default: 1).",
    )
    parser.add_argument(
        "--width",
        type=positive_float,
        default=16.0,
        metavar="IN",
        help="Figure width in inches (default: 16).",
    )
    parser.add_argument(
        "--height",
        type=positive_float,
        default=10.0,
        metavar="IN",
        help="Figure height in inches (default: 10).",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        default=False,
        help="Suppress legends on generated figures.",
    )
    add_output_arguments(parser, OUT_SUBDIR)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        metavar="NAME",
        help="Task labels to include, e.g. 'pair/force' (default: all).",
    )
    for flag, cell, kind, parse in (
        ("--c-type", "either cell", "type", _parse_cell_type),
        ("--ci-type", "the ci cell", "type", _parse_cell_type),
        ("--cj-type", "the cj cell", "type", _parse_cell_type),
        ("--c-subtype", "either cell", "subtype", _parse_cell_subtype),
        ("--ci-subtype", "the ci cell", "subtype", _parse_cell_subtype),
        ("--cj-subtype", "the cj cell", "subtype", _parse_cell_subtype),
    ):
        parser.add_argument(
            flag,
            type=parse,
            default=None,
            metavar=kind.upper(),
            help=f"Include tasks where {cell} has this {kind}.",
        )
    parser.add_argument(
        "--depth",
        type=non_negative_int,
        default=None,
        metavar="N",
        help="Include tasks where either cell is at this tree depth.",
    )
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
        output_path=args.out,
        prefix=args.prefix,
        rank=None if args.rank == "all" else args.rank,
        limit=args.limit,
        mintic=args.mintic,
        expand=args.expand,
        width=args.width,
        height=args.height,
        show_plot=args.show,
        show_legend=not args.no_legend,
        task_filter=TaskFilter(
            tasks=tuple(args.tasks) if args.tasks else None,
            c_type=args.c_type,
            ci_type=args.ci_type,
            cj_type=args.cj_type,
            c_subtype=args.c_subtype,
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
    print(f"Loading threadpool data from {threadpool_file}...", flush=True)
    threadpool_data = ThreadpoolData(threadpool_file)
    ranks = _select_ranks(task_data, rank)
    threadpool_rank = _threadpool_rank(threadpool_file, task_data, rank)
    filter_suffix = _task_filter_suffix(task_filter)
    output_root = str(output_path) if output_path is not None else None
    output_prefix = prefix or _step_prefix(task_file, threadpool_file)
    out_dir = OUT_SUBDIR

    for current_rank in ranks:
        task_records = task_data.get_rank_records(current_rank, task_filter)
        if not task_records:
            raise ValueError(
                "No task records matched the selected filters "
                f"for rank {current_rank}."
            )

        # A threadpool dump only describes the rank that wrote it.
        rank_pool = (
            threadpool_data if current_rank == threadpool_rank else None
        )
        start_ms, window_ms = _time_window(
            task_data=task_data,
            threadpool_data=rank_pool,
            rank=current_rank,
            limit=limit,
            mintic=mintic,
        )

        print(
            f"Rendering combined timeline for rank {current_rank}...",
            flush=True,
        )
        figure = _plot_combined_timeline(
            task_records=task_records,
            threadpool_records=(
                rank_pool.worker_records if rank_pool is not None else []
            ),
            background_records=(
                rank_pool.background_records if rank_pool is not None else []
            ),
            task_thread_count=task_data.thread_counts[current_rank],
            threadpool_thread_count=(
                rank_pool.thread_count if rank_pool is not None else 0
            ),
            start_ms=start_ms,
            window_ms=window_ms,
            expand=expand,
            width=width,
            height=height,
            show_legend=show_legend,
        )
        filename = f"task_plots_rank{current_rank}{filter_suffix}.png"
        output = create_output_path(
            output_root, output_prefix, filename, out_dir
        )
        print(f"Saving combined timeline to {output}...", flush=True)
        save_figure(figure, output, show=show_plot)

        report = _task_timing_report(task_records, current_rank)
        print("\n" + report)
        report_name = (
            f"task_timing_tables_rank{current_rank}{filter_suffix}.txt"
        )
        report_output = create_output_path(
            output_root, output_prefix, report_name, out_dir
        )
        report_output.write_text(report + "\n", encoding="utf-8")
        print(f"Saved task timing tables to {report_output}")

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
                output_prefix,
                activity_name,
                out_dir,
            )
            save_figure(activity_figure, activity_output, show=show_plot)


def _select_ranks(task_data: TaskDebugData, rank: int | None) -> list[int]:
    """Select the ranks to emit for the current command invocation."""
    if rank is None:
        return task_data.ranks
    if rank not in task_data.ranks:
        raise ValueError(f"Rank {rank} not present in {task_data.file_path}")
    return [rank]


def _threadpool_rank(
    threadpool_file: str | Path,
    task_data: TaskDebugData,
    rank: int | None,
) -> int:
    """Return the rank a threadpool dump belongs to.

    SWIFT names MPI dumps ``threadpool_info-rank<N>-step<M>.dat``. Without
    a rank in the name, single-rank task files use their only rank and
    multi-rank task files require an explicit ``--rank``.
    """
    match = re.search(
        r"(?:^|[-_])rank(\d+)(?:\D|$)", Path(threadpool_file).stem
    )
    if match is not None:
        file_rank = int(match.group(1))
        if rank is not None and rank != file_rank:
            raise ValueError(
                f"Threadpool file {threadpool_file} is for rank {file_rank} "
                f"but --rank {rank} was requested."
            )
        return file_rank
    if rank is not None:
        return rank
    if len(task_data.ranks) > 1:
        raise ValueError(
            f"Task file {task_data.file_path} has {len(task_data.ranks)} "
            f"ranks but threadpool file {threadpool_file} does not encode "
            "one; pass --rank to choose the matching rank."
        )
    return task_data.ranks[0]


def _step_prefix(
    task_file: str | Path, threadpool_file: str | Path
) -> str | None:
    """Infer a shared ``stepN`` output prefix from input filenames."""
    steps = [
        match.group(1)
        for path in (task_file, threadpool_file)
        if (
            match := re.search(r"(?:^|[-_])step(\d+)(?:\D|$)", Path(path).stem)
        )
    ]
    if len(set(steps)) > 1:
        raise ValueError(
            "Task and threadpool files have different step numbers."
        )
    return f"step{steps[0]}" if steps else None


def _time_window(
    task_data: TaskDebugData,
    threadpool_data: ThreadpoolData | None,
    rank: int,
    limit: float,
    mintic: int | None,
) -> tuple[float, float]:
    """Compute the common time origin and x-axis span in ms."""
    task_start, task_end = task_data.rank_bounds_ms(rank)
    pool_start, pool_end = (
        threadpool_data.bounds_ms()
        if threadpool_data is not None
        else (task_start, task_end)
    )

    if mintic is not None:
        start_ms = task_data.mintic_to_ms(mintic, rank)
        if threadpool_data is not None:
            start_ms = min(start_ms, threadpool_data.mintic_to_ms(mintic))
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
) -> Figure:
    """Render the combined top-threadpool / bottom-task timeline figure."""
    import matplotlib.pyplot as plt

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

    ax_top.set_title(
        "Threadpool"
        if threadpool_records or background_records
        else "Threadpool (no dump for this rank)"
    )
    ax_bottom.set_title("Tasks")
    ax_bottom.set_xlabel("Wall clock time [ms]")

    if show_legend:
        _add_legend(ax_top, top_intervals, ncol=4, above=True)
        _add_legend(ax_bottom, bottom_intervals, ncol=8, above=False)

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
) -> Figure:
    """Render a task-activity heatmap for a filtered task selection."""
    import matplotlib.pyplot as plt

    if thread_count < 1:
        raise ValueError("Task activity plots require at least one thread.")

    grid: np.ndarray = np.zeros((thread_count, xbins), dtype=float)
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
    axis: Axes,
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
    axis: Axes, intervals: Sequence[PlotInterval]
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
    axis: Axes,
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
    axis: Axes,
    intervals: Sequence[PlotInterval],
    ncol: int,
    above: bool,
) -> None:
    """Add a legend sorted by label frequency for one panel."""
    import matplotlib.patches as mpatches

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
            loc="lower center" if above else "upper center",
            bbox_to_anchor=(0.5, 1.02 if above else -0.16),
            ncol=ncol,
        )


def _task_timing_report(records: Sequence[TaskRecord], rank: int) -> str:
    """Build task and cell timing tables for one rank."""
    task_times: defaultdict[str, float] = defaultdict(float)
    cell_times: defaultdict[tuple[str, str], float] = defaultdict(float)
    combination_times: defaultdict[str, float] = defaultdict(float)
    task_combination_times: defaultdict[tuple[str, str], float] = defaultdict(
        float
    )

    for record in tqdm(
        records,
        desc=f"Aggregating task timings (rank {rank})",
        unit="task",
    ):
        duration = record.end_ms - record.start_ms
        task_times[record.label] += duration

        cell_pair = _cell_pair_label(record, include_subtype=False)
        combination = _cell_pair_label(record, include_subtype=True)
        if cell_pair is not None:
            cell_times[(record.label, cell_pair)] += duration
        if combination is not None:
            combination_times[combination] += duration
            task_combination_times[(record.label, combination)] += duration

    total = sum(task_times.values())
    sections = [
        f"TASK TIMING SUMMARY: RANK {rank}",
        f"Total task worker time: {total:.3f} ms",
        create_ascii_table(
            ["Task Type", "Time (ms)", "Fraction"],
            _timing_rows(task_times, total),
            "TASK TYPE TIMING",
        ),
    ]

    if cell_times:
        sections.extend(
            [
                create_ascii_table(
                    ["Task Type", "Cell Type Pair", "Time (ms)", "Fraction"],
                    _timing_rows(cell_times, total),
                    "CELL TYPE TIMING BY TASK TYPE",
                ),
                create_ascii_table(
                    ["Cell Type/Subtype Pair", "Time (ms)", "Fraction"],
                    _timing_rows(combination_times, total),
                    "CELL TYPE/SUBTYPE TIMING (ALL TASK TYPES)",
                ),
                create_ascii_table(
                    [
                        "Task Type",
                        "Cell Type/Subtype Pair",
                        "Time (ms)",
                        "Fraction",
                    ],
                    _timing_rows(task_combination_times, total),
                    "CELL TYPE/SUBTYPE TIMING BY TASK TYPE",
                ),
            ]
        )

    return "\n\n".join(sections)


K = TypeVar("K")


def _timing_rows(timings: Mapping[K, float], total: float) -> list[list[str]]:
    """Format timing mappings as descending ASCII-table rows."""
    rows: list[list[str]] = []
    for key, duration in sorted(
        timings.items(), key=lambda item: item[1], reverse=True
    ):
        labels = (
            [str(label) for label in key]
            if isinstance(key, tuple)
            else [str(key)]
        )
        fraction = 100.0 * duration / total if total else 0.0
        rows.append([*labels, f"{duration:.3f}", f"{fraction:.2f}%"])
    return rows


def _cell_pair_label(record: TaskRecord, include_subtype: bool) -> str | None:
    """Return an order-independent label for available task cell metadata."""
    cells = []
    for cell_type, subtype in (
        (record.ci_type, record.ci_subtype),
        (record.cj_type, record.cj_subtype),
    ):
        if not 0 <= cell_type < len(CELL_TYPES):
            continue
        label = CELL_TYPES[cell_type]
        if include_subtype and 0 <= subtype < len(CELL_SUBTYPES):
            subtype_label = CELL_SUBTYPES[subtype]
            if subtype_label is not None:
                label = f"{label}/{subtype_label}"
        cells.append(label)

    if not cells:
        return None
    return " + ".join(sorted(cells))


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
    if task_filter.tasks is not None:
        suffixes.append(
            "+".join(t.replace("/", "-") for t in task_filter.tasks)
        )
    if task_filter.c_type is not None:
        suffixes.append(f"c{CELL_TYPES[task_filter.c_type].lower()}")
    if task_filter.ci_type is not None:
        suffixes.append(f"ci{task_filter.ci_type}")
    if task_filter.cj_type is not None:
        suffixes.append(f"cj{task_filter.cj_type}")
    if task_filter.c_subtype is not None:
        suffixes.append(f"csub{CELL_SUBTYPES[task_filter.c_subtype].lower()}")
    if task_filter.ci_subtype is not None:
        suffixes.append(f"cisub{task_filter.ci_subtype}")
    if task_filter.cj_subtype is not None:
        suffixes.append(f"cjsub{task_filter.cj_subtype}")
    if task_filter.depth is not None:
        suffixes.append(f"depth{task_filter.depth}")

    if not suffixes:
        return ""
    return "_" + "_".join(suffixes)


def _parse_cell_type(value: str) -> int:
    """Parse a SWIFT cell type name or its integer value."""
    return _parse_cell_metadata(value, CELL_TYPES, {"bkg": "background"})


def _parse_cell_subtype(value: str) -> int:
    """Parse a SWIFT cell subtype name or its integer value."""
    return _parse_cell_metadata(value, CELL_SUBTYPES)


def _parse_cell_metadata(
    value: str, labels: Sequence[str], aliases: dict[str, str] | None = None
) -> int:
    """Parse named metadata while retaining numeric CLI compatibility."""
    normalised = value.lower()
    if aliases is not None:
        normalised = aliases.get(normalised, normalised)
    names = {label.lower(): index for index, label in enumerate(labels)}
    if normalised in names:
        return names[normalised]
    try:
        index = int(value)
    except ValueError as error:
        choices = ", ".join(names)
        raise argparse.ArgumentTypeError(
            f"Expected one of {choices}, or its integer value."
        ) from error
    if not 0 <= index < len(labels):
        raise argparse.ArgumentTypeError(
            f"Cell metadata value must be between 0 and {len(labels) - 1}."
        )
    return index
