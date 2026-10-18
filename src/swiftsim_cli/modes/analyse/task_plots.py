"""Task (and optional threadpool) timeline plots for SWIFT task dumps."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from swiftsim_cli.modes.common_args import (
    add_output_arguments,
    add_rank_argument,
    existing_dir,
    existing_file,
    non_negative_int,
    positive_float,
    positive_int,
)
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.messages import warn
from swiftsim_cli.utilities.plotting import save_figure
from swiftsim_cli.utilities.tables import create_ascii_table

from .task_debug_data import TaskDebugData, TaskFilter
from .task_names import TaskNames
from .task_plot_metadata import colour_map
from .threadpool_data import ThreadpoolData

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "add_task_plots_arguments",
    "run_swift_task_plots",
    "analyse_swift_task_plots",
]


# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "task_plots_analysis"

# Resolution of saved figures (see save_figure) used to raster lanes.
DPI = 300

# Legend entries per column before another column is added.
LEGEND_ROWS = 28


def add_task_plots_arguments(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add CLI arguments for task/threadpool timeline plots."""
    parser = subparsers.add_parser(
        "task-plots",
        help="Plot SWIFT task-debug timings (optionally with a threadpool "
        "dump) per thread.",
        description="Plot SWIFT task-debug timings per thread, optionally "
        "with the matching threadpool dump on a shared wall-clock axis. "
        "Requires SWIFT configured with --enable-task-debugging and run with "
        "-y N (and, for --threadpool, --enable-threadpool-debugging and "
        "-Y N). Task and cell names are read from the SWIFT source (the "
        "active profile's checkout or --swift-src).",
    )
    parser.set_defaults(func=run_swift_task_plots)

    parser.add_argument(
        "task_file",
        type=existing_file,
        metavar="TASK_FILE",
        help="Task dump (thread_info-stepN.dat or thread_info_MPI-stepN.dat).",
    )
    parser.add_argument(
        "--threadpool",
        type=existing_file,
        default=None,
        metavar="FILE",
        help="Threadpool dump (-Y output) to plot above the tasks. Dumps are "
        "per rank: it is drawn only on its own rank, taken from SWIFT's "
        "'threadpool_info-rank<N>-step<M>.dat' name, or from --rank if the "
        "name carries no rank (default: tasks only).",
    )
    add_rank_argument(
        parser,
        default="all",
        help="MPI rank to plot from multi-rank task files; 'all' gives one "
        "set of plots per rank.",
    )
    parser.add_argument(
        "--swift-src",
        type=existing_dir,
        default=None,
        metavar="DIR",
        help="SWIFT source tree to read task and cell names from (default: "
        "the active profile's checkout; built-in tables if neither exists).",
    )
    parser.add_argument(
        "--limit",
        type=positive_float,
        default=0.0,
        metavar="MS",
        help="Length of the time axis in ms (default: the longest step of "
        "the plotted ranks, from the dump headers).",
    )
    parser.add_argument(
        "--mintic",
        type=non_negative_int,
        default=None,
        metavar="TICK",
        help="Absolute tick to start the time axis at (default: the step "
        "start in each rank's header).",
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
        help="Task labels to include, 'type' or 'type/subtype', e.g. "
        "'pair/force' (default: all).",
    )
    for flag, cell, kind in (
        ("--c-type", "either cell", "type"),
        ("--ci-type", "the ci cell", "type"),
        ("--cj-type", "the cj cell", "type"),
        ("--c-subtype", "either cell", "subtype"),
        ("--ci-subtype", "the ci cell", "subtype"),
        ("--cj-subtype", "the cj cell", "subtype"),
    ):
        parser.add_argument(
            flag,
            type=_parse_cell_value,
            default=None,
            metavar=kind.upper(),
            help=f"Include tasks where {cell} has this {kind}, by name "
            "(e.g. zoom, background/bkg, void) or index.",
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
        threadpool_file=args.threadpool,
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
        swift_src=args.swift_src,
    )


def analyse_swift_task_plots(
    task_file: str | Path,
    threadpool_file: str | Path | None = None,
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
    swift_src: Path | None = None,
    names: TaskNames | None = None,
) -> None:
    """Create per-rank timeline plots, timing tables and activity maps.

    Ranks where no task passes the filter are skipped with a warning.

    Raises:
        ValueError: For bad inputs, or when no rank has a matching task.
    """
    if expand < 1:
        raise ValueError("Thread expansion factor must be at least 1.")

    task_data = TaskDebugData(
        task_file,
        ranks=None if rank is None else [rank],
        names=names,
        swift_src=swift_src,
    )
    pool: ThreadpoolData | None = None
    pool_rank: int | None = None
    if threadpool_file is not None:
        print(f"Loading threadpool data from {threadpool_file}...", flush=True)
        pool = ThreadpoolData(threadpool_file)
        pool_rank = _threadpool_rank(threadpool_file, task_data, rank)
        if pool_rank not in task_data.ranks:
            warn(
                f"threadpool file {threadpool_file} is for rank {pool_rank}, "
                "which is not plotted"
            )

    mask = (
        task_data.filter_mask(task_filter)
        if task_filter is not None
        else np.ones(len(task_data.tasks), dtype=bool)
    )
    filter_suffix = _task_filter_suffix(task_filter, task_data.names)
    output_prefix = prefix or _step_prefix(task_file, threadpool_file)
    output_root = str(output_path) if output_path is not None else None
    colours = _label_colours(task_data.tasks)
    pool_colours = (
        colour_map(sorted(pool.rows["function"].unique())) if pool else {}
    )
    window_ms = (
        limit
        if limit > 0
        else max(task_data.step_duration_ms(r) for r in task_data.ranks)
    )
    if window_ms <= 0:
        raise ValueError("Computed plot window is not positive.")

    plotted = 0
    for current_rank in task_data.ranks:
        in_rank = (task_data.tasks["rank"] == current_rank).to_numpy()
        tasks = task_data.tasks.loc[in_rank & mask]
        if tasks.empty:
            warn(
                f"rank {current_rank}: no tasks matched the selected filters; "
                "skipping it"
            )
            continue
        plotted += 1
        start_ms = (
            task_data.tick_to_ms(mintic, current_rank)
            if mintic is not None
            else 0.0
        )

        pool_rows = None
        if pool is not None and current_rank == pool_rank:
            _check_overlap(pool, task_data, current_rank)
            pool_rows = pool.to_ms(task_data.step_ticks[current_rank][0])

        print(f"Rendering timeline for rank {current_rank}...", flush=True)
        figure = _plot_timeline(
            tasks=tasks,
            pool_rows=pool_rows,
            task_thread_count=task_data.thread_counts[current_rank],
            pool_thread_count=pool.thread_count if pool is not None else 0,
            start_ms=start_ms,
            window_ms=window_ms,
            expand=expand,
            width=width,
            height=height,
            show_legend=show_legend,
            colours=colours,
            pool_colours=pool_colours,
            title=f"{Path(task_file).name}: rank {current_rank}",
        )
        output = create_output_path(
            output_root,
            output_prefix,
            f"task_plots_rank{current_rank}{filter_suffix}.png",
            OUT_SUBDIR,
        )
        print(f"Saving timeline to {output}...", flush=True)
        save_figure(figure, output, show=show_plot)

        report = _task_timing_report(
            tasks, current_rank, task_data.names, task_data.layout.has_cells
        )
        print("\n" + report)
        report_output = create_output_path(
            output_root,
            output_prefix,
            f"task_timing_tables_rank{current_rank}{filter_suffix}.txt",
            OUT_SUBDIR,
        )
        report_output.write_text(report + "\n", encoding="utf-8")
        print(f"Saved task timing tables to {report_output}")

        if activity_plot:
            activity_output = create_output_path(
                output_root,
                output_prefix,
                f"task_activity_rank{current_rank}{filter_suffix}.png",
                OUT_SUBDIR,
            )
            save_figure(
                _plot_task_activity(
                    tasks=tasks,
                    thread_count=task_data.thread_counts[current_rank],
                    start_ms=start_ms,
                    window_ms=window_ms,
                    width=width,
                    height=height,
                    sort_threads=sort_threads,
                ),
                activity_output,
                show=show_plot,
            )

    if plotted == 0:
        raise ValueError(
            "no task records matched the selected filters on any rank"
        )


def _check_overlap(
    pool: ThreadpoolData, task_data: TaskDebugData, rank: int
) -> None:
    """Refuse a threadpool dump whose ticks miss the task dump's step."""
    lo, hi = pool.tick_range()
    tic, toc = task_data.step_ticks[rank]
    if hi < tic or lo > toc:
        raise ValueError(
            f"threadpool file {pool.file_path} (ticks {lo}-{hi}) does not "
            f"overlap rank {rank}'s step in {task_data.file_path} (ticks "
            f"{tic}-{toc}); are they from the same run and step?"
        )


def _label_colours(tasks: pd.DataFrame) -> dict[str, str]:
    """Distinct colours for every label in the dump, by type/subtype index.

    Computed once over all loaded ranks so a label keeps its colour in
    every rank's plot.
    """
    pairs = (
        tasks[["type", "subtype", "label"]]
        .drop_duplicates()
        .sort_values(["type", "subtype"])
    )
    return colour_map([str(label) for label in pairs["label"]])


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
    if len(task_data.file_ranks) > 1:
        raise ValueError(
            f"Task file {task_data.file_path} has "
            f"{len(task_data.file_ranks)} ranks but threadpool file "
            f"{threadpool_file} does not encode one; pass --rank to choose "
            "the matching rank."
        )
    return task_data.file_ranks[0]


def _step_prefix(
    task_file: str | Path, threadpool_file: str | Path | None
) -> str | None:
    """Infer a shared ``stepN`` output prefix from input filenames."""
    steps = [
        match.group(1)
        for path in (task_file, threadpool_file)
        if path is not None
        and (
            match := re.search(r"(?:^|[-_])step(\d+)(?:\D|$)", Path(path).stem)
        )
    ]
    if len(set(steps)) > 1:
        raise ValueError(
            "Task and threadpool files have different step numbers."
        )
    return f"step{steps[0]}" if steps else None


def _lanes(
    thread: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    expand: int,
    origin_ms: float,
    window_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clip intervals to the window and assign lanes.

    Each thread's intervals are spread round-robin over ``expand`` lanes
    in start order, as in SWIFT's plots.

    Returns:
        ``(lane, start, end, index)``: clipped spans relative to the
        window start and the indices of the kept input rows.
    """
    s = np.maximum(start, origin_ms) - origin_ms
    e = np.minimum(end, origin_ms + window_ms) - origin_ms
    index = np.flatnonzero(e > s)
    order = index[np.lexsort((s[index], thread[index]))]
    t = thread[order]
    if expand > 1 and t.size:
        first = np.r_[0, np.flatnonzero(np.diff(t)) + 1]
        within = np.arange(t.size) - np.repeat(
            first, np.diff(np.r_[first, t.size])
        )
        lane = t * expand + within % expand
    else:
        lane = t * expand
    return lane, s[order], e[order], order


def _draw_spans(
    axis: Axes,
    y0: np.ndarray,
    height: float | np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    colours: Sequence[str] | np.ndarray,
    alpha: float = 1.0,
) -> None:
    """Draw rectangles as one PolyCollection (fast for millions)."""
    from matplotlib.collections import PolyCollection

    if start.size == 0:
        return
    y1 = y0 + height
    verts = np.stack(
        [
            np.column_stack([start, y0]),
            np.column_stack([start, y1]),
            np.column_stack([end, y1]),
            np.column_stack([end, y0]),
        ],
        axis=1,
    )
    axis.add_collection(
        PolyCollection(
            verts,  # type: ignore[arg-type]
            facecolors=colours,
            linewidths=0,
            antialiased=False,
            alpha=alpha,
        )
    )


def _plot_timeline(
    tasks: pd.DataFrame,
    pool_rows: pd.DataFrame | None,
    task_thread_count: int,
    pool_thread_count: int,
    start_ms: float,
    window_ms: float,
    expand: int,
    width: float,
    height: float,
    show_legend: bool,
    colours: Mapping[str, str],
    pool_colours: Mapping[str, str],
    title: str,
) -> Figure:
    """Tasks per thread, with the threadpool panel above when available."""
    import matplotlib.pyplot as plt

    if pool_rows is not None:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(width, height),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1, 1.2]},
        )
        workers = pool_rows.loc[pool_rows["thread"] >= 0]
        background = pool_rows.loc[pool_rows["thread"] < 0]
        lanes = max(pool_thread_count * expand, 1)
        _, bs, be, bi = _lanes(
            np.zeros(len(background), dtype=np.int64),
            background["start_ms"].to_numpy(),
            background["end_ms"].to_numpy(),
            1,
            start_ms,
            window_ms,
        )
        functions = background["function"].to_numpy()[bi]
        _draw_spans(
            ax_top,
            np.zeros(bs.size),
            lanes,
            bs,
            be,
            [pool_colours[f] for f in functions],
            alpha=0.15,
        )
        top = _draw_panel(
            ax_top,
            workers["thread"].to_numpy(),
            workers["start_ms"].to_numpy(),
            workers["end_ms"].to_numpy(),
            workers["function"].to_numpy(),
            pool_colours,
            pool_thread_count,
            expand,
            start_ms,
            window_ms,
            "Threadpool thread",
            int(width * DPI),
        )
        ax_top.set_title("Threadpool")
        if show_legend:
            _add_legend(ax_top, top, pool_colours)
    else:
        fig, ax_bottom = plt.subplots(
            figsize=(width, height * 0.6), constrained_layout=True
        )
        ax_top = None

    bottom = _draw_panel(
        ax_bottom,
        tasks["thread"].to_numpy(),
        tasks["start_ms"].to_numpy(),
        tasks["end_ms"].to_numpy(),
        tasks["label"].to_numpy(),
        colours,
        task_thread_count,
        expand,
        start_ms,
        window_ms,
        "Task thread",
        int(width * DPI),
    )
    ax_bottom.set_title("Tasks")
    ax_bottom.set_xlabel("Wall-clock time since step start [ms]")
    if show_legend:
        _add_legend(ax_bottom, bottom, colours)
    fig.suptitle(title)
    return fig


def _draw_panel(
    axis: Axes,
    thread: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    labels: np.ndarray,
    colours: Mapping[str, str],
    thread_count: int,
    expand: int,
    start_ms: float,
    window_ms: float,
    ylabel: str,
    pixels: int,
) -> dict[str, float]:
    """Draw one lane panel; return the time per label inside the window."""
    lane, s, e, index = _lanes(thread, start, end, expand, start_ms, window_ms)
    import pandas as pd
    from matplotlib.colors import to_rgba_array

    inverse, unique = pd.factorize(labels[index])
    palette = to_rgba_array([colours[str(label)] for label in unique])
    total_lanes = max(thread_count * expand, 1)
    axis.imshow(
        _raster_lanes(
            lane, s, e, inverse, palette, total_lanes, window_ms, pixels
        ),
        extent=(0.0, window_ms, 0.0, total_lanes),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_ylim(0.0, total_lanes)
    axis.set_xlim(-0.01 * window_ms, 1.01 * window_ms)
    axis.set_ylabel(ylabel if expand == 1 else f"{ylabel} (x{expand} lanes)")
    axis.yaxis.get_major_locator().set_params(integer=True)  # type: ignore[call-arg]
    axis.set_axisbelow(True)
    axis.grid(True, axis="y", alpha=0.3)
    for x in (0.0, window_ms):
        axis.axvline(x, color="k", linestyle="--", linewidth=1)
    return dict(
        zip(
            (str(u) for u in unique),
            np.bincount(inverse, weights=e - s, minlength=unique.size),
        )
    )


def _raster_lanes(
    lane: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    codes: np.ndarray,
    palette: np.ndarray,
    total_lanes: int,
    window_ms: float,
    pixels: int,
    rows_per_lane: int = 10,
) -> np.ndarray:
    """Paint lane intervals into an RGBA image, one column per pixel.

    Every interval covers at least one column; where intervals share a
    column the later one wins, as when drawing rectangles in order. Each
    lane's first row is left blank as a separator.

    ponytail: raster at the saved resolution (millions of vector
    rectangles cost GBs); interactive zoom (--show) is pixelated.
    """
    image = np.zeros((total_lanes, pixels), dtype=np.int32)  # 0 = empty
    scale = pixels / window_ms
    first = np.clip((start * scale).astype(np.int64), 0, pixels - 1)
    last = np.clip(np.ceil(end * scale).astype(np.int64), first + 1, pixels)
    length = last - first
    which = np.repeat(np.arange(start.size), length)
    offset = np.arange(which.size) - np.repeat(
        np.cumsum(length) - length, length
    )
    image[lane[which], first[which] + offset] = codes[which] + 1
    colours = np.vstack([np.zeros(4), palette]) * 255
    rgba = colours.round().astype(np.uint8)[image]
    rgba = np.repeat(rgba, rows_per_lane, axis=0)
    rgba[::rows_per_lane] = 0
    return rgba


def _add_legend(
    axis: Axes, time_by_label: Mapping[str, float], colours: Mapping[str, str]
) -> None:
    """Legend to the right of a panel, labels by time spent (largest first)."""
    import matplotlib.patches as mpatches

    labels = sorted(time_by_label, key=lambda k: -time_by_label[k])
    if not labels:
        return
    axis.legend(
        handles=[mpatches.Patch(color=colours[k], label=k) for k in labels],
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        ncol=math.ceil(len(labels) / LEGEND_ROWS),
        fontsize="small",
        borderaxespad=0.0,
    )


def _plot_task_activity(
    tasks: pd.DataFrame,
    thread_count: int,
    start_ms: float,
    window_ms: float,
    width: float,
    height: float,
    sort_threads: bool,
    xbins: int = 1000,
) -> Figure:
    """Busy/idle map per thread plus the number of busy threads over time."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if thread_count < 1:
        raise ValueError("Task activity plots require at least one thread.")

    scale = xbins / window_ms
    start_bin = np.clip(
        ((tasks["start_ms"].to_numpy() - start_ms) * scale).astype(np.int64),
        0,
        xbins,
    )
    end_bin = np.clip(
        np.ceil((tasks["end_ms"].to_numpy() - start_ms) * scale).astype(
            np.int64
        ),
        0,
        xbins,
    )
    keep = end_bin > start_bin
    thread = tasks["thread"].to_numpy()[keep]
    # A difference array per thread marks the bins each task covers.
    diff = np.zeros((thread_count, xbins + 1), dtype=np.int64)
    np.add.at(diff, (thread, start_bin[keep]), 1)
    np.add.at(diff, (thread, end_bin[keep]), -1)
    grid = (np.cumsum(diff, axis=1)[:, :xbins] > 0).astype(float)

    if sort_threads:
        grid = grid[np.argsort(grid.sum(axis=1))[::-1], :]

    fig, (ax_hist, ax_grid) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(width, height),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 8]},
    )
    x_edges = np.linspace(0.0, window_ms, xbins + 1)
    idle, busy = "#f0f0f0", "#1f77b4"
    ax_grid.pcolormesh(
        x_edges,
        np.arange(thread_count + 1),
        grid,
        cmap=ListedColormap([idle, busy]),
        vmin=0,
        vmax=1,
        shading="flat",
    )
    ax_grid.legend(
        handles=[
            mpatches.Patch(color=busy, label="running a task"),
            mpatches.Patch(color=idle, label="idle"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0.0,
    )
    ax_grid.set_ylabel(
        "Thread (sorted by busy time)" if sort_threads else "Thread"
    )
    ax_grid.set_xlabel("Wall-clock time since step start [ms]")

    ax_hist.bar(
        x_edges[:-1], grid.sum(axis=0), width=np.diff(x_edges), align="edge"
    )
    ax_hist.set_ylabel("Busy threads")
    ax_hist.set_ylim(0, thread_count)
    ax_hist.grid(True, alpha=0.3)
    return fig


def _task_timing_report(
    tasks: pd.DataFrame, rank: int, names: TaskNames, has_cells: bool
) -> str:
    """Build task (and cell) timing tables for one rank."""
    frame = tasks.assign(
        duration=tasks["end_ms"] - tasks["start_ms"],
        label=tasks["label"].astype(str),
    )
    by_label = frame.groupby("label")["duration"].sum()
    total = float(by_label.sum())
    sections = [
        f"TASK TIMING SUMMARY: RANK {rank}",
        f"Total task worker time: {total:.3f} ms",
        f"Names: {names.source}",
        create_ascii_table(
            ["Task Type", "Time (ms)", "Fraction"],
            _timing_rows(by_label.to_dict(), total),
            "TASK TYPE TIMING",
        ),
    ]
    if not has_cells:
        return "\n\n".join(sections)

    combos = (
        frame.groupby(
            ["label", "ci_type", "cj_type", "ci_subtype", "cj_subtype"],
            observed=True,
        )["duration"]
        .sum()
        .reset_index()
    )
    cell_times: dict[tuple[str, str], float] = {}
    combination_times: dict[str, float] = {}
    task_combination_times: dict[tuple[str, str], float] = {}
    for row in combos.itertuples(index=False):
        pair = _cell_pair_label(row, names, include_subtype=False)
        combination = _cell_pair_label(row, names, include_subtype=True)
        if pair is None or combination is None:
            continue
        key = (str(row.label), pair)
        cell_times[key] = cell_times.get(key, 0.0) + row.duration
        combination_times[combination] = (
            combination_times.get(combination, 0.0) + row.duration
        )
        key2 = (str(row.label), combination)
        task_combination_times[key2] = (
            task_combination_times.get(key2, 0.0) + row.duration
        )
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


def _timing_rows(
    timings: Mapping[str, float] | Mapping[tuple[str, str], float],
    total: float,
) -> list[list[str]]:
    """Format timing mappings as descending ASCII-table rows."""
    rows: list[list[str]] = []
    for key, duration in sorted(
        timings.items(), key=lambda item: item[1], reverse=True
    ):
        labels = list(key) if isinstance(key, tuple) else [key]
        fraction = 100.0 * duration / total if total else 0.0
        rows.append([*labels, f"{duration:.3f}", f"{fraction:.2f}%"])
    return rows


def _cell_pair_label(
    row: object, names: TaskNames, include_subtype: bool
) -> str | None:
    """Order-independent label of a task's cells (None without cells)."""
    cells = []
    for cell_type, subtype in (
        (row.ci_type, row.ci_subtype),  # type: ignore[attr-defined]
        (row.cj_type, row.cj_subtype),  # type: ignore[attr-defined]
    ):
        label = names.cell_name(int(cell_type))
        if label is None:
            continue
        sub = names.cell_name(int(subtype), subtype=True)
        if include_subtype and sub is not None:
            label = f"{label}/{sub}"
        cells.append(label)
    return " + ".join(sorted(cells)) if cells else None


def _task_filter_suffix(
    task_filter: TaskFilter | None, names: TaskNames
) -> str:
    """Filename suffix describing the active filter (names, not indices)."""
    if task_filter is None:
        return ""

    def cell(value: int | str, subtype: bool) -> str:
        index = names.cell_index(value, subtype)
        return (names.cell_name(index, subtype) or str(index)).lower()

    suffixes: list[str] = []
    if task_filter.tasks is not None:
        suffixes.append(
            "+".join(t.replace("/", "-") for t in task_filter.tasks)
        )
    for tag, value, subtype in (
        ("c", task_filter.c_type, False),
        ("ci", task_filter.ci_type, False),
        ("cj", task_filter.cj_type, False),
        ("csub", task_filter.c_subtype, True),
        ("cisub", task_filter.ci_subtype, True),
        ("cjsub", task_filter.cj_subtype, True),
    ):
        if value is not None:
            suffixes.append(f"{tag}{cell(value, subtype)}")
    if task_filter.depth is not None:
        suffixes.append(f"depth{task_filter.depth}")
    return "_" + "_".join(suffixes) if suffixes else ""


def _parse_cell_value(value: str) -> int | str:
    """A cell (sub)type index, or a name resolved against the dump later."""
    try:
        index = int(value)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z_]+", value):
            raise argparse.ArgumentTypeError(
                f"expected a cell (sub)type name or index, got {value!r}"
            ) from None
        return value.lower()
    if index < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative index, got {value!r}"
        )
    return index
