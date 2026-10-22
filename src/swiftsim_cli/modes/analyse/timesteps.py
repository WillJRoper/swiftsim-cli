"""Timestep analysis for SWIFT simulations.

This module analyses timestep evolution files from SWIFT simulations,
providing insights into simulation dynamics and timestep behavior over time.

Key functions:
- analyse_timestep_files: Main analysis and plotting routine
- run_timestep: CLI entry point for timestep analysis
- add_timestep_arguments: CLI argument setup
"""

import argparse
import math
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

from swiftsim_cli.modes.common_args import (
    add_labels_argument,
    add_output_arguments,
    add_steps_argument,
    existing_file,
    positive_int,
    resolve_labels,
)
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.messages import note, plural, warn
from swiftsim_cli.utilities.plotting import save_figure
from swiftsim_cli.utilities.step_table import (
    DEAD_TIME_COLUMN,
    WALLCLOCK_COLUMN,
    StepRow,
    StepTableReader,
)
from swiftsim_cli.utilities.tables import create_ascii_table

# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "timesteps_analysis"

# --time-column choices and the step-table columns they select
TIME_COLUMNS = {
    "step": "Step",
    "time": "Time",
    "scale-factor": "Scale-factor",
    "redshift": "Redshift",
}
# "auto": the scale factor if it varies (cosmological runs), else time
AUTO_TIME_COLUMN = "auto"

_AXIS_LABELS = {
    "Step": "Step",
    "Time": "Time [internal units]",
    "Scale-factor": "Scale factor",
    "Redshift": "Redshift",
}

_MS_PER_HOUR = 1000 * 60 * 60


class _Run(NamedTuple):
    """One run's step series, cumulative times in hours."""

    label: str
    step: np.ndarray
    x: np.ndarray
    wall: np.ndarray
    dead: np.ndarray | None


def add_timestep_arguments(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add timestep analysis arguments to the subparser."""
    timestep_parser = subparsers.add_parser(
        "timesteps",
        help="Plot the timestep files of one or more runs.",
        description="Plot the cumulative runtime and dead time of one or "
        "more runs from their timesteps files (or SWIFT logs, which "
        "contain the same step table). Steps repeated after a restart "
        "replace the earlier rows.",
    )
    timestep_parser.set_defaults(func=run_timestep)

    timestep_parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="Timestep files to analyse.",
        type=existing_file,
    )
    add_labels_argument(timestep_parser)

    timestep_parser.add_argument(
        "--time-column",
        choices=(AUTO_TIME_COLUMN, *TIME_COLUMNS),
        default=AUTO_TIME_COLUMN,
        help="Step-table column to use for the x-axis (default: auto, "
        "the scale factor if it varies, otherwise time).",
    )
    add_steps_argument(timestep_parser)

    add_output_arguments(timestep_parser, OUT_SUBDIR)

    timestep_parser.add_argument(
        "--match-runtimes",
        help="Trim each input to the x-axis extent of the shortest dataset.",
        action="store_true",
        default=False,
    )

    timestep_parser.add_argument(
        "--cores-per-run",
        dest="threads_per_run",
        nargs="+",
        metavar="N",
        help=(
            "Cores used by each run (MPI ranks x threads per rank), in the "
            "same order as the files. When given, wall-clock and dead time "
            "are multiplied by these to give core-hours."
        ),
        type=positive_int,
        default=None,
    )


def run_timestep(args: argparse.Namespace) -> None:
    """Run timestep analysis."""
    analyse_timestep_files(
        args.files,
        resolve_labels(args.files, args.labels),
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        time_column=args.time_column,
        match_runtimes=args.match_runtimes,
        nthreads=args.threads_per_run,
        step_range=args.steps,
    )


def _looks_like_row(line: str) -> bool:
    """Whether an unparsed line starts like a step-table data row."""
    tokens = line.split()
    if tokens and tokens[0].startswith("["):  # MPI rank prefix
        tokens = tokens[1:]
    return bool(tokens) and tokens[0].isdigit()


def _read_step_rows(file: str | Path) -> list[StepRow]:
    """Read a file's step-table rows, resolving restarts.

    A restarted run appends to its timesteps file, so steps rewind. A row
    whose step is not above the previous one supersedes every earlier row
    from that step on, leaving the latest occurrence of each step.

    Args:
        file: The timesteps file or SWIFT log.

    Returns:
        The rows in step order.
    """
    reader = StepTableReader()
    rows: list[StepRow] = []
    superseded = malformed = 0
    with open(file, "r") as f:
        for line in f:
            row = reader.feed(line)
            if row is None:
                if _looks_like_row(line):
                    malformed += 1
                continue
            while rows and rows[-1]["Step"] >= row["Step"]:
                rows.pop()
                superseded += 1
            rows.append(row)
    if malformed:
        warn(
            f"{file}: skipped {plural(malformed, 'malformed step-table row')} "
            "(wrong number of columns or non-numeric values)."
        )
    if superseded:
        warn(
            f"{file}: {plural(superseded, 'row')} superseded by a restart "
            "(repeated steps); using the latest occurrence of each step."
        )
    return rows


def _choose_x_column(time_column: str, tables: list[list[StepRow]]) -> str:
    """Return the step-table column to plot on the x-axis.

    Args:
        time_column: An `AUTO_TIME_COLUMN` or `TIME_COLUMNS` key.
        tables: The rows of each input.

    Returns:
        The column name.
    """
    if time_column != AUTO_TIME_COLUMN:
        return TIME_COLUMNS[time_column]
    if all(len({row["Scale-factor"] for row in rows}) > 1 for rows in tables):
        return "Scale-factor"
    note("the scale factor is constant (non-cosmological run); using time.")
    return "Time"


def _build_run(
    file: str | Path,
    label: str,
    rows: list[StepRow],
    column: str,
    time_scale: int,
) -> _Run:
    """Turn a file's rows into cumulative series.

    Args:
        file: The input (for messages).
        label: The run label.
        rows: The step rows.
        column: The x-axis column.
        time_scale: Cores per run (1 for wall-clock hours).

    Returns:
        The run's series.

    Raises:
        ValueError: If a required column is missing.
    """
    for name in (column, WALLCLOCK_COLUMN):
        if any(name not in row for row in rows):
            raise ValueError(f"'{file}' has no '{name}' column.")

    def column_values(name: str) -> np.ndarray:
        return np.array([row[name] for row in rows])

    step = column_values("Step").astype(np.int64)
    wall = np.cumsum(column_values(WALLCLOCK_COLUMN)) * time_scale
    dead = None
    if all(DEAD_TIME_COLUMN in row for row in rows):
        dead = np.cumsum(column_values(DEAD_TIME_COLUMN)) * time_scale
        dead /= _MS_PER_HOUR
    else:
        warn(f"{file}: no '{DEAD_TIME_COLUMN}' column; dead time not shown.")

    gaps = np.flatnonzero(np.diff(step) > 1)
    if gaps.size:
        missing = int(np.sum(np.diff(step)[gaps] - 1))
        warn(
            f"{file}: {plural(missing, 'step')} missing in "
            f"{plural(gaps.size, 'gap')} (first {step[gaps[0]]} -> "
            f"{step[gaps[0] + 1]}); the lines are broken there."
        )
    return _Run(label, step, column_values(column), wall / _MS_PER_HOUR, dead)


def _break_at_gaps(step: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Insert NaN where steps are missing so the plotted line breaks."""
    gaps = np.flatnonzero(np.diff(step) > 1) + 1
    return np.insert(values.astype(float), gaps, np.nan)


def _compare_runtimes(
    runs: list[_Run], files: Sequence[str | Path], match_runtimes: bool
) -> tuple[float, list[tuple[str, float]]] | None:
    """Compare runtimes at the shortest shared x-axis extent.

    Args:
        runs: The runs (trimmed in place of the list when
            ``match_runtimes``).
        files: The inputs, for messages.
        match_runtimes: Whether to trim every run to the shared extent.

    Returns:
        ``(x_limit, [(label, runtime), ...])`` sorted fastest first, or
        None if the runs cannot be compared.
    """
    for run, file in zip(runs, files):
        if np.ptp(run.x) == 0:
            warn(
                f"x-axis column is constant in {file}; skipping the runtime "
                "comparison and --match-runtimes (choose another "
                "--time-column)."
            )
            return None
    # Interpolating/truncating at a common x only makes sense when x grows
    # monotonically (e.g. not for redshift).
    if not all(np.all(np.diff(run.x) >= 0) for run in runs):
        warn(
            "x-axis column is not monotonically increasing; skipping the "
            "runtime comparison and --match-runtimes."
        )
        return None

    x_limit = float(min(run.x[-1] for run in runs))
    if match_runtimes:
        for i, run in enumerate(runs):
            mask = run.x <= x_limit
            runs[i] = _Run(
                run.label,
                run.step[mask],
                run.x[mask],
                run.wall[mask],
                None if run.dead is None else run.dead[mask],
            )
    if len(runs) < 2:
        return None
    runtimes = sorted(
        (
            (run.label, float(np.interp(x_limit, run.x, run.wall)))
            for run in runs
        ),
        key=lambda item: item[1],
    )
    if runtimes[0][1] <= 0:
        return None
    return x_limit, runtimes


def _print_comparison(
    x_label: str,
    comparison: tuple[float, list[tuple[str, float]]],
    cores: bool,
) -> None:
    """Print the runtime comparison table."""
    x_limit, runtimes = comparison
    slowest = runtimes[-1][1]
    rows = [
        [
            label,
            f"{runtime:.3f}",
            f"{slowest / runtime:.2f}x",
            f"{(slowest - runtime) / slowest * 100:.1f}",
        ]
        for label, runtime in runtimes
    ]
    header = "Core time [core-hrs]" if cores else "Runtime [hrs]"
    print(
        "\n"
        + create_ascii_table(
            ["Run", header, "Speedup vs slowest", "Less runtime [%]"],
            rows,
            title=f"RUNTIME COMPARISON AT SHARED {x_label} {x_limit:g}",
        )
    )


def _plot_runs(
    runs: list[_Run], x_label: str, cores: bool
) -> tuple[Any, tuple[Any, Any]]:
    """Plot cumulative runtime and dead-time fraction for each run.

    Returns:
        The figure and its two axes.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    place_legend_below = len(runs) > 3
    figure_height = 8.0
    if place_legend_below:
        figure_height += 0.2 * math.ceil(len(runs) / 4)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, figure_height),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    cmap = matplotlib.colormaps["tab10" if len(runs) <= 10 else "tab20"]
    for i, run in enumerate(runs):
        color = cmap(i % cmap.N)
        x = _break_at_gaps(run.step, run.x)
        ax1.plot(
            x,
            _break_at_gaps(run.step, run.wall),
            "-",
            color=color,
            linewidth=2,
        )
        if run.dead is None:
            ax2.plot([], [], "-", color=color, label=run.label)
            continue
        ax1.plot(
            x,
            _break_at_gaps(run.step, run.dead),
            "--",
            color=color,
            alpha=0.6,
            linewidth=2,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(run.wall != 0, run.dead / run.wall * 100, 0.0)
        ax2.plot(
            x,
            _break_at_gaps(run.step, fraction),
            "-",
            color=color,
            label=run.label,
            linewidth=2,
        )

    ax1.set_ylabel(
        "Cumulative core time [core-hrs]"
        if cores
        else "Cumulative wall-clock time [hrs]"
    )
    ax1.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=2),
            Line2D(
                [0], [0], color="black", linestyle="--", linewidth=2, alpha=0.6
            ),
        ],
        labels=["Wall-clock time", "Dead time"],
        loc="best",
    )

    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Cumulative dead /\nwall-clock time [%]")
    ax2.set_ylim(0, None)
    if place_legend_below:
        ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=min(4, len(runs)),
        )
        plt.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        ax2.legend(loc="best")
        plt.tight_layout()
    return fig, (ax1, ax2)


def analyse_timestep_files(
    files: Sequence[str | Path],
    labels: list[str],
    output_path: str | Path | None = None,
    prefix: str | None = None,
    show_plot: bool = False,
    time_column: str = AUTO_TIME_COLUMN,
    match_runtimes: bool = False,
    nthreads: list[int] | None = None,
    step_range: Sequence[int] | None = None,
) -> None:
    """Plot the timestep files of one or more SWIFT runs.

    Args:
        files: List of file paths to the timestep files.
        labels: List of labels for the runs.
        output_path: Directory whose ``timesteps_analysis`` subdirectory
            receives the plot. If None, the current directory is used.
        prefix: Optional prefix to add to the output filename.
        show_plot: Whether to display the plot.
        time_column: The x-axis column: ``"auto"`` (the scale factor if it
            varies in every input, otherwise time) or a `TIME_COLUMNS` key.
        match_runtimes: Whether to trim all plotted datasets to the shortest
            x-axis extent found in the inputs.
        nthreads: Optional number of cores (MPI ranks x threads per rank)
            used by each run. If provided, cumulative wall-clock and dead
            time are multiplied by these values to give core-hours.
        step_range: Optional inclusive ``(start, end)`` step range.

    Raises:
        ValueError: If the number of files and labels do not match, or if a
            file contains no step-table rows.
    """
    if len(files) != len(labels):
        raise ValueError("Number of files and labels must match.")

    if nthreads is not None:
        if len(files) != len(nthreads):
            raise ValueError(
                "Number of files and cores-per-run entries must match."
            )
        if any(nthread <= 0 for nthread in nthreads):
            raise ValueError(
                "cores-per-run entries must be positive integers."
            )

    if time_column != AUTO_TIME_COLUMN and time_column not in TIME_COLUMNS:
        raise ValueError(
            f"time_column must be one of {AUTO_TIME_COLUMN}, "
            f"{', '.join(TIME_COLUMNS)}, got {time_column!r}."
        )

    tables = []
    for file in files:
        rows = _read_step_rows(file)
        if step_range is not None:
            rows = [
                row
                for row in rows
                if step_range[0] <= row["Step"] <= step_range[1]
            ]
        if not rows:
            where = (
                f" in steps {step_range[0]}-{step_range[1]}"
                if step_range is not None
                else ""
            )
            raise ValueError(
                f"No step-table rows could be parsed from '{file}'{where}."
            )
        tables.append(rows)

    column = _choose_x_column(time_column, tables)
    x_label = _AXIS_LABELS[column]
    runs = [
        _build_run(
            file,
            label,
            rows,
            column,
            nthreads[i] if nthreads is not None else 1,
        )
        for i, (file, label, rows) in enumerate(zip(files, labels, tables))
    ]

    comparison = _compare_runtimes(runs, files, match_runtimes)
    if comparison is not None:
        _print_comparison(x_label, comparison, cores=nthreads is not None)

    fig, _ = _plot_runs(runs, x_label, cores=nthreads is not None)
    output_file = create_output_path(
        output_path, prefix, "timesteps_analysis.png", OUT_SUBDIR
    )
    save_figure(fig, output_file, show=show_plot)
    print(f"Plot saved to {output_file}")
