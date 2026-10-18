"""Timestep analysis for SWIFT simulations.

This module analyzes timestep evolution files from SWIFT simulations,
providing insights into simulation dynamics and timestep behavior over time.

Key functions:
- analyse_timestep_files: Main analysis and plotting routine
- run_timestep: CLI entry point for timestep analysis
- add_timestep_arguments: CLI argument setup
"""

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from swiftsim_cli.modes.common_args import (
    add_labels_argument,
    add_output_arguments,
)
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.plotting import save_figure
from swiftsim_cli.utilities.step_table import (
    DEAD_TIME_COLUMN,
    WALLCLOCK_COLUMN,
    StepTableReader,
)
from swiftsim_cli.utilities.tables import create_ascii_table

tab10 = matplotlib.colormaps["tab10"]

# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "timesteps_analysis"

_TIMESTEP_LABEL_ALIASES = {
    "Time": "Time [Internal Units]",
    "Scale-factor": "Scale factor",
}


def add_timestep_arguments(subparsers) -> None:
    """Add timestep analysis arguments to the subparser."""
    timestep_parser = subparsers.add_parser(
        "timesteps", help="Plot the timestep files of one or more runs."
    )

    timestep_parser.add_argument(
        "files",
        nargs="+",
        help="Timestep files to analyse.",
        type=Path,
    )
    add_labels_argument(timestep_parser, required=True)

    timestep_parser.add_argument(
        "--time-column",
        help=(
            "Zero-based column index to use for the x-axis. "
            "Use 1 for internal time, 2 for scale factor (default: 2)."
        ),
        type=int,
        default=2,
    )

    add_output_arguments(timestep_parser, OUT_SUBDIR)

    timestep_parser.add_argument(
        "--match-runtimes",
        help="Trim each input to the x-axis extent of the shortest dataset.",
        action="store_true",
        default=False,
    )

    timestep_parser.add_argument(
        "--nthreads",
        nargs="+",
        help=(
            "Number of threads used by each run, in the same order as files. "
            "When given, wall-clock and dead time are converted to CPU time."
        ),
        type=int,
        default=None,
    )


def run_timestep(args: argparse.Namespace) -> None:
    """Run timestep analysis."""
    analyse_timestep_files(
        args.files,
        args.labels,
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        time_column=args.time_column,
        match_runtimes=args.match_runtimes,
        nthreads=args.nthreads,
    )


def _get_x_axis_label(time_column: int) -> str:
    """Return the x-axis label for a selected timestep file column."""
    if time_column == 1:
        return "Time [Internal Units]"
    if time_column == 2:
        return "Scale factor"

    return f"Column {time_column + 1}"


def _normalise_timestep_label(label: str) -> str:
    """Return a user-facing label for a timestep table header."""
    return _TIMESTEP_LABEL_ALIASES.get(label, label)


def analyse_timestep_files(
    files: Sequence[str | Path],
    labels: list[str],
    plot_time: bool | None = None,
    output_path: str | Path | None = None,
    prefix: str | None = None,
    show_plot: bool = False,
    time_column: int | None = None,
    match_runtimes: bool = False,
    nthreads: list[int] | None = None,
) -> None:
    """Plot the timestep files of one or more SWIFT runs.

    Args:
        files: List of file paths to the timestep files.
        labels: List of labels for the runs.
        plot_time: Deprecated compatibility flag. If provided, True selects the
            internal-time column and False selects the scale-factor column.
        output_path: Directory whose ``timesteps_analysis`` subdirectory
            receives the plot. If None, the current directory is used.
        prefix: Optional prefix to add to the output filename.
        show_plot: Whether to display the plot.
        time_column: Zero-based column index to use for the x-axis. Defaults
            to 2, which is the scale-factor column in the timestep table.
        match_runtimes: Whether to trim all plotted datasets to the shortest
            x-axis extent found in the inputs.
        nthreads: Optional number of threads used by each run. If provided,
            cumulative wall-clock and dead time are multiplied by these values
            to show CPU time.

    Raises:
        ValueError: If the number of files and labels do not match, or if a
            file contains no step-table rows.
    """
    # Make sure the number of files and labels match
    if len(files) != len(labels):
        raise ValueError("Number of files and labels must match.")

    if nthreads is not None:
        if len(files) != len(nthreads):
            raise ValueError(
                "Number of files and nthreads entries must match."
            )
        if any(nthread <= 0 for nthread in nthreads):
            raise ValueError("nthreads entries must be positive integers.")

    if time_column is None:
        time_column = 1 if plot_time is True else 2

    if time_column < 0:
        raise ValueError("time_column must be non-negative.")

    place_legend_below = len(labels) > 3
    figure_height = 8.0

    if place_legend_below:
        legend_rows = math.ceil(len(labels) / 4)
        figure_height += 0.2 * legend_rows

    # Loop over the lines in the file and extract the relevant data
    x: list[np.ndarray[Any, Any]] = []
    y: list[np.ndarray[Any, Any]] = []
    deadtime: list[np.ndarray[Any, Any]] = []
    x_label: str | None = None
    runtime_comparison = None
    for file_index, file in enumerate(files):
        xi_values: list[float] = []
        yi_values: list[float] = []
        dti_values: list[float] = []
        step_table = StepTableReader()
        with open(file, "r") as f:
            for line in f:
                row = step_table.feed(line)
                if row is None:
                    continue

                columns = step_table.columns
                if time_column >= len(columns):
                    raise ValueError(
                        f"time_column {time_column} is out of range for "
                        f"'{file}', which has {len(columns)} columns."
                    )
                if WALLCLOCK_COLUMN not in row or DEAD_TIME_COLUMN not in row:
                    raise ValueError(
                        f"'{file}' has no '{WALLCLOCK_COLUMN}' or "
                        f"'{DEAD_TIME_COLUMN}' column."
                    )
                if x_label is None and step_table.header_seen:
                    x_label = _normalise_timestep_label(columns[time_column])

                xi_values.append(row[columns[time_column]])
                yi_values.append(row[WALLCLOCK_COLUMN])
                dti_values.append(row[DEAD_TIME_COLUMN])

        if not xi_values:
            raise ValueError(
                f"No step-table rows could be parsed from '{file}'."
            )

        time_scale = nthreads[file_index] if nthreads is not None else 1

        # Convert to numpy arrays and compute cumulative sums in hours.
        x.append(np.array(xi_values))
        y.append(
            np.cumsum(np.array(yi_values)) * time_scale / (1000 * 60 * 60)
        )
        deadtime.append(
            np.cumsum(np.array(dti_values)) * time_scale / (1000 * 60 * 60)
        )

    max_x_values = [series[-1] for series in x if len(series) > 0]
    # Interpolating/truncating at a common x only makes sense when x grows
    # monotonically (e.g. not for redshift).
    x_is_monotonic = all(np.all(np.diff(series) >= 0) for series in x)
    if max_x_values and not x_is_monotonic:
        print(
            "Warning: x-axis column is not monotonically increasing; "
            "skipping the runtime comparison and --match-runtimes."
        )
    elif max_x_values:
        runtime_limit = min(max_x_values)
        matched_runtimes = [
            (label, float(np.interp(runtime_limit, xi, yi)))
            for label, xi, yi in zip(labels, x, y)
            if len(xi) > 0
        ]
        if len(matched_runtimes) > 1:
            matched_runtimes.sort(key=lambda item: item[1])
            if matched_runtimes[0][1] > 0:
                runtime_comparison = (runtime_limit, matched_runtimes)

        if match_runtimes:
            for i, xi in enumerate(x):
                mask = xi <= runtime_limit
                x[i] = xi[mask]
                y[i] = y[i][mask]
                deadtime[i] = deadtime[i][mask]

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, figure_height),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Colors for the plots
    colors = tab10(np.linspace(0, 1, len(files)))

    # Main plot - absolute times
    for i, (xi, yi, dt, label, color) in enumerate(  # type: ignore[assignment]
        zip(x, y, deadtime, labels, colors)
    ):
        # Plot wall clock time (solid lines)
        ax1.plot(
            xi,
            yi,
            "-",
            color=color,
            linewidth=2,
        )
        # Plot dead time (dashed lines with alpha) - make more visible
        ax1.plot(xi, dt, "--", color=color, alpha=0.6, linewidth=2)

    # Set labels and title for main plot
    if x_label is None:
        x_label = _get_x_axis_label(time_column)
    y_label = "CPU hrs" if nthreads is not None else "Time [hrs]"
    ax1.set_ylabel(y_label)

    if runtime_comparison is not None:
        runtime_limit, matched_runtimes = runtime_comparison
        slowest_runtime = matched_runtimes[-1][1]
        rows = []
        for label, runtime in matched_runtimes:
            speedup = slowest_runtime / runtime
            runtime_reduction = (
                (slowest_runtime - runtime) / slowest_runtime * 100
            )
            rows.append(
                [
                    label,
                    f"{runtime:.3f}",
                    f"{speedup:.2f}x",
                    f"{(speedup - 1) * 100:.1f}",
                    f"{runtime_reduction:.1f}",
                ]
            )
        runtime_header = (
            "CPU time [hrs]" if nthreads is not None else "Runtime [hrs]"
        )
        print(
            "\n"
            + create_ascii_table(
                [
                    "Run",
                    runtime_header,
                    "Speedup vs slowest",
                    "Faster [%]",
                    "Less runtime [%]",
                ],
                rows,
                title=(
                    f"RUNTIME COMPARISON AT SHARED {x_label} {runtime_limit:g}"
                ),
            )
        )

    # Create custom legend with black lines showing line styles
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=2,
            label="Wallclock Time",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label="Dead Time",
        ),
    ]
    ax1.legend(handles=legend_elements, loc="best")

    # Deadtime percentage plot
    for i, (xi, yi, dt, label, color) in enumerate(  # type: ignore[assignment]
        zip(x, y, deadtime, labels, colors)
    ):
        # Calculate deadtime percentage: (deadtime / total_time) * 100
        # Protect against division by zero
        deadtime_percentage = np.where(yi != 0, (dt / yi) * 100, 0.0)

        # Plot deadtime percentage
        ax2.plot(
            xi,
            deadtime_percentage,
            "-",
            color=color,
            label=f"{label}",
            linewidth=2,
        )

    # Set labels and formatting for deadtime percentage plot
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Dead Time [%]")

    if place_legend_below:
        ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=min(4, len(labels)),
        )
    else:
        ax2.legend(loc="best")

    ax2.set_ylim(0, None)  # Start y-axis at 0 for percentage

    # Adjust layout to prevent overlapping
    if place_legend_below:
        plt.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        plt.tight_layout()

    output_file = create_output_path(
        output_path, prefix, "timestep_analysis.png", OUT_SUBDIR
    )
    save_figure(fig, output_file, show=show_plot)
    print(f"Plot saved to {output_file}")
