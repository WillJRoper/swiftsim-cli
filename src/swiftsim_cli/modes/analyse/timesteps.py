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

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Ensure matplotlib has the tab10 colormap
try:
    tab10 = cm.get_cmap("tab10")
except (AttributeError, ValueError):
    # Fallback if tab10 is not available
    tab10 = cm.get_cmap("Set1")

from swiftsim_cli.utilities import create_output_path


def add_timestep_arguments(subparsers) -> None:
    """Add timestep analysis arguments to the subparser."""
    timestep_parser = subparsers.add_parser(
        "timesteps", help="Analyse timestep files"
    )

    timestep_parser.add_argument(
        "files",
        nargs="+",
        help="List of timestep files to analyse and produce a plot for.",
        type=str,
    )

    timestep_parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        help="Labels for the files (same order as files).",
        type=str,
        required=True,
    )

    timestep_parser.add_argument(
        "--time-column",
        help=(
            "Zero-based column index to use for the x-axis. "
            "Use 1 for scale factor, 2 for internal time."
        ),
        type=int,
        default=2,
    )

    timestep_parser.add_argument(
        "--output-path",
        "-o",
        help="Output path for the plot.",
        type=str,
        default=None,
    )

    timestep_parser.add_argument(
        "--prefix",
        "-p",
        help="Prefix for the output files.",
        type=str,
        default=None,
    )

    timestep_parser.add_argument(
        "--show-plot",
        help="Show the plot after saving.",
        action="store_true",
        default=False,
    )


def run_timestep(args: argparse.Namespace) -> None:
    """Run timestep analysis."""
    analyse_timestep_files(
        args.files,
        args.labels,
        output_path=args.output_path,
        prefix=args.prefix,
        show_plot=args.show_plot,
        time_column=args.time_column,
    )


def _get_x_axis_label(time_column: int) -> str:
    """Return the x-axis label for a selected timestep file column."""
    if time_column == 1:
        return "Scale factor"
    if time_column == 2:
        return "Time [Internal Units]"

    return f"Column {time_column + 1}"


def analyse_timestep_files(
    files: list[str],
    labels: list[str],
    plot_time: bool | None = None,
    output_path: str | None = None,
    prefix: str | None = None,
    show_plot: bool = True,
    time_column: int | None = None,
) -> None:
    """Plot the timestep files of one or more SWIFT runs.

    Args:
        files: List of file paths to the timestep files.
        labels: List of labels for the runs.
        plot_time: Deprecated compatibility flag. If provided, True selects the
            internal-time column and False selects the scale-factor column.
        output_path: Optional path to save the plot. If None, the plot is saved
            to the current directory.
        prefix: Optional prefix to add to the output filename. If None,
            the default filename is determined by create_output_path.
        show_plot: Whether to display the plot.
        time_column: Zero-based column index to use for the x-axis. Defaults
            to 2, which is the third column in the timestep table.

    Raises:
        ValueError: If the number of files and labels do not match.
    """
    # Make sure the number of files and labels match
    if len(files) != len(labels):
        raise ValueError("Number of files and labels must match.")

    if time_column is None:
        time_column = 2 if plot_time in (None, True) else 1

    if time_column < 0:
        raise ValueError("time_column must be non-negative.")

    # Column indices in the timestep table
    x_index = time_column
    wall_clock_index = 12
    deadtime_index = -1
    place_legend_below = len(labels) > 3
    figure_height = 8.0

    if place_legend_below:
        legend_rows = math.ceil(len(labels) / 4)
        figure_height += 0.2 * legend_rows

    # Loop over the lines in the file and extract the relevant data
    x = []
    y = []
    deadtime = []
    for file in files:
        xi, yi, dti = [], [], []
        with open(file, "r") as f:
            for line in f:
                # Ensure we aren't reading a comment line
                if line.startswith("#"):
                    continue

                # If the line doesn't start with an empty space, its not a
                # data line
                if not line[0].isspace():
                    continue

                # Split the line into parts
                parts = line.split()

                # Ensure we found 15 columns
                if len(parts) != 15:
                    continue

                # Ensure all 15 columns are numbers
                try:
                    [float(part) for part in parts]
                except ValueError:
                    print(
                        "Failed to parse line:",
                        line,
                        "in file:",
                        file,
                        "(this is probably fine)",
                    )
                    continue

                xi.append(float(parts[x_index]))
                yi.append(float(parts[wall_clock_index]))
                dti.append(float(parts[deadtime_index]))

        # Convert to numpy arrays and compute cumulative sums in hours
        x.append(np.array(xi))
        y.append(np.cumsum(np.array(yi)) / (1000 * 60 * 60))
        deadtime.append(np.cumsum(np.array(dti)) / (1000 * 60 * 60))

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
    x_label = _get_x_axis_label(time_column)
    ax1.set_ylabel("Time [hrs]")

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

    # Create the output path
    output_file = create_output_path(
        output_path, prefix, "timestep_analysis.png"
    )

    # Save the figure if an output path is provided
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    plt.close()
