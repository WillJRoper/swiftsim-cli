"""Gravity checks analysis module for SWIFT simulations."""

import argparse
import glob
import re
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from swiftsim_cli.modes.common_args import (
    add_labels_argument,
    add_output_arguments,
)
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.plotting import save_figure

# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "gravity_checks_analysis"


def _find_counterpart_file(filepath: str) -> tuple[str, str]:
    """Find the counterpart file (exact <-> swift) for a given file.

    Args:
        filepath: Path to either an exact or SWIFT gravity check file.

    Returns:
        tuple: (exact_file_path, swift_file_path)

    Raises:
        ValueError: If step number cannot be extracted from filename.
        FileNotFoundError: If counterpart file cannot be found.
    """
    file_path = Path(filepath)
    filename = file_path.name
    directory = file_path.parent

    # Initialize variables
    exact_file: Optional[str]
    swift_file: str

    # Extract the step number, reusing SWIFT's own zero padding (%.4d, so
    # steps >= 10000 have more digits)
    step_match = re.search(r"step(\d+)", filename)
    if not step_match:
        raise ValueError(f"Could not extract step number from {filename}")
    step_num = step_match.group(1)

    # Check if this is an exact file
    if "exact" in filename:
        # This is an exact file, find the corresponding SWIFT file
        exact_file = filepath

        # Look for corresponding SWIFT files with different orders
        swift_pattern = (
            directory / f"gravity_checks_swift_step{step_num}_order*.dat"
        )
        swift_files = sorted(glob.glob(str(swift_pattern)))

        if not swift_files:
            raise FileNotFoundError(
                f"No SWIFT files found for step {step_num}"
            )

        # Pick deterministically if several multipole orders exist
        swift_file = swift_files[0]
        if len(swift_files) > 1:
            print(
                f"Warning: {len(swift_files)} SWIFT files found for step "
                f"{step_num}; using {swift_file}. Pass the SWIFT file "
                "directly to choose another multipole order."
            )

    else:
        # This should be a SWIFT file, find the corresponding exact file
        swift_file = filepath

        # Look for exact file - try different patterns
        exact_patterns = [
            directory / f"gravity_checks_exact_step{step_num}.dat",
            directory / f"gravity_checks_exact_periodic_step{step_num}.dat",
        ]

        exact_file = None
        for pattern in exact_patterns:
            if pattern.exists():
                exact_file = str(pattern)
                break

        if exact_file is None:
            raise FileNotFoundError(f"No exact file found for step {step_num}")

    # At this point, exact_file is guaranteed to be a string
    assert exact_file is not None
    return exact_file, swift_file


def _load_force_pair(
    exact_file: str, swift_file: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an exact/SWIFT gravity check pair aligned by particle ID.

    Args:
        exact_file: Path to the exact force file.
        swift_file: Path to the SWIFT force file.

    Returns:
        tuple: (positions, exact_a, swift_a, exact_pot, swift_pot) for the
            particles present in both files, sorted by particle ID.
            Positions are taken from the exact file.

    Raises:
        ValueError: If the files share no particle IDs.
    """
    exact_data = np.loadtxt(exact_file, ndmin=2)
    swift_data = np.loadtxt(swift_file, ndmin=2)

    common_ids, exact_index, swift_index = np.intersect1d(
        exact_data[:, 0], swift_data[:, 0], return_indices=True
    )
    if common_ids.size == 0:
        raise ValueError(
            f"No common particle IDs between {exact_file} and {swift_file}"
        )
    if common_ids.size != len(exact_data) or common_ids.size != len(
        swift_data
    ):
        print(
            f"Warning: particle IDs differ between {exact_file} "
            f"({len(exact_data)} particles) and {swift_file} "
            f"({len(swift_data)} particles); using the "
            f"{common_ids.size} common particles."
        )

    exact_data = exact_data[exact_index]
    swift_data = swift_data[swift_index]
    return (
        exact_data[:, 1:4],
        exact_data[:, 4:7],
        swift_data[:, 4:7],
        exact_data[:, 7],
        swift_data[:, 7],
    )


def analyse_force_checks(
    files: list[str],
    labels: list[str],
    output_path: Union[str, Path, None] = None,
    prefix: Union[str, None] = None,
    show_plot: bool = False,
    min_error: float = 1e-7,
    max_error: float = 3e-1,
    num_bins: int = 64,
) -> None:
    """Plot the force check analysis for one or more SWIFT runs.

    This function compares SWIFT gravity calculations with exact solutions
    by loading counterpart files and computing relative errors in force
    components and gravitational potential. It generates histograms showing
    the distribution of errors and statistical summaries.

    Args:
        files: List of file paths to either exact or SWIFT force files.
               The function will automatically find the counterpart files.
        labels: List of labels for the runs. Must match the length of files.
        output_path: Directory whose ``gravity_checks_analysis``
            subdirectory receives the plot. If None, the current directory
            is used.
        prefix: Optional prefix to add to the output filename.
        show_plot: Whether to display the plot interactively.
        min_error: Minimum error for binning (default: 1e-7).
        max_error: Maximum error for binning (default: 3e-1).
        num_bins: Number of bins for histogram (default: 64).

    Raises:
        ValueError: If the number of files and labels do not match.
        FileNotFoundError: If counterpart files cannot be found.

    Note:
        The function expects gravity check files in the format produced by
        SWIFT with names containing step numbers and 'exact' or 'swift'
        identifiers. Files should contain columns for:
        [ID, x, y, z, ax, ay, az, potential]
    """
    # Make sure the number of files and labels match
    if len(files) != len(labels):
        raise ValueError("Number of files and labels must match.")

    # Plot styling, scoped so it does not leak into later modes
    plot_style = {
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "font.size": 11,
        "legend.fontsize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "text.usetex": False,
        "figure.figsize": (12, 9),
        "figure.subplot.left": 0.06,
        "figure.subplot.right": 0.99,
        "figure.subplot.bottom": 0.06,
        "figure.subplot.top": 0.99,
        "figure.subplot.wspace": 0.14,
        "figure.subplot.hspace": 0.14,
        "lines.markersize": 6,
        "lines.linewidth": 3.0,
    }

    with plt.rc_context(plot_style):
        # Construct the bins for error histogram
        bin_edges = np.linspace(
            np.log10(min_error), np.log10(max_error), num_bins + 1
        )
        bin_size = (np.log10(max_error) - np.log10(min_error)) / num_bins
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_edges = 10**bin_edges
        bins = 10**bins

        # Colors for the plots
        cols = ["#332288", "#88CCEE", "#117733", "#DDCC77", "#CC6677"]

        # Create the figure with one panel per error component:
        # (axes, x label, y limit)
        fig = plt.figure()
        panels = (
            (plt.subplot(231), "δax/|a_exact|", 1.75),
            (plt.subplot(232), "δay/|a_exact|", 1.75),
            (plt.subplot(233), "δaz/|a_exact|", 1.75),
            (plt.subplot(234), "|δa|/|a_exact|", 2.5),
            (plt.subplot(235), "δφ/φ_exact", 1.75),
        )

        # Process each file and find its counterpart
        for i, (input_file, label) in enumerate(zip(files, labels)):
            try:
                exact_file, swift_file = _find_counterpart_file(
                    str(input_file)
                )
                _, exact_a, swift_a_grav, exact_pot, swift_pot = (
                    _load_force_pair(exact_file, swift_file)
                )
            except (ValueError, FileNotFoundError) as e:
                print(f"Error processing {input_file}: {e}")
                continue

            exact_a_norm = np.sqrt(
                exact_a[:, 0] ** 2 + exact_a[:, 1] ** 2 + exact_a[:, 2] ** 2
            )

            # Compute force differences
            diff = exact_a - swift_a_grav
            diff_pot = exact_pot - swift_pot

            # Correct for different normalization of potential
            exact_pot_corrected = exact_pot - np.mean(diff_pot)
            diff_pot = exact_pot_corrected - swift_pot

            # Calculate relative errors
            norm_diff = np.sqrt(
                diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2
            )
            norm_error = norm_diff / exact_a_norm
            error_x = np.abs(diff[:, 0]) / exact_a_norm
            error_y = np.abs(diff[:, 1]) / exact_a_norm
            error_z = np.abs(diff[:, 2]) / exact_a_norm
            error_pot = np.abs(diff_pot) / np.abs(exact_pot_corrected)

            # Use colors from the color list, cycling if necessary
            color = cols[i % len(cols)]

            # Histogram each error component and annotate its percentiles
            normalisation = np.size(norm_error) * bin_size
            panel_errors = (error_x, error_y, error_z, norm_error, error_pot)
            for (ax, _, _), errors in zip(panels, panel_errors):
                error_hist, _ = np.histogram(
                    errors, bins=bin_edges, density=False
                )
                median = np.median(errors)
                per90, per99 = np.percentile(errors, [90, 99])
                ax.semilogx(
                    bins, error_hist / normalisation, color=color, label=label
                )
                ax.text(
                    min_error * 1.5,
                    1.5 - i / 10.0,
                    f"50%→{median:.5f} 90%→{per90:.5f} 99%→{per99:.5f}",
                    ha="left",
                    va="top",
                    color=color,
                )

            norm_median = np.median(norm_error)
            norm_per90, norm_per99 = np.percentile(norm_error, [90, 99])

            # Print summary statistics
            print(f"Processed: {label}")
            print(f"  Exact file: {exact_file}")
            print(f"  SWIFT file: {swift_file}")
            print(
                f"  Norm error - median: {norm_median:.5f}, "
                f"90%: {norm_per90:.5f},"
                f" 99%: {norm_per99:.5f}",
            )
            print()

        # Set up axes labels and limits
        for ax, xlabel, ymax in panels:
            ax.set_xlabel(xlabel)
            ax.set_xlim(min_error, max_error)
            ax.set_ylim(0, ymax)
        panels[3][0].legend(loc="upper left", fontsize=8)

        # Save the figure
        png_file = create_output_path(
            output_path, prefix, "gravity_checks.png", OUT_SUBDIR
        )
        save_figure(fig, png_file, show=show_plot, dpi=200)
        print(f"Plot saved to {png_file}")


def add_gravity_checks_arguments(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Add gravity checks analysis arguments to the argument parser.

    Args:
        subparsers: The subparsers object from argparse to add commands to.

    Returns:
        The gravity checks argument parser.
    """
    gravity_parser = subparsers.add_parser(
        "gravity-checks",
        help="Compare SWIFT gravity forces with exact gravity checks.",
    )

    gravity_parser.add_argument(
        "files",
        nargs="+",
        help="Gravity check files to analyse (either exact or SWIFT files).",
        type=Path,
    )
    add_labels_argument(gravity_parser, required=True)
    add_output_arguments(gravity_parser, OUT_SUBDIR)

    gravity_parser.add_argument(
        "--min-error",
        type=float,
        help="Minimum error for binning (default: 1e-7).",
        default=1e-7,
    )

    gravity_parser.add_argument(
        "--max-error",
        type=float,
        help="Maximum error for binning (default: 3e-1).",
        default=3e-1,
    )

    gravity_parser.add_argument(
        "--num-bins",
        type=int,
        help="Number of bins for histogram (default: 64).",
        default=64,
    )

    return gravity_parser


def run_gravity_checks(args: argparse.Namespace) -> None:
    """Execute the gravity checks analysis.

    Args:
        args: Parsed command line arguments containing the analysis parameters.
    """
    analyse_force_checks(
        files=args.files,
        labels=args.labels,
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        min_error=args.min_error,
        max_error=args.max_error,
        num_bins=args.num_bins,
    )
