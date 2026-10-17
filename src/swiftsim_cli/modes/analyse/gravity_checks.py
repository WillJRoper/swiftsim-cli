"""Gravity checks analysis module for SWIFT simulations."""

import argparse
import glob
import re
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from swiftsim_cli.utilities import create_output_path


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

    # Check if this is an exact file
    if "exact" in filename:
        # This is an exact file, find the corresponding SWIFT file
        exact_file = filepath

        # Extract step number
        step_match = re.search(r"step(\d{4})", filename)
        if not step_match:
            raise ValueError(f"Could not extract step number from {filename}")
        step_num = step_match.group(1)

        # Look for corresponding SWIFT files with different orders
        swift_pattern = (
            directory / f"gravity_checks_swift_step{step_num}_order*.dat"
        )
        swift_files = list(glob.glob(str(swift_pattern)))

        if not swift_files:
            raise FileNotFoundError(
                f"No SWIFT files found for step {step_num}"
            )

        # Take the first one found
        swift_file = swift_files[0]

    else:
        # This should be a SWIFT file, find the corresponding exact file
        swift_file = filepath

        # Extract step number
        step_match = re.search(r"step(\d{4})", filename)
        if not step_match:
            raise ValueError(f"Could not extract step number from {filename}")
        step_num = step_match.group(1)

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


def analyse_force_checks(
    files: list[str],
    labels: list[str],
    output_path: Union[str, None] = None,
    prefix: Union[str, None] = None,
    show_plot: bool = True,
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
        output_path: Optional path to save the plot. If None, the plot is saved
            to the current directory.
        prefix: Optional prefix to add to the output filename if saving.
            If empty, defaults to 'gravity_checks.png'.
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

    # Set up parameters for plotting
    plt.rcParams.update(
        {
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
    )

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

    # Create the figure with subplots
    fig = plt.figure()
    ax1 = plt.subplot(231)  # X-component error
    ax2 = plt.subplot(232)  # Y-component error
    ax3 = plt.subplot(233)  # Z-component error
    ax4 = plt.subplot(234)  # Magnitude error
    ax5 = plt.subplot(235)  # Potential error

    # Process each file and find its counterpart
    for i, (input_file, label) in enumerate(zip(files, labels)):
        try:
            exact_file, swift_file = _find_counterpart_file(input_file)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error processing {input_file}: {e}")
            continue

        # Read exact data
        exact_data = np.loadtxt(exact_file)
        exact_ids = exact_data[:, 0]
        exact_pos = exact_data[:, 1:4]
        exact_a = exact_data[:, 4:7]
        exact_pot = exact_data[:, 7]

        # Sort exact data by particle ID
        sort_index = np.argsort(exact_ids)
        exact_ids = exact_ids[sort_index]
        exact_pos = exact_pos[sort_index, :]
        exact_a = exact_a[sort_index, :]
        exact_pot = exact_pot[sort_index]
        exact_a_norm = np.sqrt(
            exact_a[:, 0] ** 2 + exact_a[:, 1] ** 2 + exact_a[:, 2] ** 2
        )

        # Read SWIFT data
        swift_data = np.loadtxt(swift_file)
        swift_ids = swift_data[:, 0]
        swift_pos = swift_data[:, 1:4]
        swift_a_grav = swift_data[:, 4:7]
        swift_pot = swift_data[:, 7]

        # Sort SWIFT data by particle ID
        sort_index = np.argsort(swift_ids)
        swift_ids = swift_ids[sort_index]
        swift_pos = swift_pos[sort_index, :]
        swift_a_grav = swift_a_grav[sort_index, :]
        swift_pot = swift_pot[sort_index]

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

        # Bin the errors for histogram
        norm_error_hist, _ = np.histogram(
            norm_error, bins=bin_edges, density=False
        )
        norm_error_hist = norm_error_hist / (np.size(norm_error) * bin_size)

        error_x_hist, _ = np.histogram(error_x, bins=bin_edges, density=False)
        error_x_hist = error_x_hist / (np.size(norm_error) * bin_size)

        error_y_hist, _ = np.histogram(error_y, bins=bin_edges, density=False)
        error_y_hist = error_y_hist / (np.size(norm_error) * bin_size)

        error_z_hist, _ = np.histogram(error_z, bins=bin_edges, density=False)
        error_z_hist = error_z_hist / (np.size(norm_error) * bin_size)

        error_pot_hist, _ = np.histogram(
            error_pot, bins=bin_edges, density=False
        )
        error_pot_hist = error_pot_hist / (np.size(norm_error) * bin_size)

        # Calculate statistics
        norm_median = np.median(norm_error)
        median_x = np.median(error_x)
        median_y = np.median(error_y)
        median_z = np.median(error_z)
        median_pot = np.median(error_pot)

        norm_per99 = np.percentile(norm_error, 99)
        per99_x = np.percentile(error_x, 99)
        per99_y = np.percentile(error_y, 99)
        per99_z = np.percentile(error_z, 99)
        per99_pot = np.percentile(error_pot, 99)

        norm_per90 = np.percentile(norm_error, 90)
        per90_x = np.percentile(error_x, 90)
        per90_y = np.percentile(error_y, 90)
        per90_z = np.percentile(error_z, 90)
        per90_pot = np.percentile(error_pot, 90)

        # Use colors from the color list, cycling if necessary
        color = cols[i % len(cols)]

        # Plot X-component errors
        ax1.semilogx(bins, error_x_hist, color=color, label=label)
        ax1.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_x:.5f} 90%→{per90_x:.5f} 99%→{per99_x:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        # Plot Y-component errors
        ax2.semilogx(bins, error_y_hist, color=color, label=label)
        ax2.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_y:.5f} 90%→{per90_y:.5f} 99%→{per99_y:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        # Plot Z-component errors
        ax3.semilogx(bins, error_z_hist, color=color, label=label)
        ax3.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_z:.5f} 90%→{per90_z:.5f} 99%→{per99_z:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        # Plot magnitude errors
        ax4.semilogx(bins, norm_error_hist, color=color, label=label)
        ax4.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{norm_median:.5f} 90%→{norm_per90:.5f} 99%→{norm_per99:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        # Plot potential errors
        ax5.semilogx(bins, error_pot_hist, color=color, label=label)
        ax5.text(
            min_error * 1.5,
            1.5 - i / 10.0,
            f"50%→{median_pot:.5f} 90%→{per90_pot:.5f} 99%→{per99_pot:.5f}",
            ha="left",
            va="top",
            color=color,
        )

        # Print summary statistics
        print(f"Processed: {label}")
        print(f"  Exact file: {exact_file}")
        print(f"  SWIFT file: {swift_file}")
        print(
            f"  Norm error - median: {norm_median:.5f}, 90%: {norm_per90:.5f},"
            f" 99%: {norm_per99:.5f}",
        )
        print()

    # Set up axes labels and limits
    ax1.set_xlabel("δax/|a_exact|")
    ax1.set_xlim(min_error, max_error)
    ax1.set_ylim(0, 1.75)

    ax2.set_xlabel("δay/|a_exact|")
    ax2.set_xlim(min_error, max_error)
    ax2.set_ylim(0, 1.75)

    ax3.set_xlabel("δaz/|a_exact|")
    ax3.set_xlim(min_error, max_error)
    ax3.set_ylim(0, 1.75)

    ax4.set_xlabel("|δa|/|a_exact|")
    ax4.set_xlim(min_error, max_error)
    ax4.set_ylim(0, 2.5)
    ax4.legend(loc="upper left", fontsize=8)

    ax5.set_xlabel("δφ/φ_exact")
    ax5.set_xlim(min_error, max_error)
    ax5.set_ylim(0, 1.75)

    # Save the figure
    png_file = create_output_path(output_path, prefix, "gravity_checks.png")

    fig.savefig(png_file, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {png_file}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    plt.close(fig)


def add_gravity_checks_arguments(subparsers) -> argparse.ArgumentParser:
    """Add gravity checks analysis arguments to the argument parser.

    Args:
        subparsers: The subparsers object from argparse to add commands to.

    Returns:
        The gravity checks argument parser.
    """
    gravity_parser = subparsers.add_parser(
        "gravity-checks", help="Analyse gravity check files"
    )

    gravity_parser.add_argument(
        "files",
        nargs="+",
        help="List of gravity check files to analyse (either exact "
        "or SWIFT files).",
        type=Path,
    )

    gravity_parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        required=True,
        help="List of labels for the runs being analysed.",
        type=str,
    )

    gravity_parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Where to save analysis (default: current directory).",
        default=None,
    )

    gravity_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="A prefix to add to the analysis files (default: '').",
        default="",
    )

    gravity_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
        default=False,
    )

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
        output_path=args.output_path,
        prefix=args.prefix,
        show_plot=args.show,
        min_error=args.min_error,
        max_error=args.max_error,
        num_bins=args.num_bins,
    )
