"""Gravity error maps analysis module for SWIFT simulations.

This module provides functionality to create hexbin error maps for gravity
check files, comparing exact solutions with SWIFT simulation results. It
supports creating both continuous error visualizations and binary
threshold-based error maps.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.plotting import save_figure

from .gravity_checks import _find_counterpart_file, _load_force_pair


def add_gravity_error_maps_arguments(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add CLI arguments for gravity error maps analysis.

    Args:
        subparsers: The subparsers object to add the gravity-error-maps parser.
    """
    error_map_parser = subparsers.add_parser(
        "gravity-error-maps",
        help="Create hexbin error maps for gravity check files",
    )

    error_map_parser.add_argument(
        "files",
        nargs="+",
        help="List of gravity check files to analyse (exact or SWIFT files).",
        type=Path,
    )

    error_map_parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        required=True,
        help="List of labels for the runs being analysed.",
        type=str,
    )

    error_map_parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Where to save analysis (default: current directory).",
        default=None,
    )

    error_map_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="A prefix to add to the analysis files (default: '').",
        default="",
    )

    error_map_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
        default=False,
    )

    error_map_parser.add_argument(
        "--resolution",
        "--gridsize",
        "-r",
        type=int,
        help="Resolution (gridsize) for hexbin plot (default: 100).",
        default=100,
    )

    error_map_parser.add_argument(
        "--thresh",
        type=float,
        help="Error threshold for hexbin color scale (default: 1e-2).",
        default=1e-2,
    )


def run_gravity_error_maps(args: argparse.Namespace) -> None:
    """Execute the gravity error map analysis.

    Args:
        args: Parsed command line arguments containing analysis parameters.
    """
    analyse_gravity_error_maps(
        files=args.files,
        labels=args.labels,
        output_path=args.output_path,
        prefix=args.prefix,
        show_plot=args.show,
        resolution=args.resolution,
        error_thresh=args.thresh,
    )


def analyse_gravity_error_maps(
    files: List[Union[str, Path]],
    labels: List[str],
    output_path: Optional[Union[str, Path]] = None,
    prefix: Optional[str] = None,
    show_plot: bool = True,
    resolution: int = 100,
    error_thresh: float = 1e-2,
) -> None:
    """Create hexbin error maps for gravity check files.

    This function creates detailed error visualizations comparing exact
    gravitational force calculations with SWIFT simulation results. It
    generates
    both continuous error maps using hexbin plots and binary error maps using
    threshold-based coloring.

    The analysis includes:
    - X-Y, X-Z, and Y-Z spatial projections showing error distribution
    - Error vs distance from center analysis
    - Both logarithmic continuous error maps and binary threshold maps

    Args:
        files: List of file paths to either exact or SWIFT force files.
               The function will automatically find the counterpart files.
        labels: List of labels for the runs being analysed. Must match the
                number of files.
        output_path: Optional path to save the plot. If None, saves to current
                    directory.
        prefix: Optional prefix to add to the output filename. If empty,
                defaults to 'gravity_error_map.png'.
        show_plot: Whether to display the plot interactively.
        resolution: Resolution (gridsize) for hexbin plot. Higher values
                   provide more detail but take longer to compute.
        error_thresh: Error threshold for binary color scale. Errors above this
                     threshold are highlighted in red.

    Raises:
        ValueError: If the number of files and labels do not match.
        FileNotFoundError: If counterpart files cannot be found.

    Examples:
        >>> # Analyze a single gravity check file
        >>> analyse_gravity_error_maps(
        ...     files=["gravity_checks_exact_step0001.dat"],
        ...     labels=["Test Run"],
        ...     show_plot=False
        ... )

        >>> # Compare multiple runs with custom threshold
        >>> analyse_gravity_error_maps(
        ...     files=["run1_exact.dat", "run2_exact.dat"],
        ...     labels=["Run 1", "Run 2"],
        ...     error_thresh=1e-3,
        ...     resolution=150
        ... )
    """
    # Validate input arguments
    if len(files) != len(labels):
        raise ValueError(
            f"Number of files ({len(files)}) and labels ({len(labels)}) "
            f"must match."
        )

    # Process each file and create individual error maps
    for i, (input_file, label) in enumerate(zip(files, labels)):
        try:
            exact_file, swift_file = _find_counterpart_file(str(input_file))
            exact_pos, exact_a, swift_a_grav, _, _ = _load_force_pair(
                exact_file, swift_file
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"Error processing {input_file}: {e}")
            continue

        print(f"Processing error maps for: {label}")
        print(f"  Exact file: {exact_file}")
        print(f"  SWIFT file: {swift_file}")

        exact_a_norm = np.sqrt(
            exact_a[:, 0] ** 2 + exact_a[:, 1] ** 2 + exact_a[:, 2] ** 2
        )

        # Compute relative errors
        diff = exact_a - swift_a_grav
        norm_diff = np.sqrt(
            diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2
        )
        # Avoid division by zero by using a small epsilon
        norm_error = norm_diff / np.maximum(exact_a_norm, 1e-20)

        print(
            f"  Error range: {np.min(norm_error):.2e} to "
            f"{np.max(norm_error):.2e}"
        )

        # Create continuous error map with logarithmic scale
        _create_continuous_error_map(
            exact_pos=exact_pos,
            norm_error=norm_error,
            label=label,
            resolution=resolution,
            output_path=output_path,
            prefix=prefix,
            show_plot=show_plot,
        )

        # Create binary threshold error map
        _create_threshold_error_map(
            exact_pos=exact_pos,
            norm_error=norm_error,
            label=label,
            error_thresh=error_thresh,
            output_path=output_path,
            prefix=prefix,
            show_plot=show_plot,
        )

        print()  # Add spacing between files


def _create_continuous_error_map(
    exact_pos: np.ndarray,
    norm_error: np.ndarray,
    label: str,
    resolution: int,
    output_path: Optional[Union[str, Path]],
    prefix: Optional[str],
    show_plot: bool,
) -> None:
    """Create continuous error map with logarithmic hexbin visualization.

    Args:
        exact_pos: Array of particle positions (N, 3).
        norm_error: Array of normalized errors (N,).
        label: Label for this analysis run.
        resolution: Hexbin grid resolution.
        output_path: Output directory path.
        prefix: Filename prefix.
        show_plot: Whether to show plot interactively.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Gravity Error Maps - {label}", fontsize=16)

    # Define consistent error range for all subplots
    vmin, vmax = 1e-7, 1e-1

    # X-Y projection
    hb1 = ax1.hexbin(
        exact_pos[:, 0],
        exact_pos[:, 1],
        C=norm_error,
        gridsize=resolution,
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.set_title("X-Y Projection")
    ax1.set_aspect("equal")
    cb1 = plt.colorbar(hb1, ax=ax1)
    cb1.set_label("|δa|/|a_exact|")

    # X-Z projection
    hb2 = ax2.hexbin(
        exact_pos[:, 0],
        exact_pos[:, 2],
        C=norm_error,
        gridsize=resolution,
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Z Position")
    ax2.set_title("X-Z Projection")
    ax2.set_aspect("equal")
    cb2 = plt.colorbar(hb2, ax=ax2)
    cb2.set_label("|δa|/|a_exact|")

    # Y-Z projection
    hb3 = ax3.hexbin(
        exact_pos[:, 1],
        exact_pos[:, 2],
        C=norm_error,
        gridsize=resolution,
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax3.set_xlabel("Y Position")
    ax3.set_ylabel("Z Position")
    ax3.set_title("Y-Z Projection")
    ax3.set_aspect("equal")
    cb3 = plt.colorbar(hb3, ax=ax3)
    cb3.set_label("|δa|/|a_exact|")

    # Error vs distance from center. The check files carry no box size,
    # so use the centre of the particles' bounding box.
    center = 0.5 * (exact_pos.min(axis=0) + exact_pos.max(axis=0))
    distance_from_center = np.sqrt(np.sum((exact_pos - center) ** 2, axis=1))

    hb4 = ax4.hexbin(
        distance_from_center,
        norm_error,
        gridsize=resolution,
        cmap="viridis",
        # Counts, not errors: autoscale rather than reuse the error range
        norm=LogNorm(),
        mincnt=1,
    )
    ax4.set_xlabel("Distance from Center")
    ax4.set_ylabel("|δa|/|a_exact|")
    ax4.set_title("Error vs Distance from Center")
    ax4.set_yscale("log")
    cb4 = plt.colorbar(hb4, ax=ax4)
    cb4.set_label("Particle Density")

    plt.tight_layout()

    # Save the figure
    safe_label = label.replace(" ", "_").replace("/", "_")
    png_file = create_output_path(
        str(output_path) if output_path is not None else None,
        prefix,
        f"gravity_error_map_{safe_label}.png",
    )

    save_figure(fig, png_file, show=show_plot, dpi=200)
    print(f"Continuous error map saved to {png_file}")


def _create_threshold_error_map(
    exact_pos: np.ndarray,
    norm_error: np.ndarray,
    label: str,
    error_thresh: float,
    output_path: Optional[Union[str, Path]],
    prefix: Optional[str],
    show_plot: bool,
) -> None:
    """Create binary threshold error map with scatter plot visualization.

    Args:
        exact_pos: Array of particle positions (N, 3).
        norm_error: Array of normalized errors (N,).
        label: Label for this analysis run.
        error_thresh: Error threshold for binary classification.
        output_path: Output directory path.
        prefix: Filename prefix.
        show_plot: Whether to show plot interactively.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Gravity Error Maps (Threshold: {error_thresh:.1e}) - {label}",
        fontsize=16,
    )

    # Remove the axis from the 4th subplot for cleaner layout
    ax4.axis("off")

    # Create binary mask for errors above and below threshold
    mask = norm_error < error_thresh

    # Count particles in each category
    below_thresh = np.sum(mask)
    above_thresh = np.sum(~mask)
    total_particles = len(norm_error)

    print(
        f"  Particles below threshold ({error_thresh:.1e}): "
        f"{below_thresh}/{total_particles} "
        f"({100 * below_thresh / total_particles:.1f}%)"
    )
    print(
        f"  Particles above threshold: "
        f"{above_thresh}/{total_particles} "
        f"({100 * above_thresh / total_particles:.1f}%)"
    )

    # X-Y projection
    ax1.scatter(
        exact_pos[mask, 0],
        exact_pos[mask, 1],
        c="blue",
        s=1,
        alpha=0.6,
        label=f"|δa|/|a_exact| < {error_thresh:.1e}",
    )
    ax1.scatter(
        exact_pos[~mask, 0],
        exact_pos[~mask, 1],
        c="red",
        s=1,
        alpha=0.8,
        label=f"|δa|/|a_exact| ≥ {error_thresh:.1e}",
    )
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.set_title("X-Y Projection")
    ax1.set_aspect("equal")
    ax1.legend(markerscale=5)

    # X-Z projection
    ax2.scatter(
        exact_pos[mask, 0],
        exact_pos[mask, 2],
        c="blue",
        s=1,
        alpha=0.6,
        label=f"|δa|/|a_exact| < {error_thresh:.1e}",
    )
    ax2.scatter(
        exact_pos[~mask, 0],
        exact_pos[~mask, 2],
        c="red",
        s=1,
        alpha=0.8,
        label=f"|δa|/|a_exact| ≥ {error_thresh:.1e}",
    )
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Z Position")
    ax2.set_title("X-Z Projection")
    ax2.set_aspect("equal")

    # Y-Z projection
    ax3.scatter(
        exact_pos[mask, 1],
        exact_pos[mask, 2],
        c="blue",
        s=1,
        alpha=0.6,
        label=f"|δa|/|a_exact| < {error_thresh:.1e}",
    )
    ax3.scatter(
        exact_pos[~mask, 1],
        exact_pos[~mask, 2],
        c="red",
        s=1,
        alpha=0.8,
        label=f"|δa|/|a_exact| ≥ {error_thresh:.1e}",
    )
    ax3.set_xlabel("Y Position")
    ax3.set_ylabel("Z Position")
    ax3.set_title("Y-Z Projection")
    ax3.set_aspect("equal")

    plt.tight_layout()

    # Save the figure
    safe_label = label.replace(" ", "_").replace("/", "_")
    png_file = create_output_path(
        str(output_path) if output_path is not None else None,
        prefix,
        f"gravity_binary_error_map_{safe_label}.png",
    )

    save_figure(fig, png_file, show=show_plot, dpi=200)
    print(f"Binary error map saved to {png_file}")
