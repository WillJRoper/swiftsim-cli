"""Gravity error maps analysis module for SWIFT simulations.

This module provides functionality to create hexbin error maps for gravity
check files, comparing exact solutions with SWIFT simulation results. It
supports creating both continuous error visualizations and binary
threshold-based error maps.
"""

import argparse
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

from swiftsim_cli.modes.common_args import (
    add_labels_argument,
    add_output_arguments,
    existing_file,
    positive_float,
    positive_int,
    resolve_labels,
)
from swiftsim_cli.utilities.io import create_output_path, slugify
from swiftsim_cli.utilities.messages import plural, warn
from swiftsim_cli.utilities.plotting import save_figure

from .gravity_checks import (
    _load_force_pair,
    _relative_errors,
    _resolve_inputs,
)

# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "gravity_error_maps_analysis"

ERROR_LABEL = "|δa|/|a_exact|"

# (x, y) position columns and axis names of the three projections
_PROJECTIONS = ((0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z"))

# Below this many particles hexbins are mostly empty: scatter instead
_SCATTER_MAX_PARTICLES = 5000

# Percentiles of the errors used for the default colour range
_COLOUR_PERCENTILES = (1, 99)


def add_gravity_error_maps_arguments(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add CLI arguments for gravity error maps analysis.

    Args:
        subparsers: The subparsers object to add the gravity-error-maps parser.
    """
    error_map_parser = subparsers.add_parser(
        "gravity-error-maps",
        help="Create hexbin error maps for gravity check files.",
        description="Create maps of the relative gravity force error for "
        "gravity check files: the median error per hexagon (or per "
        "particle, for small samples) in three projections, and a binary "
        "map of the particles above --thresh. Give either file of each "
        "pair (gravity_checks_exact_* or gravity_checks_swift_*); its "
        "counterpart is found next to it. Requires SWIFT configured with "
        "--enable-gravity-force-checks=N.",
    )
    error_map_parser.set_defaults(func=run_gravity_error_maps)

    error_map_parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="Gravity check files to analyse (either exact or SWIFT files).",
        type=existing_file,
    )
    add_labels_argument(error_map_parser)
    add_output_arguments(error_map_parser, OUT_SUBDIR)

    error_map_parser.add_argument(
        "--resolution",
        type=positive_int,
        metavar="N",
        help="Hexbin grid size (default: 100).",
        default=100,
    )

    error_map_parser.add_argument(
        "--thresh",
        type=positive_float,
        metavar="ERR",
        help="Error threshold of the binary map: particles at or above it "
        "are drawn red, below it blue (default: 1e-2).",
        default=1e-2,
    )

    error_map_parser.add_argument(
        "--vmin",
        type=positive_float,
        metavar="ERR",
        help="Lower end of the error colour scale (default: the 1st "
        "percentile of the errors).",
        default=None,
    )
    error_map_parser.add_argument(
        "--vmax",
        type=positive_float,
        metavar="ERR",
        help="Upper end of the error colour scale (default: the 99th "
        "percentile of the errors).",
        default=None,
    )

    error_map_parser.add_argument(
        "--box-size",
        type=positive_float,
        metavar="L",
        help="Periodic box size, in the units of the check files. When "
        "given, distances from the centre wrap around the box.",
        default=None,
    )


def run_gravity_error_maps(args: argparse.Namespace) -> None:
    """Execute the gravity error map analysis.

    Args:
        args: Parsed command line arguments containing analysis parameters.
    """
    analyse_gravity_error_maps(
        files=args.files,
        labels=resolve_labels(args.files, args.labels),
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        resolution=args.resolution,
        error_thresh=args.thresh,
        vmin=args.vmin,
        vmax=args.vmax,
        box_size=args.box_size,
    )


def analyse_gravity_error_maps(
    files: List[Union[str, Path]],
    labels: List[str],
    output_path: Optional[Union[str, Path]] = None,
    prefix: Optional[str] = None,
    show_plot: bool = False,
    resolution: int = 100,
    error_thresh: float = 1e-2,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    box_size: Optional[float] = None,
) -> None:
    """Create error maps for gravity check files.

    For each input two figures are written: the median relative force error
    in the X-Y, X-Z and Y-Z projections plus the error against distance
    from the centre (the median particle position), and a binary map of
    particles above/below ``error_thresh`` with an error histogram.

    Args:
        files: List of file paths to either exact or SWIFT force files.
               The function will automatically find the counterpart files.
        labels: List of labels for the runs being analysed. Must match the
                number of files.
        output_path: Directory whose ``gravity_error_maps_analysis``
            subdirectory receives the plots. If None, the current directory
            is used.
        prefix: Optional prefix to add to the output filenames.
        show_plot: Whether to display the plot interactively.
        resolution: Resolution (gridsize) for hexbin plot. Higher values
                   provide more detail but take longer to compute.
        error_thresh: Error threshold for the binary map. Errors at or
            above this threshold are highlighted in red.
        vmin: Lower end of the colour scale (default: 1st percentile).
        vmax: Upper end of the colour scale (default: 99th percentile).
        box_size: Periodic box size; when given, distances from the centre
            use the nearest periodic image.

    Inputs that cannot be processed are reported as warnings and skipped.

    Raises:
        ValueError: If the number of files and labels do not match, the
            colour range is invalid, or no input could be processed.
    """
    if len(files) != len(labels):
        raise ValueError(
            f"Number of files ({len(files)}) and labels ({len(labels)}) "
            f"must match."
        )
    if vmin is not None and vmax is not None and not vmin < vmax:
        raise ValueError(f"need --vmin < --vmax, got {vmin:g} and {vmax:g}")

    used_slugs: set[str] = set()
    processed = 0
    for input_file, label, exact_file, swift_file in _resolve_inputs(
        [str(f) for f in files], labels, all_orders=False
    ):
        try:
            pair = _load_force_pair(exact_file, swift_file)
        except ValueError as e:
            warn(f"skipping {input_file}: {e}")
            continue
        norm_error = _relative_errors(pair)["norm"]
        defined = np.isfinite(norm_error)
        if not defined.any():
            warn(f"skipping {input_file}: no particle has a defined error")
            continue
        if not defined.all():
            warn(
                f"{input_file}: ignoring "
                f"{plural(int((~defined).sum()), 'particle')} with zero "
                "exact acceleration."
            )
        positions = pair.pos[defined]
        norm_error = norm_error[defined]
        processed += 1

        print(f"Processing error maps for: {label}")
        print(f"  Exact file: {exact_file}")
        print(f"  SWIFT file: {swift_file}")
        print(
            f"  Error range: {np.min(norm_error):.2e} to "
            f"{np.max(norm_error):.2e}"
        )

        # Unique, filesystem-safe name per label
        slug = base = slugify(label, fallback="run")
        n = 1
        while slug in used_slugs:
            n += 1
            slug = f"{base}_{n}"
        used_slugs.add(slug)

        _create_continuous_error_map(
            exact_pos=positions,
            norm_error=norm_error,
            label=label,
            resolution=resolution,
            output_path=output_path,
            prefix=prefix,
            show_plot=show_plot,
            slug=slug,
            vmin=vmin,
            vmax=vmax,
            box_size=box_size,
        )
        _create_threshold_error_map(
            exact_pos=positions,
            norm_error=norm_error,
            label=label,
            error_thresh=error_thresh,
            output_path=output_path,
            prefix=prefix,
            show_plot=show_plot,
            slug=slug,
        )
        print()

    if not processed:
        raise ValueError("none of the gravity check inputs could be processed")


def _colour_range(
    norm_error: np.ndarray, vmin: Optional[float], vmax: Optional[float]
) -> tuple[float, float]:
    """Return the colour range, filling unset ends from percentiles."""
    positive = norm_error[norm_error > 0]
    if positive.size == 0:  # every force exact: any range will do
        positive = np.array([1e-7, 1e-1])
    low, high = np.percentile(positive, _COLOUR_PERCENTILES)
    low = vmin if vmin is not None else low
    high = vmax if vmax is not None else high
    if not low < high:  # percentiles collapsed or clash with a given end
        if vmax is not None:
            low = high / 100
        elif vmin is not None:
            high = low * 100
        else:
            low, high = low / 10, low * 10
    return float(low), float(high)


def _distance_from_centre(
    pos: np.ndarray, box_size: Optional[float]
) -> np.ndarray:
    """Distance of each particle from the median particle position.

    The median is robust to the outliers that shift a bounding-box
    centre. With ``box_size`` the nearest periodic image is used.
    """
    offset = pos - np.median(pos, axis=0)
    if box_size is not None:
        offset -= box_size * np.round(offset / box_size)
    return np.sqrt(np.sum(offset**2, axis=1))


def _create_continuous_error_map(
    exact_pos: np.ndarray,
    norm_error: np.ndarray,
    label: str,
    resolution: int,
    output_path: Optional[Union[str, Path]],
    prefix: Optional[str],
    show_plot: bool,
    slug: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    box_size: Optional[float] = None,
) -> None:
    """Create the median-error map in three projections plus radial panel.

    Args:
        exact_pos: Array of particle positions (N, 3).
        norm_error: Array of normalized errors (N,).
        label: Label for this analysis run.
        resolution: Hexbin grid resolution.
        output_path: Output directory path.
        prefix: Filename prefix.
        show_plot: Whether to show plot interactively.
        slug: Filename-safe label (default: slugified ``label``).
        vmin: Lower end of the colour scale (default: 1st percentile).
        vmax: Upper end of the colour scale (default: 99th percentile).
        box_size: Periodic box size for the radial distances.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterSciNotation

    low, high = _colour_range(norm_error, vmin, vmax)
    scatter = len(norm_error) <= _SCATTER_MAX_PARTICLES
    # Zero errors cannot go on a log scale: show them at the bottom end
    colour_values = np.clip(norm_error, low * 1e-3, None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Gravity Error Maps - {label}", fontsize=16)

    for ax, (i, j, xname, yname) in zip(axes.flat, _PROJECTIONS):
        if scatter:
            order = np.argsort(colour_values)  # largest errors on top
            mappable: Any = ax.scatter(
                exact_pos[order, i],
                exact_pos[order, j],
                c=colour_values[order],
                s=12,
                cmap="viridis",
                norm=LogNorm(vmin=low, vmax=high),
            )
        else:
            mappable = ax.hexbin(
                exact_pos[:, i],
                exact_pos[:, j],
                C=colour_values,
                reduce_C_function=np.median,
                gridsize=resolution,
                cmap="viridis",
                norm=LogNorm(vmin=low, vmax=high),
            )
        ax.set_xlabel(f"{xname} [internal units]")
        ax.set_ylabel(f"{yname} [internal units]")
        ax.set_title(f"{xname}-{yname} Projection")
        ax.set_aspect("equal")
        colourbar = plt.colorbar(mappable, ax=ax, extend="both")
        # Narrow ranges have few decades: label some minor ticks too
        colourbar.ax.yaxis.set_minor_formatter(
            LogFormatterSciNotation(
                labelOnlyBase=False, minor_thresholds=(2, 1)
            )
        )
        colourbar.set_label(
            ERROR_LABEL if scatter else f"Median {ERROR_LABEL}"
        )

    ax4 = axes[1, 1]
    distance = _distance_from_centre(exact_pos, box_size)
    # Outliers far from the centre squash the rest: use a log axis then
    xscale = "linear"
    if distance.max() > 20 * np.median(distance) and np.any(distance > 0):
        xscale = "log"
        distance = np.clip(distance, distance[distance > 0].min(), None)
    if scatter:
        ax4.scatter(distance, colour_values, s=8, alpha=0.6)
        ax4.set_xscale(xscale)
    else:
        counts = ax4.hexbin(
            distance,
            colour_values,
            gridsize=resolution,
            cmap="viridis",
            xscale=xscale,
            yscale="log",
            # Counts, not errors: autoscale rather than reuse the error range
            norm=LogNorm(),
            mincnt=1,
        )
        plt.colorbar(counts, ax=ax4).set_label("Particles")
    ax4.set_xlabel("Distance from median position [internal units]")
    ax4.set_ylabel(ERROR_LABEL)
    ax4.set_title("Error vs Distance from Centre")
    ax4.set_yscale("log")

    plt.tight_layout()

    png_file = create_output_path(
        output_path,
        prefix,
        f"gravity_error_map_{slug or slugify(label, fallback='run')}.png",
        OUT_SUBDIR,
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
    slug: Optional[str] = None,
) -> None:
    """Create binary threshold error map plus an error histogram.

    Args:
        exact_pos: Array of particle positions (N, 3).
        norm_error: Array of normalized errors (N,).
        label: Label for this analysis run.
        error_thresh: Error threshold for binary classification.
        output_path: Output directory path.
        prefix: Filename prefix.
        show_plot: Whether to show plot interactively.
        slug: Filename-safe label (default: slugified ``label``).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Gravity Error Maps (Threshold: {error_thresh:.1e}) - {label}",
        fontsize=16,
    )

    below = norm_error < error_thresh
    total = len(norm_error)
    n_below = int(below.sum())
    n_above = total - n_below
    print(
        f"  Particles below threshold ({error_thresh:.1e}): "
        f"{n_below}/{total} ({100 * n_below / total:.1f}%)"
    )
    print(
        f"  Particles above threshold: "
        f"{n_above}/{total} ({100 * n_above / total:.1f}%)"
    )

    for ax, (i, j, xname, yname) in zip(axes.flat, _PROJECTIONS):
        ax.scatter(
            exact_pos[below, i],
            exact_pos[below, j],
            c="tab:blue",
            s=2,
            alpha=0.6,
            label=f"{ERROR_LABEL} < {error_thresh:.1e} ({n_below})",
        )
        ax.scatter(
            exact_pos[~below, i],
            exact_pos[~below, j],
            c="tab:red",
            s=2,
            alpha=0.8,
            label=f"{ERROR_LABEL} ≥ {error_thresh:.1e} ({n_above})",
        )
        ax.set_xlabel(f"{xname} [internal units]")
        ax.set_ylabel(f"{yname} [internal units]")
        ax.set_title(f"{xname}-{yname} Projection")
        ax.set_aspect("equal")
    axes[0, 0].legend(markerscale=4, fontsize=9)

    # Error distribution, with the threshold marked
    ax4 = axes[1, 1]
    positive = norm_error[norm_error > 0]
    if positive.size:
        low = min(positive.min(), error_thresh) / 2
        high = max(positive.max(), error_thresh) * 2
        ax4.hist(
            np.clip(norm_error, low, None),
            bins=np.logspace(np.log10(low), np.log10(high), 50),
            color="tab:gray",
        )
    ax4.axvline(
        error_thresh, color="tab:red", linestyle="--", label="threshold"
    )
    ax4.set_xscale("log")
    ax4.set_xlabel(ERROR_LABEL)
    ax4.set_ylabel("Particles")
    ax4.set_title(f"{100 * n_above / total:.1f}% at or above threshold")
    ax4.legend()

    plt.tight_layout()

    png_file = create_output_path(
        output_path,
        prefix,
        f"gravity_binary_error_map_{slug or slugify(label, fallback='run')}"
        ".png",
        OUT_SUBDIR,
    )
    save_figure(fig, png_file, show=show_plot, dpi=200)
    print(f"Binary error map saved to {png_file}")
