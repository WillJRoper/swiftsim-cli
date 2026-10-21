"""Gravity checks analysis module for SWIFT simulations."""

import argparse
import re
import warnings
from pathlib import Path
from typing import Any, NamedTuple, Union

import numpy as np

from swiftsim_cli.modes.common_args import (
    add_labels_argument,
    add_output_arguments,
    existing_file,
    positive_float,
    positive_int,
    resolve_labels,
)
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.messages import plural, reading, warn
from swiftsim_cli.utilities.plotting import save_figure
from swiftsim_cli.utilities.tables import create_ascii_table

# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "gravity_checks_analysis"

EXACT_PREFIX = "gravity_checks_exact"

# The leading columns shared by exact and SWIFT files:
# id, pos[3], a[3], potential (SWIFT writes the id with %lld).
_FORCE_DTYPE = np.dtype(
    [
        ("id", np.int64),
        ("pos", np.float64, (3,)),
        ("a", np.float64, (3,)),
        ("pot", np.float64),
    ]
)

# Error components: (key, axis label), in panel order
ERROR_COMPONENTS = (
    ("x", "|δa_x|/|a_exact|"),
    ("y", "|δa_y|/|a_exact|"),
    ("z", "|δa_z|/|a_exact|"),
    ("norm", "|δa|/|a_exact|"),
    ("pot", "|δφ|/|φ_exact|"),
)

# Positions agreeing to this relative tolerance belong to the same snapshot
_POSITION_RTOL = 1e-6

# Runs whose percentiles fit in the summary panel; the rest are printed only
_MAX_SUMMARY_RUNS = 6


class ForcePair(NamedTuple):
    """An exact/SWIFT check pair aligned by particle ID."""

    pos: np.ndarray
    exact_a: np.ndarray
    swift_a: np.ndarray
    exact_pot: np.ndarray
    swift_pot: np.ndarray


def _swift_order(path: Path) -> int:
    """Return the multipole order in a SWIFT check filename (-1 if none)."""
    match = re.search(r"_order(\d+)\.dat$", path.name)
    return int(match.group(1)) if match else -1


def _find_counterpart_files(
    filepath: str, all_orders: bool = False
) -> list[tuple[str, str]]:
    """Find the exact/SWIFT pair(s) for a given check file.

    Args:
        filepath: Path to either an exact or SWIFT gravity check file.
        all_orders: For an exact file, return a pair for every SWIFT file
            of that step (one per multipole order) rather than the lowest
            order only.

    Returns:
        ``(exact_file, swift_file)`` pairs.

    Raises:
        ValueError: If step number cannot be extracted from filename.
        FileNotFoundError: If counterpart file cannot be found.
    """
    file_path = Path(filepath)
    filename = file_path.name
    directory = file_path.parent

    # Extract the step number, reusing SWIFT's own zero padding (%.4d, so
    # steps >= 10000 have more digits)
    step_match = re.search(r"step(\d+)", filename)
    if not step_match:
        raise ValueError(f"Could not extract step number from {filename}")
    step_num = step_match.group(1)

    if not filename.startswith(EXACT_PREFIX):
        # A SWIFT file: find the exact file of the same step
        for name in (
            f"{EXACT_PREFIX}_step{step_num}.dat",
            f"{EXACT_PREFIX}_periodic_step{step_num}.dat",
        ):
            if (directory / name).exists():
                return [(str(directory / name), filepath)]
        raise FileNotFoundError(f"No exact file found for step {step_num}")

    # An exact file: find the SWIFT file(s) of exactly this step
    swift_re = re.compile(
        rf"gravity_checks_swift_step{step_num}(_order\d+)?\.dat"
    )
    swift_files = sorted(
        (
            path
            for path in directory.glob(f"gravity_checks_swift_step{step_num}*")
            if swift_re.fullmatch(path.name)
        ),
        key=lambda path: (_swift_order(path), path.name),
    )
    if not swift_files:
        raise FileNotFoundError(f"No SWIFT files found for step {step_num}")
    if len(swift_files) > 1 and not all_orders:
        warn(
            f"{len(swift_files)} SWIFT files found for step {step_num}; "
            f"using {swift_files[0]}. Pass --all-orders to compare every "
            "multipole order, or the SWIFT file to choose one."
        )
        swift_files = swift_files[:1]
    return [(filepath, str(path)) for path in swift_files]


def _load_check_file(path: str) -> np.ndarray:
    """Read the shared leading columns of a gravity check file.

    Raises:
        ValueError: If the file cannot be parsed or has no data rows.
    """
    with reading(path), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # empty-file warning
        data = np.loadtxt(path, usecols=range(8), dtype=_FORCE_DTYPE, ndmin=1)
    if data.size == 0:
        raise ValueError(f"{path}: no data rows")
    return data


def _load_force_pair(exact_file: str, swift_file: str) -> ForcePair:
    """Load an exact/SWIFT gravity check pair aligned by particle ID.

    Particles with non-finite values in either file are dropped with a
    warning, and a warning is given if the two files place the same
    particle at different positions (SWIFT reuses an existing exact file
    whenever N, G and the periodicity match, even from another run).

    Args:
        exact_file: Path to the exact force file.
        swift_file: Path to the SWIFT force file.

    Returns:
        The particles present in both files, sorted by particle ID.
        Positions are taken from the exact file.

    Raises:
        ValueError: If the files share no usable particles.
    """
    exact = _load_check_file(exact_file)
    swift = _load_check_file(swift_file)

    common_ids, exact_index, swift_index = np.intersect1d(
        exact["id"], swift["id"], return_indices=True
    )
    if common_ids.size == 0:
        raise ValueError(
            f"No common particle IDs between {exact_file} and {swift_file}"
        )
    if common_ids.size != len(exact) or common_ids.size != len(swift):
        warn(
            f"particle IDs differ between {exact_file} "
            f"({len(exact)} particles) and {swift_file} "
            f"({len(swift)} particles); using the "
            f"{common_ids.size} common particles."
        )
    exact = exact[exact_index]
    swift = swift[swift_index]

    finite = np.ones(len(exact), dtype=bool)
    for data in (exact, swift):
        for field in ("pos", "a", "pot"):
            values = data[field].reshape(len(data), -1)
            finite &= np.isfinite(values).all(axis=1)
    if not finite.all():
        if not finite.any():
            raise ValueError(
                f"No particles with finite values in both {exact_file} and "
                f"{swift_file}"
            )
        warn(
            f"ignoring {plural(int((~finite).sum()), 'particle')} with "
            f"non-finite values in {exact_file} or {swift_file}."
        )
        exact = exact[finite]
        swift = swift[finite]

    moved = ~np.isclose(
        exact["pos"], swift["pos"], rtol=_POSITION_RTOL, atol=0.0
    ).all(axis=1)
    if moved.any():
        warn(
            f"{plural(int(moved.sum()), 'particle')} sit at different "
            f"positions in {exact_file} and {swift_file} (max offset "
            f"{np.abs(exact['pos'] - swift['pos']).max():.3g}); the exact "
            "file may come from another run or step."
        )

    return ForcePair(
        exact["pos"], exact["a"], swift["a"], exact["pot"], swift["pot"]
    )


def _relative_errors(pair: ForcePair) -> dict[str, np.ndarray]:
    """Return the relative force and potential errors per particle.

    The potential is compared after removing the mean offset between the
    two files (their zero points differ). Errors that are undefined
    because the exact value is zero are NaN.

    Args:
        pair: The aligned exact/SWIFT values.

    Returns:
        One array per `ERROR_COMPONENTS` key.
    """
    exact_norm = np.linalg.norm(pair.exact_a, axis=1)
    diff = pair.exact_a - pair.swift_a
    exact_pot = pair.exact_pot - np.mean(pair.exact_pot - pair.swift_pot)
    with np.errstate(divide="ignore", invalid="ignore"):
        errors = {
            "x": np.abs(diff[:, 0]) / exact_norm,
            "y": np.abs(diff[:, 1]) / exact_norm,
            "z": np.abs(diff[:, 2]) / exact_norm,
            "norm": np.linalg.norm(diff, axis=1) / exact_norm,
            "pot": np.abs(exact_pot - pair.swift_pot) / np.abs(exact_pot),
        }
    for values in errors.values():
        values[~np.isfinite(values)] = np.nan
    return errors


def _resolve_inputs(
    files: list[str], labels: list[str], all_orders: bool
) -> list[tuple[str, str, str, str]]:
    """Find the check pairs of every input, skipping failures.

    Returns:
        ``(input, label, exact_file, swift_file)`` per pair; with several
        multipole orders per input the order is appended to the label.
    """
    jobs = []
    for input_file, label in zip(files, labels):
        try:
            pairs = _find_counterpart_files(str(input_file), all_orders)
        except (ValueError, FileNotFoundError) as e:
            warn(f"skipping {input_file}: {e}")
            continue
        for exact_file, swift_file in pairs:
            order = _swift_order(Path(swift_file))
            run_label = label
            if len(pairs) > 1:
                run_label += f" order {order}" if order >= 0 else " (no order)"
            jobs.append((str(input_file), run_label, exact_file, swift_file))
    return jobs


def _style(index: int, count: int) -> dict[str, Any]:
    """Line colour and style for run ``index`` of ``count``.

    Colours come from tab10 (tab20 for more than ten runs); the line style
    changes each time the colours run out, so no two runs look alike.
    """
    import matplotlib

    cmap = matplotlib.colormaps["tab10" if count <= 10 else "tab20"]
    styles = ("-", "--", ":", "-.")
    return {
        "color": cmap(index % cmap.N),
        "linestyle": styles[(index // cmap.N) % len(styles)],
    }


def analyse_force_checks(
    files: list[str],
    labels: list[str],
    output_path: Union[str, Path, None] = None,
    prefix: Union[str, None] = None,
    show_plot: bool = False,
    min_error: float = 1e-7,
    max_error: float = 3e-1,
    num_bins: int = 64,
    all_orders: bool = False,
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
        min_error: Lower edge of the histograms (default: 1e-7).
        max_error: Upper edge of the histograms (default: 3e-1). Errors
            outside the range are counted and reported, not plotted.
        num_bins: Number of bins for histogram (default: 64).
        all_orders: For an exact file with SWIFT files of several
            multipole orders, compare every order (default: the lowest).

    Inputs that cannot be processed are reported as warnings and skipped.

    Raises:
        ValueError: If the number of files and labels do not match, the
            error range is invalid, or no input could be processed.

    Note:
        The function expects gravity check files in the format produced by
        SWIFT with names containing step numbers and 'exact' or 'swift'
        identifiers. Files should contain columns for:
        [ID, x, y, z, ax, ay, az, potential]
    """
    import matplotlib.pyplot as plt

    # Make sure the number of files and labels match
    if len(files) != len(labels):
        raise ValueError("Number of files and labels must match.")
    if not 0 < min_error < max_error:
        raise ValueError(
            f"need 0 < --min-error < --max-error, got {min_error:g} and "
            f"{max_error:g}"
        )

    # Load everything first so a failure leaves no plot behind
    runs = []
    for input_file, label, exact_file, swift_file in _resolve_inputs(
        files, labels, all_orders
    ):
        try:
            errors = _relative_errors(_load_force_pair(exact_file, swift_file))
        except ValueError as e:
            warn(f"skipping {input_file}: {e}")
            continue
        runs.append((label, exact_file, swift_file, errors))
    if not runs:
        raise ValueError("none of the gravity check inputs could be processed")

    log_edges = np.linspace(
        np.log10(min_error), np.log10(max_error), num_bins + 1
    )
    bin_size = log_edges[1] - log_edges[0]
    bin_edges = 10**log_edges
    bin_centres = 10 ** (0.5 * (log_edges[1:] + log_edges[:-1]))

    # Plot styling, scoped so it does not leak into later modes
    # (inline so newer matplotlib stubs can check the rcParams keys)
    with plt.rc_context(
        {
            "axes.labelsize": 13,
            "font.size": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "text.usetex": False,
            "lines.linewidth": 2.0,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        panels = axes.flat[:5]
        summary_ax = axes.flat[5]
        summary_ax.axis("off")
        summary_lines: list[tuple[str, Any]] = []

        for i, (label, exact_file, swift_file, errors) in enumerate(runs):
            style = _style(i, len(runs))
            rows = []
            for ax, (key, name) in zip(panels, ERROR_COMPONENTS):
                values = errors[key][np.isfinite(errors[key])]
                undefined = errors[key].size - values.size
                if values.size == 0:
                    rows.append([name, "-", "-", "-", "-", "-", undefined])
                    continue
                hist, _ = np.histogram(values, bins=bin_edges)
                ax.semilogx(
                    bin_centres,
                    hist / (values.size * bin_size),
                    label=label,
                    **style,
                )
                median, per90, per99 = np.percentile(values, [50, 90, 99])
                ax.axvline(median, color=style["color"], linewidth=1, ls=":")
                below = 100 * np.mean(values < min_error)
                above = 100 * np.mean(values > max_error)
                rows.append(
                    [
                        name,
                        f"{median:.3e}",
                        f"{per90:.3e}",
                        f"{per99:.3e}",
                        f"{below:.1f}",
                        f"{above:.1f}",
                        undefined,
                    ]
                )
                if key == "norm":
                    summary_lines.append(
                        (
                            f"{label}\n  50% {median:.2e}  90% {per90:.2e}"
                            f"  99% {per99:.2e}\n  outside range: "
                            f"{below:.1f}% below, {above:.1f}% above",
                            style["color"],
                        )
                    )

            print(
                create_ascii_table(
                    [
                        "Error",
                        "Median",
                        "90%",
                        "99%",
                        f"< {min_error:g} [%]",
                        f"> {max_error:g} [%]",
                        "Undefined",
                    ],
                    rows,
                    title=f"{label}: {Path(exact_file).name} vs "
                    f"{Path(swift_file).name}",
                )
            )
            print()

        for ax, (_, name) in zip(panels, ERROR_COMPONENTS):
            ax.set_xlabel(name)
            ax.set_xlim(min_error, max_error)
            ax.set_ylim(0, None)
            ax.margins(y=0.1)
            ax.grid(True, alpha=0.3)
        for ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("Particle fraction per dex")
        axes[1, 0].legend(loc="upper left")

        summary_ax.text(
            0.0,
            1.0,
            "|δa|/|a_exact| percentiles\n(dotted lines: medians)",
            transform=summary_ax.transAxes,
            va="top",
            fontweight="bold",
        )
        for j, (text, color) in enumerate(summary_lines[:_MAX_SUMMARY_RUNS]):
            summary_ax.text(
                0.0,
                0.86 - j * 0.145,
                text,
                transform=summary_ax.transAxes,
                va="top",
                family="monospace",
                fontsize=8.5,
                color=color,
            )
        if len(summary_lines) > _MAX_SUMMARY_RUNS:
            summary_ax.text(
                0.0,
                0.0,
                f"+{len(summary_lines) - _MAX_SUMMARY_RUNS} more runs: see "
                "the terminal output",
                transform=summary_ax.transAxes,
                fontsize=8.5,
            )
        fig.tight_layout()

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
        description="Compare SWIFT gravity forces with exact gravity "
        "checks. Give either file of each pair (gravity_checks_exact_* or "
        "gravity_checks_swift_*); its counterpart is found next to it. An "
        "exact file with SWIFT files of several multipole orders is "
        "compared with the lowest order unless --all-orders is given. "
        "Requires SWIFT configured with --enable-gravity-force-checks=N.",
    )
    gravity_parser.set_defaults(func=run_gravity_checks)

    gravity_parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="Gravity check files to analyse (either exact or SWIFT files).",
        type=existing_file,
    )
    add_labels_argument(gravity_parser)
    add_output_arguments(gravity_parser, OUT_SUBDIR)

    gravity_parser.add_argument(
        "--min-error",
        type=positive_float,
        metavar="ERR",
        help="Lower edge of the error histograms (default: 1e-7). Errors "
        "below it are counted in the summary.",
        default=1e-7,
    )

    gravity_parser.add_argument(
        "--max-error",
        type=positive_float,
        metavar="ERR",
        help="Upper edge of the error histograms (default: 3e-1). Errors "
        "above it are counted in the summary.",
        default=3e-1,
    )

    gravity_parser.add_argument(
        "--num-bins",
        type=positive_int,
        metavar="N",
        help="Number of bins for histogram (default: 64).",
        default=64,
    )

    gravity_parser.add_argument(
        "--all-orders",
        action="store_true",
        help="For an exact file, compare the SWIFT files of every "
        "multipole order (default: the lowest order only).",
    )

    return gravity_parser


def run_gravity_checks(args: argparse.Namespace) -> None:
    """Execute the gravity checks analysis.

    Args:
        args: Parsed command line arguments containing the analysis parameters.
    """
    analyse_force_checks(
        files=args.files,
        labels=resolve_labels(args.files, args.labels),
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        min_error=args.min_error,
        max_error=args.max_error,
        num_bins=args.num_bins,
        all_orders=args.all_orders,
    )
