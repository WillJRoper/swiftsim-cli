"""Utility mode for small SWIFT helper calculations."""

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
from swiftsimio import load  # type: ignore[import-untyped]
from swiftsimio.metadata import particle_types  # type: ignore[import-untyped]

from swiftsim_cli.params import load_parameters
from swiftsim_cli.profile import load_swift_profile

SPECIES_LABELS = {
    "gas": "Gas",
    "dark_matter": "Dark matter",
    "boundary": "Boundary",
    "sinks": "Sinks",
    "stars": "Stars",
    "black_holes": "Black holes",
    "neutrinos": "Neutrinos",
}


def _as_float(value: object) -> float:
    """Return the scalar value from plain numbers or unit-aware arrays."""
    raw_value = getattr(value, "value", value)
    return float(cast(Any, raw_value))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'utils' mode."""
    subparsers = parser.add_subparsers(
        dest="utils_type",
        help="Utility calculation to run",
        required=True,
    )

    softenings = subparsers.add_parser(
        "softenings",
        help="Compute per-species softenings from an IC file.",
    )
    softenings.add_argument(
        "--ics",
        "--inic",
        dest="ics",
        type=Path,
        required=True,
        help="Path to the SWIFT initial conditions file.",
    )
    softenings.add_argument(
        "-p",
        "--params",
        type=Path,
        required=True,
        help="Path to the SWIFT parameter file.",
    )
    softenings.add_argument(
        "--softening-coeff",
        type=float,
        default=None,
        help="Softening coefficient in units of mean separation.",
    )
    softenings.add_argument(
        "--pivot-redshift",
        type=float,
        default=None,
        help="Pivot redshift for the maximal physical softening.",
    )


def _get_cubic_box_size(boxsize):
    """Return a scalar box size, enforcing the cubic-box assumption."""
    raw_values = boxsize.value if hasattr(boxsize, "value") else boxsize
    values = np.atleast_1d(np.asarray(raw_values))

    if values.size > 1 and not np.allclose(values, values[0]):
        raise ValueError(
            "Softening calculation assumes a cubic box, but the IC box "
            "dimensions differ."
        )

    if hasattr(boxsize, "__getitem__") and values.size > 1:
        return boxsize[0]

    return boxsize


def compute_softening_rows(
    ics_file: Path,
    softening_coeff: float,
    pivot_redshift: float,
) -> tuple[object, list[dict[str, object]]]:
    """Compute softening values for each populated particle species."""
    dataset = load(ics_file)
    box_size = _get_cubic_box_size(dataset.metadata.boxsize)

    rows = []
    for _, species_name in particle_types.particle_name_underscores.items():
        count = int(getattr(dataset.metadata, f"n_{species_name}", 0))
        if count <= 0:
            continue

        mean_separation = box_size / (count ** (1 / 3))
        comoving_softening = softening_coeff * mean_separation
        maximal_softening = comoving_softening / (1 + pivot_redshift)

        rows.append(
            {
                "species": species_name,
                "label": SPECIES_LABELS.get(
                    species_name,
                    species_name.replace("_", " ").title(),
                ),
                "count": count,
                "mean_separation": mean_separation,
                "comoving_softening": comoving_softening,
                "maximal_softening": maximal_softening,
            }
        )

    if not rows:
        raise ValueError(
            f"No supported particle species were found in '{ics_file}'."
        )

    return box_size, rows


def format_softening_report(
    ics_file: Path,
    param_file: Path,
    softening_coeff: float,
    pivot_redshift: float,
    a_begin: float | None,
    box_size,
    rows: list[dict[str, object]],
) -> str:
    """Format a compact human-readable softening report."""
    z_begin = None if a_begin in (None, 0) else (1 / a_begin) - 1
    length_unit = str(getattr(box_size, "units", "code_length"))
    box_value = float(
        box_size.value if hasattr(box_size, "value") else box_size
    )
    mean_sep_header = f"Mean sep [{length_unit}]"
    comoving_header = f"Comoving [{length_unit}]"
    maximal_header = f"Max physical [{length_unit}]"

    lines = [
        f"Softening report for {ics_file}",
        f"Parameter file: {param_file}",
        f"Box size: {box_value:.3e} {length_unit}",
        (
            f"a_begin={a_begin:.6g}, z_begin={z_begin:.6g}"
            if a_begin is not None and z_begin is not None
            else "a_begin: not set"
        ),
        (
            f"softening_coeff={softening_coeff:.6g}, "
            f"pivot_redshift={pivot_redshift:.6g}"
        ),
        "",
        (
            f"{'Species':<14} {'N':>12} {mean_sep_header:>22} "
            f"{comoving_header:>22} {maximal_header:>25}"
        ),
    ]

    for row in rows:
        mean_separation = row["mean_separation"]
        comoving_softening = row["comoving_softening"]
        maximal_softening = row["maximal_softening"]
        mean_sep_value = _as_float(mean_separation)
        comoving_value = _as_float(comoving_softening)
        maximal_value = _as_float(maximal_softening)

        lines.append(
            f"{row['label']:<14} "
            f"{row['count']:>12d} "
            f"{mean_sep_value:>22.3e} "
            f"{comoving_value:>22.3e} "
            f"{maximal_value:>25.3e}"
        )

    return "\n".join(lines)


def run_softenings(args: argparse.Namespace) -> None:
    """Run the softening utility."""
    profile = load_swift_profile()
    params = load_parameters()

    softening_coeff = (
        args.softening_coeff
        if args.softening_coeff is not None
        else profile.softening_coeff
    )
    pivot_redshift = (
        args.pivot_redshift
        if args.pivot_redshift is not None
        else profile.softening_pivot_z
    )

    cosmology = params.get("Cosmology", {})
    a_begin = cosmology.get("a_begin")
    if a_begin is not None:
        a_begin = float(a_begin)

    box_size, rows = compute_softening_rows(
        ics_file=args.ics,
        softening_coeff=softening_coeff,
        pivot_redshift=pivot_redshift,
    )

    print(
        format_softening_report(
            ics_file=args.ics,
            param_file=args.params,
            softening_coeff=softening_coeff,
            pivot_redshift=pivot_redshift,
            a_begin=a_begin,
            box_size=box_size,
            rows=rows,
        )
    )


def run(args: argparse.Namespace) -> None:
    """Run the requested utility command."""
    if args.utils_type == "softenings":
        run_softenings(args)
    else:
        raise ValueError(f"Unknown utility type: {args.utils_type}")
