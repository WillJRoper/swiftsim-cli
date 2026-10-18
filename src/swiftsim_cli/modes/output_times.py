"""Output-times mode for generating SWIFT output lists.

The list is described by a first output, a spacing and a final output,
each in its own unit (redshift, time or scale factor; the spacing can also
be in log10 scale factor). The outputs are generated in the spacing's
unit, converted to scale factor (through the run's cosmology where time is
involved) to merge snapshots and snipshots, sort them and drop
duplicates, then converted once to the list's unit and written.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from swiftsim_cli.cosmology import (
    convert_redshift_to_time,
    convert_time_to_redshift,
    get_cosmology,
    internal_time_unit_in_gyr,
)
from swiftsim_cli.modes.common_args import add_params_argument, positive_float
from swiftsim_cli.params import load_parameters
from swiftsim_cli.utilities.messages import note

# Units of the first/final outputs, and of the spacings
POINT_UNITS = ("z", "time", "scale_factor")
DELTA_UNITS = ("z", "time", "scale_factor", "log_scale_factor")

# List units (--list-units value -> internal unit, file header)
LIST_UNITS = {
    "redshift": ("z", "Redshift"),
    "time": ("time", "Time"),
    "scale-factor": ("scale_factor", "Scale Factor"),
}

UNIT_NAMES = {
    "z": "redshift",
    "time": "time",
    "scale_factor": "scale factor",
    "log_scale_factor": "log10 scale factor",
}

TIME_UNITS_HELP = (
    "in Gyr since the Big Bang for a cosmological run (written to the list "
    "in internal time units), otherwise in internal time units"
)

# Relative tolerance in scale factor for two outputs to be the same (the
# time <-> scale factor conversion is accurate to ~1e-8)
DUPLICATE_RTOL = 1e-7


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'output-times' mode."""
    parser.add_argument(
        "--output-list",
        type=Path,
        default=Path("output_list.txt"),
        metavar="FILE",
        help="File to write the list to (default: output_list.txt).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output list.",
    )
    add_params_argument(parser)
    parser.add_argument(
        "--no-cosmology",
        action="store_true",
        help="Ignore the parameter file's Cosmology section (treat the run "
        "as non-cosmological).",
    )

    first = parser.add_mutually_exclusive_group()
    final = parser.add_mutually_exclusive_group()
    for group, which, extra in (
        (first, "first", ""),
        (
            final,
            "final",
            " (default: Cosmology:a_end, for a cosmological run)",
        ),
    ):
        group.add_argument(
            f"--{which}-snap-z",
            type=float,
            metavar="Z",
            help=f"Redshift of the {which} output{extra}.",
        )
        group.add_argument(
            f"--{which}-snap-time",
            type=float,
            metavar="T",
            help=f"Time of the {which} output, {TIME_UNITS_HELP}.",
        )
        group.add_argument(
            f"--{which}-snap-scale-factor",
            type=float,
            metavar="A",
            help=f"Scale factor of the {which} output{extra}.",
        )

    delta = parser.add_mutually_exclusive_group()
    snip = parser.add_mutually_exclusive_group()
    for group, prefix, what in (
        (delta, "delta", "snapshots"),
        (snip, "snipshot-delta", "snipshots"),
    ):
        group.add_argument(
            f"--{prefix}-z",
            type=positive_float,
            metavar="DZ",
            help=f"Redshift interval between {what}.",
        )
        group.add_argument(
            f"--{prefix}-time",
            type=positive_float,
            metavar="DT",
            help=f"Time interval between {what}, {TIME_UNITS_HELP}.",
        )
        group.add_argument(
            f"--{prefix}-scale-factor",
            type=positive_float,
            metavar="DA",
            help=f"Scale factor interval between {what}.",
        )
        group.add_argument(
            f"--{prefix}-log-scale-factor",
            type=positive_float,
            metavar="DLOGA",
            help=f"Interval in log10(scale factor) between {what}.",
        )
    parser.add_argument(
        "--list-units",
        choices=list(LIST_UNITS),
        default=None,
        metavar="UNITS",
        help="Units of the list: redshift, time or scale-factor (default: "
        "those of --delta-*). Lists with snipshots need "
        "Snapshots:select_output_on and a select-output file in SWIFT.",
    )


@dataclass
class Point:
    """A value in one of the output units."""

    unit: str
    value: float


def _given(args: dict, prefix: str, units: tuple[str, ...]) -> Point | None:
    """Return the one ``<prefix>_<unit>`` option given, if any."""
    given = [
        Point(unit, float(args[f"{prefix}_{unit}"]))
        for unit in units
        if args.get(f"{prefix}_{unit}") is not None
    ]
    if len(given) > 1:
        names = ", ".join(UNIT_NAMES[p.unit] for p in given)
        raise ValueError(f"give the {prefix} in one unit only, not {names}")
    return given[0] if given else None


class Converter:
    """Convert between output units through the scale factor.

    Redshift and scale factor convert without a cosmology. Time needs one;
    without one, time can only be used on its own and is passed through
    unchanged (it orders outputs just as the scale factor would).
    """

    def __init__(self, cosmological: bool) -> None:
        """Create a converter (with or without a cosmology)."""
        self.cosmological = cosmological

    def to_a(self, unit: str, values: ArrayLike) -> np.ndarray:
        """Convert values in a unit to scale factor."""
        values = np.asarray(values, dtype=float)
        if unit == "z":
            return 1.0 / (1.0 + values)
        if unit == "time" and self.cosmological:
            return 1.0 / (1.0 + np.asarray(convert_time_to_redshift(values)))
        return values

    def from_a(self, unit: str, a: ArrayLike) -> np.ndarray:
        """Convert scale factors to a unit."""
        a = np.asarray(a, dtype=float)
        if unit == "z":
            z = 1.0 / a - 1.0
            return np.where(np.abs(z) < 1e-12, 0.0, z)
        if unit == "time" and self.cosmological:
            return np.asarray(convert_redshift_to_time(1.0 / a - 1.0))
        return a

    def convert(self, point: Point, unit: str) -> float:
        """Convert a single value to another unit."""
        if point.unit == unit:
            return point.value
        return float(self.from_a(unit, self.to_a(point.unit, point.value)))


def generate_times(
    unit: str, first: float, delta: float, final: float
) -> np.ndarray:
    """Generate evenly spaced outputs from first to final (inclusive).

    Redshifts count down, everything else counts up. ``first`` and
    ``final`` are in ``unit``, except for log_scale_factor, where they are
    scale factors. No output goes past ``final``.

    Args:
        unit: The unit of the spacing.
        first: The first output.
        delta: The (positive) spacing.
        final: The final output.

    Returns:
        The outputs, in ``unit`` (scale factor for log_scale_factor).
    """
    if delta <= 0:
        raise ValueError(f"the spacing must be positive, got {delta}")
    if unit == "log_scale_factor":
        start, stop = np.log10(first), np.log10(final)
    else:
        start, stop = first, final
    sign = -1.0 if unit == "z" else 1.0
    span = sign * (stop - start)
    if span < 0:
        raise ValueError(
            f"the final output ({final:g}) comes before the first "
            f"({first:g}) in {UNIT_NAMES[unit]}"
        )
    steps = int(np.floor(span / delta + 1e-9))
    values = start + sign * delta * np.arange(steps + 1)
    return 10**values if unit == "log_scale_factor" else values


def _merge(
    snapshots: np.ndarray, snipshots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge outputs (scale factors), sorted, without duplicates.

    Snapshots win over snipshots at the same time.

    Returns:
        The scale factors and a matching array of "Snapshot"/"Snipshot".
    """
    a = np.concatenate([snapshots, snipshots])
    kind = np.array(
        ["Snapshot"] * len(snapshots) + ["Snipshot"] * len(snipshots)
    )
    order = np.argsort(a, kind="stable")
    a, kind = a[order], kind[order]
    keep = np.ones(len(a), dtype=bool)
    last = -1
    for i, value in enumerate(a):
        if last >= 0 and np.isclose(
            value, a[last], rtol=DUPLICATE_RTOL, atol=0
        ):
            keep[i] = False
            if kind[i] == "Snapshot":  # the snapshot's value wins
                a[last], kind[last] = value, kind[i]
        else:
            last = i
    return a[keep], kind[keep]


def generate_output_list(args: dict) -> None:
    """Generate and write a SWIFT output list.

    Args:
        args: The mode's arguments (as a dict).
    """
    out_file = Path(args.get("output_list") or "output_list.txt")
    if out_file.exists() and not args.get("force"):
        raise FileExistsError(
            f"{out_file} already exists, use --force to overwrite."
        )

    params = load_parameters()
    cosmological = not args.get("no_cosmology") and get_cosmology() is not None
    convert = Converter(cosmological)

    first = _given(args, "first_snap", POINT_UNITS)
    final = _given(args, "final_snap", POINT_UNITS)
    delta = _given(args, "delta", DELTA_UNITS)
    snip = _given(args, "snipshot_delta", DELTA_UNITS)
    if first is None:
        raise ValueError("give the first output with --first-snap-*")
    if delta is None:
        raise ValueError("give the snapshot spacing with --delta-*")

    a_begin = a_end = None
    if cosmological:
        cosmo = params["Cosmology"]
        a_end = float(cosmo.get("a_end", 1.0))
        if cosmo.get("a_begin") is not None:
            a_begin = float(cosmo["a_begin"])
    if final is None:
        if a_end is None:
            raise ValueError(
                "give the final output with --final-snap-* (it defaults to "
                "Cosmology:a_end only for a cosmological run)"
            )
        final = Point("scale_factor", a_end)

    list_units = args.get("list_units")
    if list_units is None:
        list_units = {"z": "redshift", "time": "time"}.get(
            delta.unit, "scale-factor"
        )
    unit, header = LIST_UNITS[list_units]
    units = {first.unit, final.unit, delta.unit, unit}
    if snip is not None:
        units.add(snip.unit)
    if not cosmological and "time" in units and len(units) > 1:
        raise ValueError(
            "converting between time and redshift or scale factor needs a "
            "cosmology: give a parameter file with a Cosmology section "
            "(-P/--params), or give everything in time"
        )

    # Check the range against the run's
    a_first = float(convert.to_a(first.unit, first.value))
    a_final = float(convert.to_a(final.unit, final.value))
    if a_begin is not None and a_first < a_begin * (1 - 1e-9):
        raise ValueError(
            f"the first output (a={a_first:.6g}) is before the start of the "
            f"run (Cosmology:a_begin={a_begin:.6g})"
        )
    if a_end is not None and a_final > a_end * (1 + 1e-9):
        raise ValueError(
            f"the final output (a={a_final:.6g}) is after the end of the "
            f"run (Cosmology:a_end={a_end:.6g})"
        )

    def outputs(spacing: Point) -> np.ndarray:
        """Generate outputs natively and return them as scale factors."""
        unit = spacing.unit
        native = "scale_factor" if unit == "log_scale_factor" else unit
        values = generate_times(
            unit,
            convert.convert(first, native),
            spacing.value,
            convert.convert(final, native),
        )
        return convert.to_a(native, values)

    a, kind = _merge(outputs(delta), outputs(snip) if snip else np.array([]))
    if len(a) < 2:
        raise ValueError(
            f"only {len(a)} output between the first and final outputs; "
            "SWIFT needs at least 2 (use a smaller spacing)"
        )

    # Convert once to the list's unit
    values = convert.from_a(unit, a)
    if unit == "time" and cosmological:
        # SWIFT reads a cosmological time list in internal time units
        values = values / internal_time_unit_in_gyr(params)

    mixed = snip is not None and "Snipshot" in kind
    lines = [f"# {header}" + (", Select Output" if mixed else "")]
    for value, label in zip(values, kind):
        lines.append(f"{value:.12g}" + (f", {label}" if mixed else ""))
    out_file.write_text("\n".join(lines) + "\n")

    n_snip = int(np.sum(kind == "Snipshot"))
    print(f"Generated {len(a) - n_snip} snapshots")
    if n_snip:
        print(f"Generated {n_snip} snipshots")
        note(
            "SWIFT needs Snapshots:select_output_on: 1 and a "
            "Snapshots:select_output file defining the Snipshot output "
            "selection to use this list"
        )
    print(f"Output list written to {out_file}.")


def run(args: argparse.Namespace) -> None:
    """Execute the output-times mode."""
    generate_output_list(vars(args))
