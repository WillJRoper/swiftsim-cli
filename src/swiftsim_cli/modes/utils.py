"""Utility mode for small SWIFT helper calculations.

This module also holds the initial-conditions header reader and the
softening calculation shared with ``new``, so ``utils softenings`` reports
exactly what ``new`` writes.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from swiftsim_cli.modes.common_args import (
    add_params_argument,
    existing_file,
    positive_float,
)
from swiftsim_cli.profile import load_swift_profile

# Newton's constant and a megaparsec in cgs (as SWIFT's constants)
_G_CGS = 6.67430e-8
_MPC_CGS = 3.08567758149e24

# (IC Units attribute, SWIFT InternalUnitSystem parameter)
IC_UNIT_NAMES = (
    ("Unit mass in cgs (U_M)", "UnitMass_in_cgs"),
    ("Unit length in cgs (U_L)", "UnitLength_in_cgs"),
    ("Unit current in cgs (U_I)", "UnitCurrent_in_cgs"),
    ("Unit temperature in cgs (U_T)", "UnitTemp_in_cgs"),
)


def _scalar(value: Any) -> float:
    """Return a float from an HDF5 attribute (scalar or 1-element array)."""
    return float(np.ravel(value)[0])


@dataclass
class ICHeader:
    """What swift-cli needs from an initial conditions file's header.

    Attributes:
        path: The IC file.
        box_size: The box side lengths (3 values, in IC length units).
        counts: Total particle count per type (high words included).
        redshift: Header Redshift, if present.
        time: Header Time, if present.
        units: The Units group attributes ({} if the group is missing).
        dimension: The number of spatial dimensions.
        masses: Minimum particle mass per type, read when needed.
    """

    path: Path
    box_size: np.ndarray
    counts: list[int]
    redshift: float | None
    time: float | None
    units: dict[str, float]
    dimension: int = 3
    masses: dict[int, float] = field(default_factory=dict)

    def count(self, part_type: int) -> int:
        """Return the number of particles of a type (0 if absent)."""
        return self.counts[part_type] if part_type < len(self.counts) else 0

    @property
    def is_cosmological(self) -> bool:
        """Whether the ICs look cosmological.

        ponytail: heuristic, a positive Redshift or 0 < Time < 1 (a scale
        factor) means cosmological; pick a template with --template if
        the guess is wrong.
        """
        if self.redshift is not None and self.redshift > 0:
            return True
        return self.time is not None and 0 < self.time < 1

    @property
    def a_begin(self) -> float | None:
        """The starting scale factor of cosmological ICs."""
        if self.redshift is not None and self.redshift > 0:
            return 1 / (1 + self.redshift)
        if self.time is not None and 0 < self.time <= 1:
            return self.time
        return None

    @property
    def is_zoom(self) -> bool:
        """Whether there are background (PartType2) particles."""
        return self.count(2) > 0

    def particle_mass(self, part_type: int) -> float:
        """Return the smallest particle mass of a type (IC units)."""
        if part_type not in self.masses:
            import h5py

            with h5py.File(self.path, "r") as hdf:
                table = hdf["Header"].attrs.get("MassTable")
                mass = 0.0
                if table is not None and len(np.ravel(table)) > part_type:
                    mass = float(np.ravel(table)[part_type])
                if mass <= 0:
                    name = f"PartType{part_type}/Masses"
                    if name not in hdf:
                        raise ValueError(
                            f"{self.path}: no {name} or MassTable entry"
                        )
                    mass = float(np.min(hdf[name][:]))
            self.masses[part_type] = mass
        return self.masses[part_type]


def read_ic_header(path: Path) -> ICHeader:
    """Read the header of a SWIFT/Gadget HDF5 initial conditions file.

    Scalar and 1-element-array attributes are both accepted, as are a
    missing Units group, Redshift or Time.

    Args:
        path: The IC file.

    Returns:
        The header.

    Raises:
        ValueError: If the file is not HDF5 or has no usable Header.
    """
    import h5py

    try:
        hdf = h5py.File(path, "r")
    except OSError as e:
        raise ValueError(f"{path}: not a readable HDF5 file ({e})") from e
    with hdf:
        if "Header" not in hdf:
            raise ValueError(f"{path}: no Header group")
        attrs = hdf["Header"].attrs
        for name in ("BoxSize", "NumPart_Total"):
            if name not in attrs:
                raise ValueError(f"{path}: the Header has no {name}")
        box = np.atleast_1d(np.asarray(attrs["BoxSize"], dtype=float))
        if box.size == 1:
            box = np.repeat(box, 3)
        low = np.atleast_1d(np.asarray(attrs["NumPart_Total"], dtype=np.int64))
        high = np.atleast_1d(
            np.asarray(
                attrs.get("NumPart_Total_HighWord", np.zeros_like(low)),
                dtype=np.int64,
            )
        )
        high = np.resize(high, low.shape) if high.size else 0 * low
        counts = [int(n) for n in low + (high << 32)]
        units = {}
        if "Units" in hdf:
            units = {k: _scalar(v) for k, v in hdf["Units"].attrs.items()}
        return ICHeader(
            path=path,
            box_size=box,
            counts=counts,
            redshift=_scalar(attrs["Redshift"])
            if "Redshift" in attrs
            else None,
            time=_scalar(attrs["Time"]) if "Time" in attrs else None,
            units=units,
            dimension=int(_scalar(attrs.get("Dimension", 3))),
        )


def ic_unit_parameters(header: ICHeader) -> dict[str, float]:
    """Return the InternalUnitSystem parameters given by the ICs' Units.

    SWIFT has no time unit parameter, so the velocity unit is taken from
    the ICs or derived as U_L / U_t.

    Args:
        header: The IC header.

    Returns:
        The unit parameters found (empty without a Units group).
    """
    params = {
        name: header.units[ic_name]
        for ic_name, name in IC_UNIT_NAMES
        if ic_name in header.units
    }
    length = header.units.get("Unit length in cgs (U_L)")
    time = header.units.get("Unit time in cgs (U_t)")
    if "Unit velocity in cgs (U_V)" in header.units:
        params["UnitVelocity_in_cgs"] = header.units[
            "Unit velocity in cgs (U_V)"
        ]
    elif length is not None and time is not None:
        params["UnitVelocity_in_cgs"] = length / time
    return params


def _round(value: float) -> float:
    """Round to 6 significant figures for a readable parameter file."""
    return float(f"{value:.6g}")


def compute_softenings(
    header: ICHeader,
    params: Mapping[str, Any],
    softening_coeff: float,
    pivot_redshift: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Compute SWIFT's softening parameters for some ICs.

    The softening is ``softening_coeff`` times the mean inter-particle
    separation of each species (SWIFT has DM, baryon and neutrino
    softenings). In cosmological runs that is the comoving softening and
    the maximal physical softening is it divided by (1 + pivot redshift).
    In non-cosmological runs SWIFT only reads the maximal physical
    softening, which is then coefficient x separation.

    The separation is (box volume / N)^(1/dimension), except for zoom ICs
    (with background PartType2 particles), where the high-resolution DM
    separation follows from its particle mass,
    (m_DM / (Omega rho_crit))^(1/3), and the background softening is set
    through Gravity:softening_ratio_background. Omega is Omega_cdm when
    there are baryons and Omega_m for DM-only ICs.

    Lengths are in the IC units, converted to h-free units when
    InitialConditions:cleanup_h_factors is set.

    Args:
        header: The IC header.
        params: The run's parameters (Cosmology, InternalUnitSystem and
            InitialConditions are read).
        softening_coeff: Softening in units of the mean separation.
        pivot_redshift: Redshift at which the softening becomes physical.

    Returns:
        A row per species (for reports) and the Gravity parameters.

    Raises:
        ValueError: If the ICs have no gravitating particles.
    """
    cosmo = params.get("Cosmology") or {}
    ics = params.get("InitialConditions") or {}
    h_scale = (
        float(cosmo["h"]) if cosmo and ics.get("cleanup_h_factors") else 1.0
    )
    dim = header.dimension
    volume = float(np.prod(header.box_size[:dim])) / h_scale**dim

    n_dm = header.count(1)
    n_baryon = header.count(0) + header.count(4) + header.count(5)
    if ics.get("generate_gas_in_ics") and header.count(0) == 0:
        n_baryon = n_dm  # SWIFT splits every DM particle
    species = [("DM", "Dark matter", n_dm), ("baryon", "Baryons", n_baryon)]
    species.append(("nu", "Neutrinos", header.count(6)))
    if not any(n for _, _, n in species):
        raise ValueError(
            f"{header.path}: no dark matter, gas, star or neutrino particles"
        )

    zoom_sep = None
    if cosmo and header.is_zoom and n_dm:
        units = params.get("InternalUnitSystem") or {}
        h = float(cosmo["h"])
        hubble = 100 * h * 1e5 / _MPC_CGS
        rho_crit = 3 * hubble**2 / (8 * np.pi * _G_CGS)
        rho_crit *= float(units.get("UnitLength_in_cgs", _MPC_CGS)) ** 3
        rho_crit /= float(units.get("UnitMass_in_cgs", 1.98841e43))
        omega = float(cosmo["Omega_cdm"])
        if not n_baryon:
            omega += float(cosmo["Omega_b"])
        mass = header.particle_mass(1) / h_scale
        zoom_sep = (mass / (omega * rho_crit)) ** (1 / 3)

    rows = []
    gravity: dict[str, float] = {}
    for key, label, count in species:
        if count <= 0:
            continue
        separation = (
            zoom_sep
            if zoom_sep is not None and key != "nu"
            else (volume / count) ** (1 / dim)
        )
        softening = softening_coeff * separation
        maximal = softening / (1 + pivot_redshift) if cosmo else softening
        if cosmo:
            gravity[f"comoving_{key}_softening"] = _round(softening)
        gravity[f"max_physical_{key}_softening"] = _round(maximal)
        rows.append(
            {
                "species": key,
                "label": label,
                "count": count,
                "mean_separation": separation,
                "comoving_softening": softening if cosmo else None,
                "maximal_softening": maximal,
            }
        )
    if zoom_sep is not None:
        gravity["softening_ratio_background"] = _round(softening_coeff)
    return rows, gravity


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'utils' mode."""
    subparsers = parser.add_subparsers(
        dest="utils_type",
        metavar="UTILITY",
        help="Utility calculation to run.",
        required=True,
    )

    softenings = subparsers.add_parser(
        "softenings",
        help="Compute the softenings 'new' would use for an IC file.",
    )
    softenings.set_defaults(func=run_softenings)
    softenings.add_argument(
        "--ics",
        type=existing_file,
        required=True,
        metavar="FILE",
        help="SWIFT initial conditions file.",
    )
    add_params_argument(softenings)
    softenings.add_argument(
        "--softening-coeff",
        type=positive_float,
        default=None,
        metavar="COEFF",
        help="Softening coefficient in units of the mean separation "
        "(default: from the profile).",
    )
    softenings.add_argument(
        "--pivot-redshift",
        type=float,
        default=None,
        metavar="Z",
        help="Pivot redshift for the maximal physical softening "
        "(default: from the profile).",
    )


def format_softening_report(
    header: ICHeader,
    param_source: str,
    params: Mapping[str, Any],
    softening_coeff: float,
    pivot_redshift: float,
    rows: list[dict[str, Any]],
    gravity: Mapping[str, float],
) -> str:
    """Format a compact human-readable softening report."""
    cosmo = params.get("Cosmology") or {}
    a_begin = cosmo.get("a_begin")
    length_cgs = (params.get("InternalUnitSystem") or {}).get(
        "UnitLength_in_cgs"
    )
    unit = f"U_L = {float(length_cgs):.6g} cm" if length_cgs else "U_L"
    lines = [
        f"Softening report for {header.path}",
        f"Parameters: {param_source}",
        f"Box size: {header.box_size[0]:.6g} (IC length units)",
        (
            f"Cosmological: a_begin={float(a_begin):.6g}, "
            f"z_begin={1 / float(a_begin) - 1:.6g}"
            if cosmo and a_begin
            else "Non-cosmological"
        ),
        (
            f"softening_coeff={softening_coeff:.6g}"
            + (f", pivot_redshift={pivot_redshift:.6g}" if cosmo else "")
        ),
        f"Lengths in internal units ({unit})",
        "",
        f"{'Species':<12} {'N':>12} {'Mean sep':>12} {'Comoving':>12} "
        f"{'Max physical':>13}",
    ]
    for row in rows:
        comoving = row["comoving_softening"]
        lines.append(
            f"{row['label']:<12} {row['count']:>12d} "
            f"{row['mean_separation']:>12.4e} "
            + (
                f"{comoving:>12.4e} "
                if comoving is not None
                else f"{'-':>12} "
            )
            + f"{row['maximal_softening']:>13.4e}"
        )
    if "softening_ratio_background" in gravity:
        lines.append(
            "Background (PartType2): softening_ratio_background="
            f"{gravity['softening_ratio_background']:.6g}"
        )
    return "\n".join(lines)


def run_softenings(args: argparse.Namespace) -> None:
    """Run the softening utility."""
    # Imported here: new imports this module
    from swiftsim_cli.modes.new import apply_ic_settings, base_parameters

    profile = load_swift_profile()
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

    header = read_ic_header(args.ics)
    params, source = base_parameters(header, params_file=args.params)
    apply_ic_settings(params, header, quiet=True)
    rows, gravity = compute_softenings(
        header, params, softening_coeff, pivot_redshift
    )
    print(
        format_softening_report(
            header,
            source,
            params,
            softening_coeff,
            pivot_redshift,
            rows,
            gravity,
        )
    )


def run(args: argparse.Namespace) -> None:
    """Run the requested utility command."""
    args.func(args)
