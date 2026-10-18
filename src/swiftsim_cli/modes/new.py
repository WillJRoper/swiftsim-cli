"""New mode for creating new SWIFT run directories."""

import argparse
import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from ruamel.yaml import YAML

from swiftsim_cli.modes.common_args import existing_file
from swiftsim_cli.modes.utils import _get_cubic_box_size
from swiftsim_cli.params import round_trip_yaml
from swiftsim_cli.profile import load_swift_profile
from swiftsim_cli.swiftsim_dir import get_swiftsim_dir
from swiftsim_cli.utilities.messages import warn

yaml = round_trip_yaml()

# Plain loader used to turn --set override strings into YAML scalars
_scalar_yaml = YAML(typ="safe")


def _as_float(value: Any) -> float:
    """Convert an HDF5 attribute (numpy scalar or 1-element array) to float.

    ruamel cannot represent numpy types, so everything written to the
    parameter file must be a plain Python value.
    """
    return float(np.squeeze(value))


def _kv_pair(arg: str) -> Tuple[str, str]:
    """Parse a KEY=VALUE pair from a string.

    Argparse "type" function: split a KEY=VALUE into a (key, value) tuple,
    or raise an error if the syntax is wrong.
    """
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"invalid parameter override '{arg}'; expected ROOT:KEY=VALUE"
        )
    key, val = arg.split("=", 1)
    return key, val


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'new' mode."""
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory to create the new run in.",
    )
    parser.add_argument(
        "--ics",
        required=True,
        type=existing_file,
        metavar="FILE",
        help="Initial conditions HDF5 file.",
    )

    # Add the ability to override internal parameters
    parser.add_argument(
        "--set",
        metavar="ROOT:KEY=VALUE",
        type=_kv_pair,
        action="append",
        help="Override a SWIFT parameter, e.g. --set "
        "TimeIntegration:time_end=1.0; can be repeated.",
        default=[],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing params.yml in the run directory.",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the new mode."""
    make_new_run_dir(
        output_dir=args.out,
        inicond_file=args.ics,
        overide_params=dict(args.set),
        force=args.force,
    )


def derive_params_from_ics(
    inicond_file: Path,
    params: dict,
) -> dict:
    """Derive parameters from the initial conditions file.

    Args:
        inicond_file: The initial conditions file to read.
        params: The parameter dictionary to update.

    Returns:
        dict: The updated parameters dictionary with derived values.
    """
    import h5py

    # Get the config
    config = load_swift_profile()

    # Ensure the initial conditions file exists
    if not inicond_file.exists():
        raise FileNotFoundError(
            f"Initial conditions file does not exist: {inicond_file}"
        )

    # Read the initial conditions file
    with h5py.File(inicond_file, "r") as hdf:
        # Get the box size from the initial conditions
        box_size = _as_float(
            _get_cubic_box_size(hdf["Header"].attrs["BoxSize"])
        )

        # How many particles do we have?
        ngas = int(hdf["Header"].attrs["NumPart_Total"][0])
        ndm = int(hdf["Header"].attrs["NumPart_Total"][1])

        # Get the redshift from the initial conditions
        if "Redshift" in hdf["Header"].attrs:
            redshift = _as_float(hdf["Header"].attrs["Redshift"])
        else:
            redshift = None

        # We also might have Time as a beginning point, annoyingly, this could
        # be a bog standard time or a scale factor depending on whether we are
        # working with or without cosmology
        if "Time" in hdf["Header"].attrs:
            time = _as_float(hdf["Header"].attrs["Time"])
        else:
            time = None

        # If we have a Units group them get the unit system
        units = {}
        if "Units" in hdf:
            for key, value in hdf["Units"].attrs.items():
                units[key] = _as_float(value)
        else:
            # Warn the user that they need to set the units manually
            warn(
                "No 'Units' group found in the initial conditions file. You "
                "will need to set the units manually in the parameter file."
            )

    # Compute the mean separation of each particle type (assuming a cubic
    # box). If we have no gas in the ICs it'll be generated from the DM, and
    # with no DM the DM softening is unused so just mirror the gas.
    if ndm == 0 and ngas == 0:
        raise ValueError(
            f"No gas or dark matter particles found in {inicond_file}; "
            "cannot derive softening lengths."
        )
    mean_separation_dm = box_size / ((ndm if ndm > 0 else ngas) ** (1 / 3))
    mean_separation_gas = (
        (box_size / (ngas ** (1 / 3))) if ngas > 0 else mean_separation_dm
    )

    # Compute the softening lengths and their maximal values
    dm_soft = config.softening_coeff * mean_separation_dm
    gas_soft = config.softening_coeff * mean_separation_gas
    max_dm_soft = dm_soft / (1 + config.softening_pivot_z)
    max_gas_soft = gas_soft / (1 + config.softening_pivot_z)

    print("Setting softening lengths to:")
    print(
        f"  Dark Matter: {dm_soft:.3e} (max @ z={config.softening_pivot_z}:"
        f" {max_dm_soft:.3e})"
    )
    print(
        f"  Baryons:     {gas_soft:.3e} (max @ z={config.softening_pivot_z}:"
        f" {max_gas_soft:.3e})"
    )

    # Update the parameters dictionary with the derived values
    params["Gravity"]["comoving_DM_softening"] = dm_soft
    params["Gravity"]["max_physical_DM_softening"] = max_dm_soft
    params["Gravity"]["comoving_baryon_softening"] = gas_soft
    params["Gravity"]["max_physical_baryon_softening"] = max_gas_soft

    # Set the start of the run. A positive Redshift means cosmological ICs.
    # Otherwise Time is a scale factor only if the template has a Cosmology
    # section and 0 < Time <= 1; anything else is a plain start time.
    # ponytail: heuristic, a non-cosmological IC with 0 < Time <= 1 is
    # misread as a scale factor; fix with --set TimeIntegration:time_begin=....
    if redshift is not None and redshift > 0:
        params["Cosmology"]["a_begin"] = 1 / (1 + redshift)
    elif time is not None:
        if "Cosmology" in params and 0 < time <= 1:
            params["Cosmology"]["a_begin"] = time
        else:
            params["TimeIntegration"]["time_begin"] = time
    if len(units) > 0:
        unit_params = params["InternalUnitSystem"]
        for ic_name, param_name in (
            ("Unit mass in cgs (U_M)", "UnitMass_in_cgs"),
            ("Unit length in cgs (U_L)", "UnitLength_in_cgs"),
            ("Unit current in cgs (U_I)", "UnitCurrent_in_cgs"),
            ("Unit temperature in cgs (U_T)", "UnitTemp_in_cgs"),
        ):
            if ic_name in units:
                unit_params[param_name] = units[ic_name]
            else:
                warn(
                    f"No '{ic_name}' found in the initial conditions file."
                    f" Keeping the template value of {param_name}."
                )

        # SWIFT has no time unit parameter, ICs rarely store a velocity
        # unit, so derive velocity from length / time when needed
        unit_length = units.get("Unit length in cgs (U_L)")
        unit_time = units.get("Unit time in cgs (U_t)")
        if "Unit velocity in cgs (U_V)" in units:
            unit_params["UnitVelocity_in_cgs"] = units[
                "Unit velocity in cgs (U_V)"
            ]
        elif unit_length is not None and unit_time is not None:
            unit_params["UnitVelocity_in_cgs"] = unit_length / unit_time
        else:
            warn(
                "Cannot derive UnitVelocity_in_cgs from the initial "
                "conditions file. Keeping the template value."
            )

    return params


def apply_overrides(params: dict, overide_params: dict | None = None) -> None:
    """Apply overrides to the parameters dictionary.

    Args:
        params: The parameter dictionary to update.
        overide_params: Optional dictionary of parameters to override in the
            parameter file. Keys should be in the format 'PARENTKEY:KEY=VALUE'.
    """
    # Loop over the override parameters
    if overide_params is None:
        return
    for key, value in overide_params.items():
        # Split parent and child keys, if we have that structure, otherwise
        # something has gone wrong
        split_key = key.split(":")
        if len(split_key) > 2:
            raise ValueError(
                f"Invalid parameter key '{key}'. "
                "Keys should be in the format 'parent:child'."
            )
        elif len(split_key) == 2:
            parent, child = key.split(":")
        else:
            raise ValueError(
                f"Invalid parameter key '{key}'. "
                "Keys should be provided in the format: "
                "'PARENTKEY:KEY=VALUE'."
            )

        # If the parent key is not in the parameters, raise an error
        if parent not in params:
            raise ValueError(
                f"Parent key '{parent}' not found in parameters. "
                "Available keys: " + ", ".join(params.keys())
            )

        # Ensure the child key exists in the parent
        if child not in params[parent]:
            raise ValueError(
                f"Key '{child}' not found in parent '{parent}'. "
                "Available keys: " + ", ".join(params[parent].keys())
            )

        # Set the value in the parameters dictionary, parsing it as a YAML
        # scalar so numbers are written as numbers (SWIFT reads them with
        # sscanf) while e.g. file names stay strings
        parsed = _scalar_yaml.load(value)
        params[parent][child] = value if parsed is None else parsed


def make_new_parameter_file(
    output_dir: Path,
    inicond_file: Path,
    swift_dir: Path | None = None,
    overide_params: dict | None = None,
    force: bool = False,
) -> dict:
    """Create a new parameter file for the SWIFT run.

    Args:
        output_dir: The directory where the new parameter file will be created.
        inicond_file: The initial conditions file to use for the new run.
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.
        overide_params: Optional dictionary of parameters to override in the
            parameter file.
        force: Overwrite an existing params.yml if True.

    Raises:
        FileExistsError: If params.yml already exists and force is False.
    """
    # SWIFT runs from inside the run directory, so paths must be absolute
    output_dir = output_dir.resolve()
    inicond_file = inicond_file.resolve()

    # Refuse to clobber an existing parameter file unless asked to
    output_param_file = output_dir / "params.yml"
    if output_param_file.exists() and not force:
        raise FileExistsError(
            f"{output_param_file} already exists, use --force to overwrite."
        )

    # Get the SWIFT directory
    swift_dir = get_swiftsim_dir(swift_dir)

    # Read the example parameter file into a CommentedMap (preserves comments)
    example_param_file = swift_dir / "examples" / "parameter_example.yml"
    with example_param_file.open("r") as f:
        raw = f.read()

    # Convert tabs to spaces to satisfy YAML spec (tabs are not allowed
    # in indentation)
    raw = raw.expandtabs(4)
    params = yaml.load(raw)

    # Update the parameters with the initial conditions file
    params["InitialConditions"]["file_name"] = str(inicond_file)

    # Set various directories
    params["Restarts"]["subdir"] = str(output_dir / "restart")
    params["Snapshots"]["subdir"] = str(output_dir / "snapshots")

    # Derive parameters from the initial conditions file
    params = derive_params_from_ics(inicond_file, params)

    # Apply any overrides provided by the user
    if overide_params is not None:
        apply_overrides(params, overide_params)

    # Write the new parameters atomically so a failed dump never leaves a
    # truncated params.yml behind
    tmp_file = output_param_file.with_name(output_param_file.name + ".tmp")
    try:
        with open(tmp_file, "w") as f:
            yaml.dump(params, f)
        os.replace(tmp_file, output_param_file)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise

    return params


def make_new_run_dir(
    output_dir: Path,
    inicond_file: Path,
    swift_dir: Path | None = None,
    overide_params: dict | None = None,
    force: bool = False,
) -> None:
    """Create a new SWIFT run directory with a parameter file.

    Args:
        output_dir: The directory where the new run will be created.
        inicond_file: The initial conditions file to use for the new run.
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.
        overide_params: Optional dictionary of parameters to override in the
            parameter file.
        force: Overwrite an existing params.yml if True.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the new parameters file
    _ = make_new_parameter_file(
        output_dir,
        inicond_file,
        swift_dir,
        overide_params,
        force,
    )
