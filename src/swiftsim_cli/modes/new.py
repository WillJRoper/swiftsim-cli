"""New mode for creating new SWIFT run directories."""

import argparse
import warnings
from pathlib import Path
from typing import Tuple

import h5py  # type: ignore[import-untyped]
from ruamel.yaml import YAML

from swiftsim_cli.profile import load_swift_profile
from swiftsim_cli.swiftsim_dir import get_swiftsim_dir

# Configure YAML for round-trip comment preservation and consistent formatting
yaml = YAML()
yaml.default_flow_style = False
yaml.indent(mapping=4, sequence=4, offset=2)
yaml.width = 80
yaml.allow_unicode = True


def _kv_pair(arg: str) -> Tuple[str, str]:
    """Parse a KEY=VALUE pair from a string.

    Argparse "type" function: split a KEY=VALUE into a (key, value) tuple,
    or raise an error if the syntax is wrong.
    """
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"invalid parameter override '{arg}'; expected KEY=VALUE"
        )
    key, val = arg.split("=", 1)
    return key, val


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'new' mode."""
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the new SWIFT run.",
        type=Path,
    )
    parser.add_argument(
        "--inic",
        required=True,
        help="Path to the initial conditions HDF5 file.",
        type=Path,
    )

    # Add the ability to override internal parameters
    parser.add_argument(
        "--param",
        metavar="KEY=VALUE",
        type=_kv_pair,
        action="append",
        help=(
            "Override a SWIFT parameter in the form ROOTKEY:KEY=VALUE.  "
            "Can be repeated for multiple overrides."
        ),
        default=[],
    )


def run(args: argparse.Namespace) -> None:
    """Execute the new mode."""
    make_new_run_dir(
        output_dir=args.path,
        inicond_file=args.inic,
        overide_params=dict(args.param),
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
        box_size = hdf["Header"].attrs["BoxSize"]

        # How many particles do we have?
        ngas = hdf["Header"].attrs["NumPart_Total"][:][0]
        ndm = hdf["Header"].attrs["NumPart_Total"][:][1]

        time_begin = None

        # Get the redshift from the initial conditions
        if "Redshift" in hdf["Header"].attrs:
            redshift = hdf["Header"].attrs["Redshift"]
        else:
            redshift = None

        # Derive a_begin from the redshift
        if redshift is not None:
            a_begin = 1 / (1 + redshift)
        else:
            a_begin = None

        # We also might have Time as a beginning point, annoyingly, this could
        # be a bog standard time or a scale factor depending on weather we are
        # workign with or without cosmology
        if "Time" in hdf["Header"].attrs:
            time_begin = hdf["Header"].attrs["Time"]
            a_begin = time_begin

        # If we have a Units group them get the unit system
        units = {}
        if "Units" in hdf:
            for key, value in hdf["Units"].attrs.items():
                units[key] = value
        else:
            # Warn the user that they need to set the units manually
            warnings.warn(
                "No 'Units' group found in the initial conditions file. You "
                "will need to set the units manually in the parameter file.",
                UserWarning,
            )

        # Compute the mean separation of each particle type
        # (assuming a cubic box)
        mean_separation_dm = box_size / (ndm ** (1 / 3))
        mean_separation_gas = (
            (box_size / (ngas ** (1 / 3))) if ngas > 0 else mean_separation_dm
        )  # if we have no gas in ics itll be generated from dm

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
    params["Gravity"]["comoving_DM_softening"] = str(dm_soft)
    params["Gravity"]["max_physical_DM_softening"] = str(max_dm_soft)
    params["Gravity"]["comoving_baryon_softening"] = str(gas_soft)
    params["Gravity"]["max_physical_baryon_softening"] = str(max_gas_soft)
    if redshift is not None:
        params["Cosmology"]["a_begin"] = str(a_begin)
    if time_begin is not None:
        params["Cosmology"]["a_begin"] = str(time_begin)
        params["TimeIntegration"]["time_begin"] = str(time_begin)
    if len(units) > 0:
        if "Unit mass in cgs (U_M)" in units:
            params["InternalUnitSystem"]["UnitMass_in_cgs"] = units[
                "Unit mass in cgs (U_M)"
            ]
        else:
            warnings.warn(
                "No 'Unit mass in cgs (U_M)' found in the initial"
                " conditions file. Using default value of 1.0.",
                UserWarning,
            )
        if "Unit length in cgs (U_L)" in units:
            params["InternalUnitSystem"]["UnitLength_in_cgs"] = units[
                "Unit length in cgs (U_L)"
            ]
        else:
            warnings.warn(
                "No 'Unit length in cgs (U_L)' found in the initial "
                "conditions file. Using default value of 1.0.",
                UserWarning,
            )
        if "Unit time in cgs (U_t)" in units:
            params["InternalUnitSystem"]["UnitTime_in_cgs"] = units[
                "Unit time in cgs (U_t)"
            ]
        else:
            warnings.warn(
                "No 'Unit time in cgs (U_t)' found in the initial "
                "conditions file. Using default value of 1.0.",
                UserWarning,
            )
        if "Unit current in cgs (U_I)" in units:
            params["InternalUnitSystem"]["UnitCurrent_in_cgs"] = units[
                "Unit current in cgs (U_I)"
            ]
        else:
            warnings.warn(
                "No 'Unit current in cgs (U_I)' found in the initial "
                "conditions file. Using default value of 1.0.",
                UserWarning,
            )
        if "Unit temperature in cgs (U_T)" in units:
            params["InternalUnitSystem"]["UnitTemp_in_cgs"] = units[
                "Unit temperature in cgs (U_T)"
            ]
        else:
            warnings.warn(
                "No 'Unit temperature in cgs (U_T)' found in the initial "
                "conditions file. Using default value of 1.0.",
                UserWarning,
            )
        if "Unit velocity in cgs (U_V)" in units:
            params["InternalUnitSystem"]["UnitVelocity_in_cgs"] = units[
                "Unit velocity in cgs (U_V)"
            ]
        else:
            warnings.warn(
                "No 'Unit velocity in cgs (U_V)' found in the initial "
                "conditions file. Using default value of 1.0.",
                UserWarning,
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

        # Set the value in the parameters dictionary
        params[parent][child] = value


def make_new_parameter_file(
    output_dir: Path,
    inicond_file: Path,
    swift_dir: Path | None = None,
    overide_params: dict | None = None,
) -> dict:
    """Create a new parameter file for the SWIFT run.

    Args:
        output_dir: The directory where the new parameter file will be created.
        inicond_file: The initial conditions file to use for the new run.
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.
        overide_params: Optional dictionary of parameters to override in the
            parameter file.
    """
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

    # Write the new parameters to the output directory
    output_param_file = output_dir / "params.yml"
    with output_param_file.open("w") as f:
        yaml.dump(params, f)

    return params


def make_new_run_dir(
    output_dir: Path,
    inicond_file: Path,
    swift_dir: Path | None = None,
    overide_params: dict | None = None,
) -> None:
    """Create a new SWIFT run directory with a parameter file.

    Args:
        output_dir: The directory where the new run will be created.
        inicond_file: The initial conditions file to use for the new run.
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.
        overide_params: Optional dictionary of parameters to override in the
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the new parameters file
    _ = make_new_parameter_file(
        output_dir,
        inicond_file,
        swift_dir,
        overide_params,
    )
