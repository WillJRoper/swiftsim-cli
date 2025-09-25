"""Output-times mode for generating output time lists."""

import argparse
from pathlib import Path

import numpy as np

from swiftsim_cli.cosmology import (
    convert_redshift_to_scale_factor,
    convert_redshift_to_time,
    convert_scale_factor_to_redshift,
    convert_time_to_redshift,
    get_cosmology,
)
from swiftsim_cli.params import load_parameters


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'output-times' mode."""
    # We always need the output file
    parser.add_argument(
        "--out",
        "-o",
        default="output_list.txt",
        help="Output file for the list of times.",
    )

    # We can optionally provide a parameter file, this will be used to
    # get cosmology parameters and other settings.
    parser.add_argument(
        "-p",
        "--params",
        type=Path,
        help="Path to a parameter file.",
        default=None,
    )

    # We will always need one definition of the first snapshot but this can be
    # defined in terms of redshift, time, or scale factor.
    parser.add_argument(
        "--first-snap-z",
        "-fz",
        type=float,
        default=None,
        help="Redshift of the first snapshot to include.",
    )
    parser.add_argument(
        "--first-snap-time",
        "-ft",
        type=float,
        default=None,
        help="Time of the first snapshot in internal units.",
    )
    parser.add_argument(
        "--first-snap-scale-factor",
        "-fa",
        type=float,
        default=None,
        help="Scale factor of the first snapshot to include.",
    )

    # Similarly with the delta between snapshots, we can define this in terms
    # of redshift, time, scale factor, or logarithmic scale factor.
    parser.add_argument(
        "--delta-z",
        "-dz",
        type=float,
        default=None,
        help="Redshift interval between snapshots.",
    )
    parser.add_argument(
        "--delta-time",
        "-dt",
        type=float,
        default=None,
        help="Time interval between snapshots in internal units.",
    )
    parser.add_argument(
        "--delta-scale-factor",
        "-da",
        type=float,
        default=None,
        help="Scale factor interval between snapshots.",
    )
    parser.add_argument(
        "--delta-log-scale-factor",
        "-dla",
        type=float,
        default=None,
        help="Logarithmic scale factor interval between snapshots.",
    )

    # If we want snipshots, in between we just define a smaller delta for these
    # (again, in terms of redshift, time, scale factor, or
    # logarithmic scale factor).
    parser.add_argument(
        "--snipshot-delta-z",
        "-sdz",
        type=float,
        default=None,
        help="Redshift interval between snipshots.",
    )
    parser.add_argument(
        "--snipshot-delta-time",
        "-sdt",
        type=float,
        default=None,
        help="Time interval between snipshots in internal units.",
    )
    parser.add_argument(
        "--snipshot-delta-scale-factor",
        "-sda",
        type=float,
        default=None,
        help="Scale factor interval between snipshots.",
    )
    parser.add_argument(
        "--snipshot-delta-log-scale-factor",
        "-sdla",
        type=float,
        default=None,
        help="Logarithmic scale factor interval between snipshots.",
    )

    # We will also need to the final snapshot, for redshift and scale factor
    # this has a well defined default, for time it does not.
    parser.add_argument(
        "--final-snap-z",
        "-fzf",
        type=float,
        default=0.0,
        help="Redshift of the final snapshot to include (default: 0.0).",
    )
    parser.add_argument(
        "--final-snap-time",
        "-ftf",
        type=float,
        default=None,
        help="Time of the final snapshot in internal units (default: None).",
    )
    parser.add_argument(
        "--final-snap-scale-factor",
        "-faf",
        type=float,
        default=1.0,
        help="Scale factor of the final snapshot to include (default: 1.0).",
    )


def _get_out_list_z(
    first_snap: float, delta: float, final_snap: float
) -> np.ndarray:
    """Generate an output list in redshift."""
    if delta > 0:
        delta = -delta
    return np.arange(first_snap, final_snap + delta, delta)


def _get_out_list_time(
    first_snap: float, delta: float, final_snap: float
) -> np.ndarray:
    """Generate an output list in time."""
    if delta < 0:
        delta = -delta
    return np.arange(first_snap, final_snap + delta, delta)


def _get_out_list_scale_factor(
    first_snap: float, delta: float, final_snap: float
) -> np.ndarray:
    """Generate an output list in scale factor."""
    if delta < 0:
        delta = -delta
    return np.arange(first_snap, final_snap + delta, delta)


def _get_out_list_log_scale_factor(
    first_snap: float, delta: float, final_snap: float
) -> np.ndarray:
    """Generate an output list in logarithmic scale factor."""
    if delta < 0:
        delta = -delta
    return 10 ** np.arange(
        np.log10(first_snap), np.log10(final_snap) + delta, delta
    )


def write_output_list(
    out_file: str,
    snapshot_times: np.ndarray,
    snipshot_times: np.ndarray,
    doing_z: bool = False,
    doing_time: bool = False,
    doing_scale_factor: bool = False,
) -> None:
    """Write the output list to a file.

    This function will clean up the snapshot and snipshot times, sort them,
    and write them to the specified output file in a two-column format.

    Args:
        out_file: The name of the output file.
        snapshot_times: An array of times for the snapshots.
        snipshot_times: An array of times for the snipshots.
        doing_z: Whether the times are in redshift.
        doing_time: Whether the times are in time units.
        doing_scale_factor: Whether the times are in scale factor units.
    """
    # No point in doubling up on snapshots and snipshots, remove any snipshots
    # that are already in the snapshot list.
    snipshot_times = snipshot_times[~np.isin(snipshot_times, snapshot_times)]

    # Define the first line of the output text file (with two columns)
    output_lines = []
    if doing_z:
        output_lines.append("# Redshift, Select Output")
    elif doing_time:
        output_lines.append("# Time, Select Output")
    elif doing_scale_factor:
        output_lines.append("# Scale Factor, Select Output")
    else:
        raise ValueError(
            "Shouldn't be able to get here, something went wrong with "
            "the input parameters and all the consistency checks."
        )

    # Report how many snapshots and snipshots we have
    print(f"Generated {len(snapshot_times)} snapshots")
    if len(snipshot_times) > 0:
        print(f"Generated {len(snipshot_times)} snipshots")
    print(
        "In total, this will generate "
        f"{len(snapshot_times) + len(snipshot_times)} outputs."
    )

    # Create the select output column
    select_output_snap = ["Snapshot"] * len(snapshot_times)
    select_output_snip = ["Snipshot"] * len(snipshot_times)

    # Combine the times and select output into lines
    times = np.concatenate((snapshot_times, snipshot_times))
    select_output = np.concatenate((select_output_snap, select_output_snip))

    # Sort the times and select output together (here we need to treat redshift
    # as descending and time/scale factor as ascending)
    if doing_z:
        sorted_indices = np.argsort(times)[::-1]
    else:
        sorted_indices = np.argsort(times)
    sorted_times = times[sorted_indices]
    sorted_select_output = select_output[sorted_indices]

    # Create the output lines
    for time, select in zip(sorted_times, sorted_select_output):
        output_lines.append(f"{time}, {select}")

    # Write the output to the file
    with open(out_file, "w") as f:
        f.write("\n".join(output_lines) + "\n")

    print(f"Output list written to {out_file}.")


def _generate_output_list_no_cosmo(args: dict) -> None:
    """Generate an output list file containing times for each snapshot.

    This is the version used when the user has not provided a parameter file
    and this cosmology is not available. With no cosmology, we can't convert
    so everything must be provided consistently and the end must be given.

    Args:
        args: Command-line arguments containing the configuration for the
                output list generation.
    """
    # Unpack the arguments
    out_file = args.get("out", "output_list.txt")
    first_snap_z = args.get("first_snap_z", None)
    first_snap_time = args.get("first_snap_time", None)
    first_snap_scale_factor = args.get("first_snap_scale_factor", None)
    delta_z = args.get("delta_z", None)
    delta_time = args.get("delta_time", None)
    delta_scale_factor = args.get("delta_scale_factor", None)
    delta_log_scale_factor = args.get("delta_log_scale_factor", None)
    snip_delta_z = args.get("snipshot_delta_z", None)
    snip_delta_time = args.get("snipshot_delta_time", None)
    snip_delta_scale_factor = args.get("snipshot_delta_scale_factor", None)
    snip_delta_log_scale_factor = args.get(
        "snipshot_delta_log_scale_factor", None
    )
    final_snap_z = args.get("final_snap_z", None)
    final_snap_time = args.get("final_snap_time", None)
    final_snap_scale_factor = args.get("final_snap_scale_factor", None)

    # Are we doing redshift, time, or scale factor?
    doing_z = first_snap_z is not None
    doing_time = first_snap_time is not None
    doing_scale_factor = first_snap_scale_factor is not None
    # Were we given a start?
    if not (doing_z or doing_time or doing_scale_factor):
        raise ValueError(
            "You must specify the first snapshot in terms of redshift, "
            "time, or scale factor."
        )

    # Were we given a delta?
    if not (
        delta_z or delta_time or delta_scale_factor or delta_log_scale_factor
    ):
        raise ValueError(
            "You must specify the delta between snapshots in terms of "
            "redshift, time, scale factor, or logarithmic scale factor."
        )

    # Do we have everything we need?
    if doing_z and not delta_z:
        raise ValueError(
            "If you specify the first snapshot redshift, you must also "
            "specify the delta in terms of redshift."
        )
    if doing_time and not delta_time:
        raise ValueError(
            "If you specify the first snapshot time, you must also "
            "specify the delta in terms of time."
        )
    if (
        doing_scale_factor
        and delta_scale_factor is None
        and delta_log_scale_factor is None
    ):
        raise ValueError(
            "If you specify the first snapshot scale factor, you must also "
            "specify the delta in terms of scale factor or logarithmic"
            " scale factor."
        )

    # Collapse the variables into a simpler set now we know we have what
    # we need.
    first_snap = first_snap_z or first_snap_time or first_snap_scale_factor
    delta = (
        delta_z or delta_time or delta_scale_factor or delta_log_scale_factor
    )
    snip_delta = (
        snip_delta_z
        or snip_delta_time
        or snip_delta_scale_factor
        or snip_delta_log_scale_factor
    )

    # Make sure we use the right final snapshot time, this has defaults so we
    # need to be more careful
    final_snap = None
    if doing_z:
        final_snap = final_snap_z
    elif doing_time:
        final_snap = final_snap_time
    elif doing_scale_factor:
        final_snap = final_snap_scale_factor

    # Ensure we have a meaningful final snapshot
    if final_snap is None:
        raise ValueError(
            "You must specify the final snapshot time, redshift, or scale "
            "factor."
        )

    # Is our delta logarithmic? This will require special handling.
    doing_log_scale_factor = delta_log_scale_factor is not None
    doing_snip_log_scale_factor = snip_delta_log_scale_factor is not None

    # Do we have snipshots?
    has_snipshots = snip_delta is not None

    # Make sure our snipshot delta is the same type as our delta if we have it.
    if has_snipshots:
        if doing_z and snip_delta_z is None:
            raise ValueError(
                "Snipshot delta must be given in terms of redshift."
            )
        if doing_time and snip_delta_time is None:
            raise ValueError("Snipshot delta must be given in terms of time.")
        if (
            doing_scale_factor
            and snip_delta_scale_factor is None
            and snip_delta_log_scale_factor is None
        ):
            raise ValueError(
                "Snipshot delta must be given in terms of scale factor or "
                "logarithmic scale factor."
            )

    # OK, we seem to have what we need. Lets tell the user what we have
    print("Generating output list with the following parameters:")
    print(f"  First snapshot: {first_snap} ", end="")
    if doing_z:
        print(" (Redshift)")
    elif doing_time:
        print(" (Time in internal units)")
    elif doing_scale_factor:
        print(" (Scale factor)")
    print(f"  Delta: {delta} ", end="")
    if doing_z:
        print(" (Redshift)")
    elif doing_time:
        print(" (Time in internal units)")
    elif doing_scale_factor and not doing_log_scale_factor:
        print(" (Scale factor)")
    elif doing_scale_factor and doing_log_scale_factor:
        print(" (Logarithmic scale factor)")
    if has_snipshots:
        print(f"  Snipshot delta: {snip_delta} ", end="")
        if doing_z:
            print(" (Redshift)")
        elif doing_time:
            print(" (Time in internal units)")
        elif doing_scale_factor and not doing_snip_log_scale_factor:
            print(" (Scale factor)")
        elif doing_scale_factor and doing_snip_log_scale_factor:
            print(" (Logarithmic scale factor)")
    print(f"  Final snapshot: {final_snap} ", end="")
    if doing_z:
        print(" (Redshift)")
    elif doing_time:
        print(" (Time in internal units)")
    elif doing_scale_factor:
        print(" (Scale factor)")

    # Get the output list of times for the snapshots
    if doing_z:
        snapshot_times = _get_out_list_z(first_snap, delta, final_snap)
    elif doing_time:
        snapshot_times = _get_out_list_time(first_snap, delta, final_snap)
    elif doing_scale_factor and not doing_log_scale_factor:
        snapshot_times = _get_out_list_scale_factor(
            first_snap, delta, final_snap
        )
    elif doing_scale_factor and doing_log_scale_factor:
        snapshot_times = _get_out_list_log_scale_factor(
            first_snap, delta, final_snap
        )
    else:
        raise ValueError(
            "You must specify the first snapshot, final snapshot, and delta"
            " in terms of redshift, time, scale factor, or logarithmic "
            "scale factor."
        )

    # If we are getting them, get the snipshot times
    if has_snipshots:
        if doing_z:
            snipshot_times = _get_out_list_z(
                first_snap, snip_delta, final_snap
            )
        elif doing_time:
            snipshot_times = _get_out_list_time(
                first_snap, snip_delta, final_snap
            )
        elif doing_scale_factor and not doing_snip_log_scale_factor:
            snipshot_times = _get_out_list_scale_factor(
                first_snap, snip_delta, final_snap
            )
        elif doing_scale_factor and doing_snip_log_scale_factor:
            snipshot_times = _get_out_list_log_scale_factor(
                first_snap, snip_delta, final_snap
            )
        else:
            raise ValueError(
                "You must specify the first snapshot, final snapshot, and "
                "snipshot delta in terms of redshift, time, scale factor, or "
                "logarithmic scale factor."
            )
    else:
        snipshot_times = np.array([])

    # Write the output list to the file
    write_output_list(
        out_file,
        snapshot_times,
        snipshot_times,
        doing_z=doing_z,
        doing_time=doing_time,
        doing_scale_factor=doing_scale_factor,
    )


def unify_snapshot_times(
    first_snap_z: float | None = None,
    first_snap_time: float | None = None,
    first_snap_scale_factor: float | None = None,
    final_snap_z: float | None = None,
    final_snap_time: float | None = None,
    final_snap_scale_factor: float | None = None,
    doing_z: bool = False,
    doing_time: bool = False,
    doing_scale_factor: bool = False,
    doing_log_scale_factor: bool = False,
) -> tuple[float, float]:
    """Convert the start and finish snapshots to a common type.

    This function will convert the start and finish snapshots to the same type
    (redshift, time, or scale factor) based on the provided parameters. It will
    raise an error if the parameters are inconsistent or if the first snapshot
    is not specified in any of the supported formats.

    Args:
        first_snap_z: The redshift of the first snapshot.
        first_snap_time: The time of the first snapshot in internal units.
        first_snap_scale_factor: The scale factor of the first snapshot.
        final_snap_z: The redshift of the final snapshot.
        final_snap_time: The time of the final snapshot in internal units.
        final_snap_scale_factor: The scale factor of the final snapshot.
        doing_z: Whether we are working in redshift.
        doing_time: Whether we are working in time.
        doing_scale_factor: Whether we are working in scale factor.
        doing_log_scale_factor: Whether we are working in logarithmic scale
            factor.

    Returns:
        A tuple containing the first and final snapshots in the common type.

    Raises:
        ValueError: If the parameters are inconsistent or if the first snapshot
                    is not specified in any of the supported formats.
    """
    # Convert start and finishes to appropriate types
    if doing_z:
        # Convert the first snapshot to redshift
        first_snap = first_snap_z
        if first_snap is None and first_snap_time is not None:
            first_snap = convert_time_to_redshift(first_snap_time)
        elif first_snap is None and first_snap_scale_factor is not None:
            first_snap = convert_scale_factor_to_redshift(
                first_snap_scale_factor
            )
        else:
            raise ValueError(
                "You must specify the first snapshot in terms of redshift, "
                "time, or scale factor."
            )

        # Convert the final snapshot to redshift
        final_snap = final_snap_z
        if final_snap is None and final_snap_time is not None:
            final_snap = convert_time_to_redshift(final_snap_time)
        elif final_snap is None and final_snap_scale_factor is not None:
            final_snap = convert_scale_factor_to_redshift(
                final_snap_scale_factor
            )
        else:
            raise ValueError(
                "You must specify the final snapshot in terms of redshift, "
                "time, or scale factor."
            )
    elif doing_time:
        # Convert the first snapshot to time
        first_snap = first_snap_time
        if first_snap is None and first_snap_z is not None:
            first_snap = convert_redshift_to_time(first_snap_z)
        elif first_snap is None and first_snap_scale_factor is not None:
            first_snap = convert_redshift_to_time(
                convert_scale_factor_to_redshift(first_snap_scale_factor)
            )
        else:
            raise ValueError(
                "You must specify the first snapshot in terms of redshift, "
                "time, or scale factor."
            )

        # Convert the final snapshot to time
        final_snap = final_snap_time
        if final_snap is None and final_snap_z is not None:
            final_snap = convert_redshift_to_time(final_snap_z)
        elif final_snap is None and final_snap_scale_factor is not None:
            final_snap = convert_redshift_to_time(
                convert_scale_factor_to_redshift(final_snap_scale_factor)
            )
        else:
            raise ValueError(
                "You must specify the final snapshot in terms of redshift, "
                "time, or scale factor."
            )
    elif doing_scale_factor:
        # Convert the first snapshot to scale factor
        first_snap = first_snap_scale_factor
        if first_snap is None and first_snap_z is not None:
            first_snap = convert_redshift_to_scale_factor(first_snap_z)
        elif first_snap is None and first_snap_time is not None:
            first_snap = convert_redshift_to_scale_factor(
                convert_time_to_redshift(first_snap_time)
            )
        else:
            raise ValueError(
                "You must specify the first snapshot in terms of redshift, "
                "time, or scale factor."
            )

        # Convert the final snapshot to scale factor
        final_snap = final_snap_scale_factor
        if final_snap is None and final_snap_z is not None:
            final_snap = convert_redshift_to_scale_factor(final_snap_z)
        elif final_snap is None and final_snap_time is not None:
            final_snap = convert_redshift_to_scale_factor(
                convert_time_to_redshift(final_snap_time)
            )
        else:
            raise ValueError(
                "You must specify the final snapshot in terms of redshift, "
                "time, or scale factor."
            )
    elif doing_log_scale_factor:
        # Convert the first snapshot to log10 scale factor
        first_snap = np.log10(first_snap_scale_factor)
        if first_snap is None and first_snap_z is not None:
            first_snap = np.log10(
                convert_redshift_to_scale_factor(first_snap_z)
            )
        elif first_snap is None and first_snap_time is not None:
            first_snap = np.log10(
                convert_redshift_to_scale_factor(
                    convert_time_to_redshift(first_snap_time)
                )
            )
        else:
            raise ValueError(
                "You must specify the first snapshot in terms of redshift, "
                "time, or scale factor."
            )

        # Convert the final snapshot to log10 scale factor
        final_snap = np.log10(final_snap_scale_factor)
        if final_snap is None and final_snap_z is not None:
            final_snap = np.log10(
                convert_redshift_to_scale_factor(final_snap_z)
            )
        elif final_snap is None and final_snap_time is not None:
            final_snap = np.log10(
                convert_redshift_to_scale_factor(
                    convert_time_to_redshift(final_snap_time)
                )
            )
        else:
            raise ValueError(
                "You must specify the final snapshot in terms of redshift, "
                "time, or scale factor."
            )
    else:
        raise ValueError(
            "Found no valid snapshot type. You must specify the first "
            "snapshot in terms of redshift, time, or scale factor."
        )

    return first_snap, final_snap


def _generate_output_list_with_cosmo(args: dict, cosmo) -> None:
    """Generate an output list file containing times for each snapshot.

    Args:
        args: Command-line arguments containing the configuration for the
              output list generation.
        cosmo: Cosmology object to use for conversions.
    """
    # Get the parameter file (if we have a cosmology we already know we have
    # one)
    params = load_parameters()

    # Unpack the arguments
    out_file = args.get("out", "output_list.txt")
    first_snap_z = args.get("first_snap_z", None)
    first_snap_time = args.get("first_snap_time", None)
    first_snap_scale_factor = args.get("first_snap_scale_factor", None)
    delta_z = args.get("delta_z", None)
    delta_time = args.get("delta_time", None)
    delta_scale_factor = args.get("delta_scale_factor", None)
    delta_log_scale_factor = args.get("delta_log_scale_factor", None)
    snip_delta_z = args.get("snipshot_delta_z", None)
    snip_delta_time = args.get("snipshot_delta_time", None)
    snip_delta_scale_factor = args.get("snipshot_delta_scale_factor", None)
    snip_delta_log_scale_factor = args.get(
        "snipshot_delta_log_scale_factor", None
    )
    final_snap_z = args.get("final_snap_z", None)
    final_snap_time = args.get("final_snap_time", None)
    final_snap_scale_factor = args.get("final_snap_scale_factor", None)

    # Will we do snipshots?
    has_snipshots = (
        snip_delta_z is not None
        or snip_delta_time is not None
        or snip_delta_scale_factor is not None
        or snip_delta_log_scale_factor is not None
    )

    # Compute the end if we need to
    if (
        final_snap_z is None
        and final_snap_time is None
        and final_snap_scale_factor is None
    ):
        final_snap_scale_factor = params["Cosmology"]["a_end"]

    # When we have a cosmology, the delta defines what quantity we work in
    doing_z = delta_z is not None
    doing_time = delta_time is not None
    doing_scale_factor = delta_scale_factor is not None
    doing_log_scale_factor = delta_log_scale_factor is not None

    # And same consideration for snipshots?
    snip_doing_z = snip_delta_z is not None
    snip_doing_time = snip_delta_time is not None
    snip_doing_scale_factor = snip_delta_scale_factor is not None
    snip_doing_log_scale_factor = snip_delta_log_scale_factor is not None

    # Convert the first and final snapshots to the appropriate type
    first_snap, final_snap = unify_snapshot_times(
        first_snap_z=first_snap_z,
        first_snap_time=first_snap_time,
        first_snap_scale_factor=first_snap_scale_factor,
        final_snap_z=final_snap_z,
        final_snap_time=final_snap_time,
        final_snap_scale_factor=final_snap_scale_factor,
        doing_z=doing_z,
        doing_time=doing_time,
        doing_scale_factor=doing_scale_factor,
        doing_log_scale_factor=doing_log_scale_factor,
    )
    snip_first_snap, snip_final_snap = unify_snapshot_times(
        first_snap_z=first_snap_z,
        first_snap_time=first_snap_time,
        first_snap_scale_factor=first_snap_scale_factor,
        final_snap_z=final_snap_z,
        final_snap_time=final_snap_time,
        final_snap_scale_factor=final_snap_scale_factor,
        doing_z=snip_doing_z,
        doing_time=snip_doing_time,
        doing_scale_factor=snip_doing_scale_factor,
        doing_log_scale_factor=snip_doing_log_scale_factor,
    )

    # Check that this first snap is after a_begin in the parameter file
    a_begin = params["Cosmology"]["a_begin"]
    if doing_z and convert_redshift_to_scale_factor(first_snap) < a_begin:
        z_begin = convert_scale_factor_to_redshift(a_begin)
        raise ValueError(
            f"The first snapshot redshift ({first_snap}) is before the "
            f"beginning of the simulation ({z_begin})."
        )
    elif (
        doing_time
        and convert_redshift_to_scale_factor(
            convert_time_to_redshift(first_snap)
        )
        < a_begin
    ):
        t_begin = convert_redshift_to_time(
            convert_scale_factor_to_redshift(a_begin)
        )
        raise ValueError(
            f"The first snapshot time ({first_snap}) is before the "
            f"beginning of the simulation ({t_begin})."
        )
    elif doing_scale_factor and first_snap < a_begin:
        raise ValueError(
            f"The first snapshot scale factor ({first_snap}) is before the "
            f"beginning of the simulation ({a_begin})."
        )
    elif doing_log_scale_factor and 10**first_snap < a_begin:
        raise ValueError(
            f"The first snapshot scale factor ({10**first_snap}) is "
            f"before the beginning of the simulation ({a_begin})."
        )
    else:
        pass  # nothing to do, we are good

    # Check that this final snap is before a_end in the parameter file
    a_end = params["Cosmology"]["a_end"]
    if doing_z and convert_redshift_to_scale_factor(final_snap) > a_end:
        z_end = convert_scale_factor_to_redshift(a_end)
        raise ValueError(
            f"The final snapshot redshift ({final_snap}) is after the "
            f"end of the simulation ({z_end})."
        )
    elif (
        doing_time
        and convert_redshift_to_scale_factor(
            convert_time_to_redshift(final_snap)
        )
        > a_end
    ):
        t_end = convert_redshift_to_time(
            convert_scale_factor_to_redshift(a_end)
        )
        raise ValueError(
            f"The final snapshot time ({final_snap}) is after the "
            f"end of the simulation ({t_end})."
        )
    elif doing_scale_factor and final_snap > a_end:
        raise ValueError(
            f"The final snapshot scale factor ({final_snap}) is after the "
            f"end of the simulation ({a_end})."
        )
    elif doing_log_scale_factor and 10**final_snap > a_end:
        raise ValueError(
            f"The final snapshot scale factor ({10**final_snap}) is "
            f"after the end of the simulation ({a_end})."
        )
    else:
        pass  # nothing to do, we are good

    # Get the output list of times for the snapshots
    if doing_z:
        snapshot_times = _get_out_list_z(first_snap, delta_z, final_snap)
    elif doing_time:
        snapshot_times = _get_out_list_time(first_snap, delta_time, final_snap)
    elif doing_scale_factor:
        snapshot_times = _get_out_list_scale_factor(
            first_snap, delta_scale_factor, final_snap
        )
    elif doing_log_scale_factor:
        snapshot_times = _get_out_list_log_scale_factor(
            first_snap, delta_log_scale_factor, final_snap
        )
    else:
        raise ValueError(
            "You must specify the first snapshot, final snapshot, and delta"
            " in terms of redshift, time, scale factor, or logarithmic "
            "scale factor."
        )

    # If we are getting them, get the snipshot times
    if has_snipshots:
        if snip_doing_z:
            snipshot_times = _get_out_list_z(
                snip_first_snap, snip_delta_z, snip_final_snap
            )
        elif snip_doing_time:
            snipshot_times = _get_out_list_time(
                snip_first_snap, snip_delta_time, snip_final_snap
            )
        elif snip_doing_scale_factor:
            snipshot_times = _get_out_list_scale_factor(
                snip_first_snap, snip_delta_scale_factor, snip_final_snap
            )
        elif snip_doing_log_scale_factor:
            snipshot_times = _get_out_list_log_scale_factor(
                snip_first_snap, snip_delta_log_scale_factor, snip_final_snap
            )
        else:
            raise ValueError(
                "You must specify the first snapshot, final snapshot, and "
                "snipshot delta in terms of redshift, time, scale factor, or "
                "logarithmic scale factor."
            )

        # We also need to make sure that the snapshots and snipshots are in
        # the same units
        if doing_z and snip_doing_z:
            pass  # nothing to do
        elif doing_time and snip_doing_time:
            pass  # nothing to do
        elif doing_scale_factor and snip_doing_scale_factor:
            pass  # nothing to do
        elif doing_log_scale_factor and snip_doing_log_scale_factor:
            pass  # nothing to do
        elif doing_scale_factor and snip_doing_log_scale_factor:
            pass  # nothing to do (result is still in scale factor)
        elif doing_log_scale_factor and snip_doing_scale_factor:
            pass  # nothing to do (result is still in scale factor)
        elif doing_z and snip_doing_time:
            snipshot_times = convert_time_to_redshift(snipshot_times)
        elif doing_z and (
            snip_doing_scale_factor or snip_doing_log_scale_factor
        ):
            snipshot_times = convert_scale_factor_to_redshift(snipshot_times)
        elif doing_time and snip_doing_z:
            snipshot_times = convert_redshift_to_time(snipshot_times)
        elif doing_time and (
            snip_doing_scale_factor or snip_doing_log_scale_factor
        ):
            snipshot_times = convert_redshift_to_time(
                convert_scale_factor_to_redshift(snipshot_times)
            )
        elif (doing_scale_factor or doing_log_scale_factor) and snip_doing_z:
            snipshot_times = convert_redshift_to_scale_factor(snipshot_times)
        elif (
            doing_scale_factor or doing_log_scale_factor
        ) and snip_doing_time:
            snipshot_times = convert_redshift_to_scale_factor(
                convert_time_to_redshift(snipshot_times)
            )
        else:
            raise ValueError(
                "You surely haven't been able to get here, something went "
                "very wrong! (Or we didn't cover all bases, contact the "
                "developers to fix this.)"
            )

    else:
        snipshot_times = np.array([])

    # Write the output list to the file
    write_output_list(
        out_file,
        snapshot_times,
        snipshot_times,
        doing_z=doing_z,
        doing_time=doing_time,
        doing_scale_factor=doing_scale_factor,
    )


def generate_output_list(args: dict) -> None:
    """Generate an output list file containing times for each snapshot.

    Args:
        args: Command-line arguments containing the configuration for the
              output list generation.
    """
    # Get a cosmology (None if no parameters were passed)
    cosmo = get_cosmology()

    # If we have a cosmology we can convert all the users inputs to redshift
    # which is simpler and more flexible
    if cosmo is not None:
        _generate_output_list_with_cosmo(args, cosmo)
    else:
        _generate_output_list_no_cosmo(args)


def run(args: argparse.Namespace) -> None:
    """Execute the output-times mode."""
    generate_output_list(vars(args))
