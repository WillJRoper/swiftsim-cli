"""Config mode for configuring the active SWIFT profile."""

import argparse

from swiftsim_cli.profile import get_profile
from swiftsim_cli.swiftsim_dir import (
    _run_command_in_swift_dir,
    configure_profile,
    ensure_configure_script,
    get_swiftsim_dir,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for configuring SWIFT itself."""
    parser.epilog = (
        "Any other --<option> flags are passed to ./configure after being "
        "checked against its options (list them with --list-options)."
    )
    # Show the configuration options (equivalent to running
    # `./configure --help`).
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List the available ./configure options (./configure --help) "
        "and exit.",
    )
    parser.add_argument(
        "--debug-build",
        action="store_true",
        help="Configure using the preset debug options "
        "(--enable-debug --enable-debugging-checks --disable-optimization).",
    )
    parser.add_argument(
        "--gravity",
        action="store_true",
        help="Configure using the preset gravity-only options "
        "(--enable-ipo --with-tbbmalloc --with-parmetis).",
    )
    parser.add_argument(
        "--eagle",
        action="store_true",
        help="Configure using the preset EAGLE options "
        "(--with-subgrid=EAGLE --with-hydro=sphenix --with-kernel=wendland-C2 "
        "--enable-ipo --with-tbbmalloc --with-parmetis).",
    )
    parser.add_argument(
        "--eaglexl",
        action="store_true",
        help="Configure using the preset EAGLE-XL options "
        "(--with-subgrid=EAGLE-XL --with-hydro=sphenix "
        "--with-kernel=wendland-C2 --enable-ipo --with-tbbmalloc "
        "--with-parmetis).",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the config mode.

    The presets and any extra ``--<option>`` flags become the active
    profile's configure flags (validated against ``./configure --help``).
    With no presets or flags, the profile is reconfigured with its stored
    flags.

    Args:
        args: The parsed arguments from the command line.
    """
    # Are we just showing the config options?
    if args.list_options:
        show_config_options()
    else:
        # Build up the options to pass to the configure script
        opts = []
        if args.debug_build:
            opts.extend(
                [
                    "--enable-debug",
                    "--enable-debugging-checks",
                    "--disable-optimization",
                ]
            )
        if args.gravity:
            opts.extend(
                [
                    "--enable-ipo",
                    "--with-tbbmalloc",
                    "--with-parmetis",
                ]
            )
        if args.eagle:
            opts.extend(
                [
                    "--with-subgrid=EAGLE",
                    "--with-hydro=sphenix",
                    "--with-kernel=wendland-C2",
                    "--enable-ipo",
                    "--with-tbbmalloc",
                    "--with-parmetis",
                ]
            )
        if args.eaglexl:
            opts.extend(
                [
                    "--with-subgrid=EAGLE-XL",
                    "--with-hydro=sphenix",
                    "--with-kernel=wendland-C2",
                    "--enable-ipo",
                    "--with-tbbmalloc",
                    "--with-parmetis",
                ]
            )
        opts.extend(args.options)

        # Remove any duplicate options while preserving order
        unique_opts = list(dict.fromkeys(opts))
        args.options = unique_opts

        # Now run the configuration
        config_swiftsim(opts=unique_opts or None)


def config_swiftsim(opts: list[str] | None = None) -> None:
    """Configure the active profile's worktree.

    Args:
        opts: Flags for ``./configure``; these replace the profile's stored
            configure flags. If None, the stored flags are used.

    Raises:
        ProfileError: If a flag is unknown or configure fails.
    """
    configure_profile(get_profile(), opts)


def show_config_options() -> None:
    """Show the configure options for the active profile's worktree."""
    swift_dir = get_swiftsim_dir()
    ensure_configure_script(swift_dir)
    _run_command_in_swift_dir(["./configure", "--help"], swift_dir)
