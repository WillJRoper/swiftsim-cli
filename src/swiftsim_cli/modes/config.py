"""A module containing tools for configuring SWIFT."""

import argparse
from pathlib import Path

from swiftsim_cli.swiftsim_dir import (
    _run_command_in_swift_dir,
    get_swiftsim_dir,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for configuring SWIFT itself."""
    # Show the configuration options (equivalent to running
    # `./configure --help`).
    parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Show the available configuration options.",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Configure using the preset debug options "
        "(--enable-debug --enable-debugging-checks --disable-optimization).",
    )
    parser.add_argument(
        "--gravity",
        "-g",
        action="store_true",
        help="Configure using the preset gravity-only options "
        "(--enable-ipo --with-tbbmalloc --with-parmetis).",
    )
    parser.add_argument(
        "--eagle",
        "-e",
        action="store_true",
        help="Configure using the preset EAGLE options "
        "(--with-subgrid=EAGLE --with-hydro=sphenix --with-kernel=wendland-C2 "
        "--enable-ipo --with-tbbmalloc  --with-parmetis).",
    )
    parser.add_argument(
        "--eaglexl",
        "-x",
        action="store_true",
        help="Configure using the preset EAGLE-XL options "
        "(--with-subgrid=EAGLE-XL --with-hydro=sphenix "
        "--with-kernel=wendland-C2 --enable-ipo --with-tbbmalloc "
        "--with-parmetis).",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the config mode.

    Note that the user can very readily pass incorrectly formed options. We let
    SWIFT itself handle that, as it will give a useful error message.

    Args:
        args: The parsed arguments from the command line.
    """
    # Are we just showing the config options?
    if args.show:
        show_config_options()
    else:
        # Build up the options to pass to the configure script
        opts = []
        if args.debug:
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
        seen = set()
        unique_opts = []
        for x in opts:
            if x not in seen:
                seen.add(x)
                unique_opts.append(x)
        args.options = unique_opts

        # Now run the configuration
        config_swiftsim(opts=unique_opts)


def config_swiftsim(opts: list[str], swift_dir: Path | None = None) -> None:
    """Configure SWIFT itself.

    This will navigate to the SWIFT directory (erroring if there is not one
    set) and then run the SWIFT configuration script with the provided options.

    Args:
        opts: The options to pass to the SWIFT configuration script.
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If the SWIFT directory is not set in the configuration.
    """
    # Get the SWIFT directory
    swift_dir = get_swiftsim_dir(swift_dir)

    # Run the command in the SWIFT directory
    _run_command_in_swift_dir(["./configure", *opts], swift_dir)


def show_config_options(swift_dir: Path | None = None) -> None:
    """Show the configuration options for SWIFT.

    This will navigate to the SWIFT directory (erroring if there is not one
    set) and then run the SWIFT configuration script with the --help option.

    Args:
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If the SWIFT directory is not set in the configuration.
    """
    # Get the SWIFT directory
    swift_dir = get_swiftsim_dir(swift_dir)

    # Run the command in the SWIFT directory
    _run_command_in_swift_dir(["./configure", "--help"], swift_dir)
