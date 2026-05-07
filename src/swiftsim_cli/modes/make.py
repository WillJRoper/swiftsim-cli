"""Make mode for compiling SWIFT."""

import argparse
from pathlib import Path

from swiftsim_cli.swiftsim_dir import (
    _run_command_in_swift_dir,
    get_swiftsim_dir,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'make' mode."""
    # Add the number of threads to use for compilation.
    parser.add_argument(
        "--nr-threads",
        "-j",
        type=int,
        default=1,
        help="Number of threads to use for compilation (default: 1).",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the make mode."""
    make_swift(nr_threads=args.nr_threads)


def make_swift(swift_dir: Path | None = None, nr_threads: int = 1) -> None:
    """Compile the SWIFT simulation code.

    This will navigate to the SWIFT directory (erroring if there is not one
    set) and then run the make command to compile the SWIFT code.

    Args:
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.
        nr_threads: The number of threads to use for compilation. Defaults
            to 1.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If the SWIFT directory is not set in the configuration or
            if the number of threads is less than 1.
    """
    # Get the SWIFT directory
    swift_dir = get_swiftsim_dir(swift_dir)

    # Run the command in the SWIFT directory with optional threading
    if nr_threads < 1:
        raise ValueError("Number of threads must be at least 1.")
    elif nr_threads > 1:
        _run_command_in_swift_dir(["make", "-j", str(nr_threads)], swift_dir)
    else:
        _run_command_in_swift_dir(["make"], swift_dir)
