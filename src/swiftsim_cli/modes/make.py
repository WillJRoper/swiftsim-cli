"""Make mode for compiling the active SWIFT profile."""

import argparse

from swiftsim_cli.profile import get_profile
from swiftsim_cli.swiftsim_dir import build_profile


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
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Run 'make clean' before compiling.",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the make mode."""
    make_swift(nr_threads=args.nr_threads, clean=args.clean)


def make_swift(nr_threads: int = 1, clean: bool = False) -> None:
    """Compile the active profile's SWIFT worktree.

    The worktree is (re)configured first if it has never been configured or
    its configure flags changed since the last configure.

    Args:
        nr_threads: The number of threads to use for compilation.
        clean: Whether to run ``make clean`` before compiling.

    Raises:
        ValueError: If the number of threads is less than 1.
        ProfileError: If configure or make fails.
    """
    build_profile(get_profile(), nr_threads=nr_threads, clean=clean)
