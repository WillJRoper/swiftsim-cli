"""Update mode for pulling and rebuilding the active SWIFT profile.

This is equivalent to 'swift-cli profile --update' for the active profile.
"""

import argparse

from swiftsim_cli.profile import get_profile
from swiftsim_cli.swiftsim_dir import update_profile


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'update' mode."""
    parser.add_argument(
        "--nr-threads",
        "-j",
        type=int,
        default=1,
        help="Number of threads to use for compilation (default: 1).",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the update mode."""
    update_profile(get_profile(), nr_threads=args.nr_threads)
