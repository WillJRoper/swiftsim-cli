"""Update mode for pulling and rebuilding the active SWIFT profile.

This is equivalent to 'swift-cli profile --update' for the active profile.
"""

import argparse

from swiftsim_cli.modes.common_args import add_nr_threads_argument
from swiftsim_cli.profile import get_profile
from swiftsim_cli.swiftsim_dir import update_profile
from swiftsim_cli.utilities.messages import note

# Modes that (re)build the profile themselves when they follow update
BUILDING_MODES = ("config", "make")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'update' mode."""
    add_nr_threads_argument(parser)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Only pull; don't reconfigure or build. Implied when a later "
        "'config' or 'make' in the same command does that.",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the update mode."""
    # cli.main sets later_modes to the modes after this one in the chain
    later = [
        m for m in getattr(args, "later_modes", []) if m in BUILDING_MODES
    ]
    if later and not args.no_build:
        note(f"not building here: the later '{later[0]}' in this command does")
    update_profile(
        get_profile(),
        nr_threads=args.nr_threads,
        build=not (args.no_build or later),
    )
