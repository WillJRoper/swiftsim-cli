"""Show mode for displaying a SWIFT profile.

This is a shortcut for 'swift-cli profile --show [NAME]'.
"""

import argparse

from swiftsim_cli.modes.profile import display_profile


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the show mode."""
    parser.add_argument(
        "name",
        nargs="?",
        default=None,
        metavar="NAME",
        help="The profile to show (default: the active one).",
    )
    parser.add_argument(
        "--cosmo",
        action="store_true",
        help="Show only the cosmology from the parameter file.",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the show mode.

    Like 'profile --show': the profile's settings, then its parameter
    file's cosmology (with --cosmo, only the cosmology).
    """
    if args.cosmo:
        display_profile(print_header=False, show_profile=False, name=args.name)
    else:
        display_profile(name=args.name)
