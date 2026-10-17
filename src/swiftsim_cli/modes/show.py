"""Show mode for displaying the current SWIFT profile.

This is a shortcut for 'swift-cli profile --show'.
"""

import argparse

from swiftsim_cli.modes.profile import display_profile


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the show mode."""
    parser.add_argument(
        "--cosmo",
        action="store_true",
        help="Show only the cosmology from the parameter file.",
        default=False,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show both the profile and cosmology.",
        default=False,
    )


def run(args: argparse.Namespace) -> None:
    """Execute the show mode.

    This calls display_profile() from the profile mode with appropriate flags.
    """
    # Determine what to show
    if args.cosmo:
        # Show only cosmology
        display_profile(
            print_header=False,
            show_profile=False,
            show_cosmology=True,
        )
    elif args.all:
        # Show both
        display_profile(show_profile=True, show_cosmology=True)
    else:
        # Show only profile (default)
        display_profile(show_profile=True, show_cosmology=False)
