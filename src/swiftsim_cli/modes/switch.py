"""Switch mode for changing the current SWIFT profile.

This is an alias for 'swift-cli profile --switch NAME'.
"""

import argparse

from swiftsim_cli.modes.profile import switch_profile


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'switch' mode."""
    parser.add_argument(
        "name",
        type=str,
        help="The profile to make current (see 'swift-cli profile --list').",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the switch mode."""
    switch_profile(args.name)
