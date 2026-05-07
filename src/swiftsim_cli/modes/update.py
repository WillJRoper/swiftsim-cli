"""Update mode for pulling latest SWIFT changes."""

import argparse
from pathlib import Path

from swiftsim_cli.swiftsim_dir import (
    _run_command_in_swift_dir,
    get_swiftsim_dir,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'update' mode."""
    # No additional arguments for this mode
    pass


def run(args: argparse.Namespace) -> None:
    """Execute the update mode."""
    update_swift()


def update_swift(swift_dir: Path | None = None) -> None:
    """Update the SWIFT simulation directory.

    This will navigate to the SWIFT directory (erroring if there is not one
    set) and then run the git pull command to update the SWIFT repository.

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
    _run_command_in_swift_dir(["git", "pull"], swift_dir)
