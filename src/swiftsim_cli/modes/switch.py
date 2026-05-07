"""Switch mode for switching SWIFT branches."""

import argparse
from pathlib import Path

from swiftsim_cli.profile import update_current_profile_value
from swiftsim_cli.swiftsim_dir import (
    _run_command_in_swift_dir,
    get_swiftsim_dir,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'switch' mode."""
    # Add the nameless argument for the branch to switch to.
    parser.add_argument(
        "branch",
        type=str,
        help="The branch to switch to in the SWIFT repository.",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the switch mode."""
    switch_swift_branch(branch=args.branch)


def switch_swift_branch(branch: str, swift_dir: Path | None = None) -> None:
    """Switch the SWIFT simulation directory to a different branch.

    This will navigate to the SWIFT directory (erroring if there is not one
    set) and then run the git checkout command to switch branches.

    Args:
        branch: The name of the branch to switch to.
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If the SWIFT directory is not set in the configuration.
    """
    # Get the SWIFT directory
    swift_dir = get_swiftsim_dir(swift_dir)

    # Run the command in the SWIFT directory
    _run_command_in_swift_dir(["git", "checkout", branch], swift_dir)

    # Update the current config
    update_current_profile_value("branch", branch)
