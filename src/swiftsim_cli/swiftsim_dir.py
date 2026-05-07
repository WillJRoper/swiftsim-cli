"""A module containing tools for managing the SWIFT simulation directory."""

from pathlib import Path
from typing import Sequence

from swiftsim_cli.profile import load_swift_profile
from swiftsim_cli.utilities import run_command_in_dir


def get_swiftsim_dir(swift_dir: Path | None = None) -> Path:
    """Get the SWIFT simulation directory.

    This will attempt to read the SWIFT directory from the configuration file
    if one is not passed. It will then check if the directory exists, raising
    an error if it does not.

    Args:
        swift_dir: Optional path to the SWIFT directory. If None, uses the
            directory from the SWIFT-utils config.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If the SWIFT directory is not set in the configuration.
    """
    # Load the SWIFT configuration
    config = load_swift_profile()

    # If we haven't been given a SWIFT directory, use the one from the config
    if swift_dir is None:
        swift_dir = config.swiftsim_dir

    # If we still don't have a SWIFT directory, raise an error
    if swift_dir is None:
        raise ValueError(
            "SWIFT directory not passed (--swift-dir) and "
            "not found in the configuration file."
        )

    # Does it actually exist?
    if not swift_dir.exists():
        raise FileNotFoundError(f"SWIFT directory does not exist: {swift_dir}")

    return swift_dir


def _run_command_in_swift_dir(command: Sequence[str], swift_dir: Path) -> None:
    """Run a command in the SWIFT directory.

    This will navigate to the SWIFT directory and run the provided command.

    Args:
        command: The command to run in the SWIFT directory.
        swift_dir: The path to the SWIFT directory.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If the SWIFT directory is not set in the configuration.
    """
    # Get the SWIFT directory
    swift_dir = get_swiftsim_dir(swift_dir)

    # Run the command in the SWIFT directory
    run_command_in_dir(command, swift_dir)
