"""Filesystem, subprocess and output-path helpers."""

import re
import subprocess
from pathlib import Path
from typing import Sequence

ascii_art = (
    r"    ______       _________________        _______   ____",
    r"   / ___/ |     / /  _/ ___/_  __/ ____  / ___/ /  /  _/",
    r"   \__ \| | /| / // // /_   / /   /___/ / /  / /   / /",
    r"  ___/ /| |/ |/ // // __/  / /         / /__/ /__ / /",
    r" /____/ |__/|__/___/_/    /_/         /____/____/___/",
)


def run_command_in_dir(command: Sequence[str], directory: Path) -> None:
    """Run a command in a specified directory.

    This function runs the command in the specified directory using subprocess.

    Args:
        command: The command to run as a sequence of arguments.
        directory: The directory in which to run the command.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        CalledProcessError: If the command fails.
    """
    subprocess.run(command, cwd=str(directory), check=True)


def create_output_path(
    output_path: str | Path | None = None,
    prefix: str | None = None,
    filename: str = "output.png",
    out_dir: str | None = None,
) -> Path:
    """Create and validate output path for saving files.

    The result is ``<output_path>/<out_dir>/<prefix>_<filename>``; the
    prefix is applied to the filename only, never to ``out_dir``.

    Args:
        output_path: Optional directory to save the file in. If None, uses
           the current directory.
        prefix: Optional prefix to add to the filename.
        filename: Base filename to use (default: "output.png").
        out_dir: An optional subdirectory of ``output_path`` to hold the
            outputs.

    Returns:
        Path: Complete path to the output file.

    Raises:
        ValueError: If the output path is not a directory.
    """
    # Create the output path
    if output_path is not None:
        path = Path(output_path)
    else:
        path = Path.cwd()

    # Ensure the output directory exists and is a directory
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path {path} exists but is not a directory.")

    # Create directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    # Create the output filename with optional prefix
    output_filename = f"{prefix + '_' if prefix else ''}{filename}"
    if out_dir is not None:
        # Create the output subdirectory if it doesn't exist
        (path / out_dir).mkdir(parents=True, exist_ok=True)
        output_file = path / out_dir / output_filename
    else:
        output_file = path / output_filename

    return output_file


def slugify(
    text: str,
    fallback: str = "item",
    lowercase: bool = False,
    keep: str = "",
) -> str:
    """Convert display text into a filesystem-friendly slug.

    Every run of characters other than ASCII letters, digits and the
    characters in ``keep`` is replaced by a single underscore, and leading
    or trailing underscores are removed.

    Args:
        text: The text to convert.
        fallback: Returned when nothing usable is left of ``text``.
        lowercase: Whether to lowercase the text first.
        keep: Extra characters to keep as they are (e.g. ``"._-"``).

    Returns:
        The slug, or ``fallback`` if the slug would be empty.
    """
    text = text.strip()
    if lowercase:
        text = text.lower()
    slug = re.sub(rf"[^A-Za-z0-9{re.escape(keep)}]+", "_", text)
    return slug.strip("_") or fallback
