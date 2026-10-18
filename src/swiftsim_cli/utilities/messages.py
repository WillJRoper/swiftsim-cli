"""User-facing messages that are not part of a command's normal output.

Warnings and notes go to stderr with a fixed prefix, so they never mix
with tables or values a user may pipe elsewhere.
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def warn(message: str) -> None:
    """Print ``warning: <message>`` to stderr.

    Args:
        message: The warning.
    """
    print(f"warning: {message}", file=sys.stderr)


def note(message: str) -> None:
    """Print ``note: <message>`` to stderr.

    Args:
        message: The note.
    """
    print(f"note: {message}", file=sys.stderr)


def plural(count: int, noun: str, suffix: str = "s") -> str:
    """Return ``"<count> <noun>"`` with the noun pluralised as needed.

    Args:
        count: The number of things.
        noun: The singular noun.
        suffix: The plural suffix.

    Returns:
        For example ``"1 rank"`` or ``"4 ranks"``.
    """
    return f"{count} {noun}{'' if count == 1 else suffix}"


@contextmanager
def reading(path: str | Path) -> Iterator[None]:
    """Name ``path`` in any ValueError raised while reading it.

    Use around parsing an input file so a one-line error says which file
    was bad (errors that already name it are left alone).

    Args:
        path: The file being read.

    Raises:
        ValueError: The original error, prefixed with the path.
    """
    try:
        yield
    except ValueError as e:
        if str(path) in str(e):
            raise
        raise ValueError(f"{path}: {e}") from e
