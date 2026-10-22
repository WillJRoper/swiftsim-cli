"""Argument definitions and validators shared by the modes.

Defining these once keeps flag names, defaults, metavars, validation and
help text identical everywhere they appear.

Input paths are checked when the command line is parsed, so a typo fails
before any mode runs. The one exception is ``-P/--params``: it is checked
when its mode runs, since an earlier ``new`` in the same chain may create
it.
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# Argument types
# ---------------------------------------------------------------------------


def _number(
    kind: Callable[[str], Any], test: Callable[[Any], bool], what: str
) -> Callable[[str], Any]:
    """Build an argparse type converting with ``kind`` and checking ``test``.

    Args:
        kind: The conversion (int or float).
        test: The check the converted value must pass.
        what: Description of a valid value for the error message.

    Returns:
        The argparse type function.
    """

    def convert(text: str) -> Any:
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"expected {what}, got {text!r}"
            ) from None
        if not test(value):
            raise argparse.ArgumentTypeError(f"expected {what}, got {text!r}")
        return value

    return convert


positive_int = _number(int, lambda v: v > 0, "a positive integer")
non_negative_int = _number(int, lambda v: v >= 0, "a non-negative integer")
positive_float = _number(float, lambda v: v > 0, "a positive number")


def _path_type(
    test: Callable[[Path], bool], what: str
) -> Callable[[str], Path]:
    """Build an argparse type returning a Path that must pass ``test``.

    Args:
        test: The check the path must pass.
        what: Description of the expected kind of path.

    Returns:
        The argparse type function.
    """

    def convert(text: str) -> Path:
        path = Path(text).expanduser()
        if not test(path):
            raise argparse.ArgumentTypeError(f"{what} not found: '{text}'")
        return path

    return convert


existing_file = _path_type(Path.is_file, "file")
existing_dir = _path_type(Path.is_dir, "directory")
existing_path = _path_type(Path.exists, "file or directory")


def rank_arg(text: str) -> int | str:
    """Parse ``--rank``: a non-negative rank number or ``all``.

    Args:
        text: The command-line value.

    Returns:
        The rank number, or the string ``"all"``.
    """
    if text == "all":
        return text
    try:
        return non_negative_int(text)
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(
            f"expected a rank number or 'all', got {text!r}"
        ) from None


class StepRange(argparse.Action):
    """Store ``--steps START END`` as a list, checking 0 <= START <= END."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        """Validate and store the step range."""
        start, end = values
        if start > end:
            parser.error(
                f"{option_string}: START ({start}) must not exceed END ({end})"
            )
        setattr(namespace, self.dest, [start, end])


def resolve_labels(
    inputs: Sequence[str | Path], labels: Sequence[str] | None
) -> list[str]:
    """Return one label per input.

    Given labels must match the inputs one to one. Otherwise each input is
    labelled by its file stem, with ``_1``, ``_2``, ... appended to stems
    shared by several inputs.

    Args:
        inputs: The input files (or directories / patterns).
        labels: The labels given on the command line, if any.

    Returns:
        The labels, in input order.

    Raises:
        ValueError: If the number of labels does not match the inputs.
    """
    if labels is not None:
        if len(labels) != len(inputs):
            raise ValueError(
                f"got {len(labels)} label(s) for {len(inputs)} input(s); "
                "give one label per input or none"
            )
        return list(labels)
    stems = [Path(p).stem or str(p) for p in inputs]
    counts = Counter(stems)
    seen: Counter[str] = Counter()
    result = []
    for stem in stems:
        if counts[stem] > 1:
            seen[stem] += 1
            stem = f"{stem}_{seen[stem]}"
        result.append(stem)
    return result


# ---------------------------------------------------------------------------
# Shared arguments
# ---------------------------------------------------------------------------


def add_output_arguments(parser: argparse.ArgumentParser, subdir: str) -> None:
    """Add the ``-o/--out``, ``-p/--prefix`` and ``--show`` output options.

    Every analysis writes its files into ``<out>/<subdir>/``, with the
    prefix prepended once to each filename.

    Args:
        parser: The parser to add the options to.
        subdir: The subdirectory of ``--out`` the outputs are written to.
    """
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help=f"Directory to write outputs to; files go in its '{subdir}' "
        "subdirectory (default: current directory).",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default=None,
        metavar="PREFIX",
        help="Prefix prepended to each output filename (default: none).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively after saving them.",
    )


def add_labels_argument(parser: argparse.ArgumentParser) -> None:
    """Add the optional ``-l/--labels`` option naming each input.

    Resolve the labels with `resolve_labels`.

    Args:
        parser: The parser to add the option to.
    """
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        type=str,
        default=None,
        metavar="LABEL",
        help="One label per input, in the same order (default: the file "
        "stems). Put the inputs before -l.",
    )


def add_steps_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--steps START END`` step-range option.

    Args:
        parser: The parser to add the option to.
    """
    parser.add_argument(
        "--steps",
        nargs=2,
        type=non_negative_int,
        action=StepRange,
        metavar=("START", "END"),
        default=None,
        help="Inclusive step range to analyse (default: all steps).",
    )


def add_rank_argument(
    parser: argparse.ArgumentParser, default: int | str, help: str
) -> None:
    """Add the ``--rank N|all`` MPI rank option.

    Args:
        parser: The parser to add the option to.
        default: The default rank (a number or ``"all"``).
        help: What the rank selects; the accepted values and default are
            appended.
    """
    parser.add_argument(
        "--rank",
        type=rank_arg,
        default=default,
        metavar="N|all",
        help=f"{help} Give a rank number or 'all' (default: {default}).",
    )


def add_nr_threads_argument(
    parser: argparse.ArgumentParser,
    help: str = "Number of threads to use for compilation (default: 1).",
) -> None:
    """Add the ``-j/--nr-threads`` compilation option.

    Args:
        parser: The parser to add the option to.
        help: The help text for the option.
    """
    parser.add_argument(
        "-j",
        "--nr-threads",
        type=positive_int,
        default=1,
        metavar="N",
        help=help,
    )


def add_params_argument(
    parser: argparse.ArgumentParser, required: bool = False
) -> None:
    """Add the ``-P/--params`` SWIFT parameter file option.

    The file is checked when the mode runs (an earlier ``new`` in the
    chain may create it), not when the command line is parsed.

    Args:
        parser: The parser to add the option to.
        required: Whether a parameter file must be given.
    """
    parser.add_argument(
        "-P",
        "--params",
        type=Path,
        required=required,
        default=None,
        metavar="FILE",
        help="SWIFT parameter file"
        + ("." if required else " (default: none)."),
    )
