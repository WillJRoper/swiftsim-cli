"""Argument definitions shared by several modes and analyse subcommands.

Defining these once keeps flag names, defaults and help text identical
everywhere they appear.
"""

import argparse
from pathlib import Path


def add_output_arguments(parser: argparse.ArgumentParser, subdir: str) -> None:
    """Add the ``--out``, ``--prefix`` and ``--show`` output options.

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
        help=f"Directory to write outputs to; files go in its '{subdir}' "
        "subdirectory (default: current directory).",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default=None,
        help="Prefix prepended to each output filename (default: none).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively after saving them.",
    )


def add_labels_argument(
    parser: argparse.ArgumentParser, required: bool = False
) -> None:
    """Add the ``-l/--labels`` option naming each input.

    Args:
        parser: The parser to add the option to.
        required: Whether labels must be given.
    """
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        type=str,
        required=required,
        default=None,
        help="Labels for the inputs, in the same order"
        + ("." if required else " (default: derived from the filenames)."),
    )


def add_steps_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--steps START END`` step-range option.

    Args:
        parser: The parser to add the option to.
    """
    parser.add_argument(
        "--steps",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Inclusive step range to analyse (default: all steps).",
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
        type=int,
        default=1,
        help=help,
    )


def add_params_argument(
    parser: argparse.ArgumentParser, required: bool = False
) -> None:
    """Add the ``-P/--params`` SWIFT parameter file option.

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
        help="Path to a SWIFT parameter file"
        + ("." if required else " (default: none)."),
    )
