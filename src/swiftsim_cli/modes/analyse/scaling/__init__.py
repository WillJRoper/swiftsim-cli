"""``analyse scaling``: how SWIFT scales across runs of different sizes.

Subcommands:
    timers: how each timer's exclusive time scales (``timers.py``).
    runtime: whole-run strong scaling from the step table (``runtime.py``).

Every log is scanned once (``data.py``); plots are drawn by ``plots.py``.
"""

from __future__ import annotations

import argparse

from swiftsim_cli.modes.analyse.log_timing import add_timer_db_arguments
from swiftsim_cli.modes.analyse.scaling.data import resolve_log_files
from swiftsim_cli.modes.analyse.scaling.runtime import (
    OUT_SUBDIR as RUNTIME_OUT_SUBDIR,
)
from swiftsim_cli.modes.analyse.scaling.runtime import (
    analyse_swift_runtime_scaling,
)
from swiftsim_cli.modes.analyse.scaling.timers import (
    OUT_SUBDIR as TIMERS_OUT_SUBDIR,
)
from swiftsim_cli.modes.analyse.scaling.timers import analyse_swift_scaling
from swiftsim_cli.modes.common_args import (
    add_output_arguments,
    add_steps_argument,
    positive_float,
)

__all__ = [
    "RUNTIME_OUT_SUBDIR",
    "TIMERS_OUT_SUBDIR",
    "add_scaling_arguments",
    "analyse_swift_runtime_scaling",
    "analyse_swift_scaling",
    "run_swift_runtime_scaling",
    "run_swift_scaling",
]


def _add_common(parser: argparse.ArgumentParser, subdir: str) -> None:
    parser.add_argument(
        "log_files",
        nargs="+",
        type=str,
        metavar="FILE",
        help="SWIFT log files (or glob patterns) of runs with different MPI "
        "rank counts.",
    )
    add_output_arguments(parser, subdir)
    add_steps_argument(parser)


def add_scaling_arguments(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``scaling`` subcommand and its ``timers``/``runtime``."""
    parser = subparsers.add_parser(
        "scaling",
        help="Analyse how SWIFT scales across runs with different MPI rank "
        "counts.",
        description="Compare SWIFT logs of runs with different MPI rank "
        "counts.",
    )
    kinds = parser.add_subparsers(
        dest="scaling_type",
        metavar="KIND",
        help="Type of scaling analysis to perform.",
        required=True,
    )
    timers = kinds.add_parser(
        "timers",
        help="Analyse how SWIFT timers scale across log files with "
        "different MPI rank counts.",
        description="Exclusive time of each timer (from the per-step timer "
        "trees, so nested timers are not double counted) against the MPI "
        "rank count.",
    )
    timers.set_defaults(func=run_swift_scaling)
    _add_common(timers, TIMERS_OUT_SUBDIR)
    timers.add_argument(
        "--min-percent-threshold",
        type=positive_float,
        metavar="PCT",
        default=5.0,
        help="Minimum share of the step time (the timer's exclusive time) "
        "in at least one log for a timer to be included (default: 5).",
    )
    timers.add_argument(
        "--rank-source",
        choices=("root", "all"),
        default="root",
        help="Ranks used for the main view: 'root' is rank 0, 'all' the "
        "mean over the ranks that printed timers (-v 2 logs) "
        "(default: root).",
    )
    add_timer_db_arguments(timers)

    runtime = kinds.add_parser(
        "runtime",
        help="Analyse whole-run strong scaling using the step-table "
        "wall-clock totals in SWIFT logs.",
        description="Summed step-table wall-clock time of each log against "
        "the MPI rank count.",
    )
    runtime.set_defaults(func=run_swift_runtime_scaling)
    _add_common(runtime, RUNTIME_OUT_SUBDIR)


def run_swift_scaling(args: argparse.Namespace) -> None:
    """Entry point for ``analyse scaling timers``."""
    analyse_swift_scaling(
        log_files=resolve_log_files(args.log_files),
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        step_range=tuple(args.steps) if args.steps else None,
        min_percent_threshold=args.min_percent_threshold,
        rank_source=args.rank_source,
        swift_src=args.swift_src,
        match_revision=args.match_revision,
        regenerate_db=args.regenerate_db,
    )


def run_swift_runtime_scaling(args: argparse.Namespace) -> None:
    """Entry point for ``analyse scaling runtime``."""
    analyse_swift_runtime_scaling(
        log_files=resolve_log_files(args.log_files),
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        step_range=tuple(args.steps) if args.steps else None,
    )
