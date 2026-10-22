"""Analyse mode for analysing SWIFT runs.

Each subcommand lives in its own submodule, which adds its parser and
binds its entry point with ``set_defaults(func=...)``:

- timesteps: timestep files (runtime, dead time, timestep evolution)
- gravity-checks / gravity-error-maps: force accuracy against exact checks
- log: timer call trees from a SWIFT log run with -v 1
- task-counts: engine_print_task_counts output
- mpiuse: mpiuse reports (communication volume)
- task-plots: task-debug and threadpool timelines
- scaling: timer and runtime scaling across MPI rank counts

Every subcommand writes its files into ``<--out>/<subcommand>_analysis/``
(``--out`` defaults to the current directory), with ``--prefix``
prepended once to each filename. The submodules import their plotting
dependencies inside their functions, so building this parser is cheap.
"""

import argparse

from .gravity_checks import add_gravity_checks_arguments
from .gravity_error_maps import add_gravity_error_maps_arguments
from .log_task_counts import add_task_counts_arguments
from .log_timing import add_log_arguments
from .mpiuse import add_mpiuse_arguments
from .scaling import add_scaling_arguments
from .task_plots import add_task_plots_arguments
from .timesteps import add_timestep_arguments


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the analysis subcommands to the 'analyse' mode parser."""
    parser.epilog = (
        "Each analysis writes its files into <out>/<analysis>_analysis/ "
        "(-o/--out, default: the current directory), with -p/--prefix "
        "prepended to each filename."
    )
    subparsers = parser.add_subparsers(
        dest="analysis_type",
        metavar="ANALYSIS",
        help="Type of analysis to perform.",
        required=True,
    )
    add_timestep_arguments(subparsers)
    add_gravity_checks_arguments(subparsers)
    add_gravity_error_maps_arguments(subparsers)
    add_log_arguments(subparsers)
    add_task_counts_arguments(subparsers)
    add_mpiuse_arguments(subparsers)
    add_task_plots_arguments(subparsers)
    add_scaling_arguments(subparsers)


def run(args: argparse.Namespace) -> None:
    """Run the selected analysis."""
    args.func(args)
