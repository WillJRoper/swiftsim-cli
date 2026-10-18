"""Analyse mode for analysing SWIFT runs.

This module provides comprehensive analysis capabilities for SWIFT
simulation data:

- Timestep analysis: Analyse timestep evolution and plotting
- Gravity checks: Force accuracy validation and error analysis
- Error maps: Gravity error visualization and mapping
- Log timing: Performance profiling and timing analysis with plots
- Task counts: Analyse engine_print_task_counts output for task distribution
- MPI use: Analyse mpiuse reports for communication volume
- Task plots: Plot task-debug and threadpool timelines
- Scaling: Analyse timer and runtime scaling across MPI rank counts

Every subcommand writes its files into ``<--out>/<subcommand>_analysis/``
(``--out`` defaults to the current directory), with ``--prefix``
prepended once to each filename.

Each analysis type is implemented in its own submodule for better organization
and maintainability.
"""

import argparse

from .gravity_checks import run_gravity_checks
from .gravity_error_maps import run_gravity_error_maps
from .log_task_counts import run_swift_task_counts
from .log_timing import run_swift_log_timing
from .mpiuse import run_swift_mpiuse
from .scaling import run_scaling
from .task_plots import run_swift_task_plots
from .timesteps import run_timestep


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'analyse' mode with subparsers."""
    parser.epilog = (
        "Each analysis writes its files into <out>/<analysis>_analysis/ "
        "(-o/--out, default: the current directory), with -p/--prefix "
        "prepended to each filename."
    )
    # Create subparsers for different analysis types
    subparsers = parser.add_subparsers(
        dest="analysis_type",
        help="Type of analysis to perform.",
        required=True,
    )

    # Import and set up each submodule's arguments
    from .gravity_checks import add_gravity_checks_arguments
    from .gravity_error_maps import add_gravity_error_maps_arguments
    from .log_task_counts import add_task_counts_arguments
    from .log_timing import add_log_arguments
    from .mpiuse import add_mpiuse_arguments
    from .scaling import add_scaling_arguments
    from .task_plots import add_task_plots_arguments
    from .timesteps import add_timestep_arguments

    add_timestep_arguments(subparsers)
    add_gravity_checks_arguments(subparsers)
    add_gravity_error_maps_arguments(subparsers)
    add_log_arguments(subparsers)
    add_task_counts_arguments(subparsers)
    add_mpiuse_arguments(subparsers)
    add_task_plots_arguments(subparsers)
    add_scaling_arguments(subparsers)


def run(args: argparse.Namespace) -> None:
    """Run the appropriate analysis based on the selected type."""
    if args.analysis_type == "timesteps":
        run_timestep(args)
    elif args.analysis_type == "gravity-checks":
        run_gravity_checks(args)
    elif args.analysis_type == "gravity-error-maps":
        run_gravity_error_maps(args)
    elif args.analysis_type == "log":
        run_swift_log_timing(args)
    elif args.analysis_type == "task-counts":
        run_swift_task_counts(args)
    elif args.analysis_type == "mpiuse":
        run_swift_mpiuse(args)
    elif args.analysis_type == "task-plots":
        run_swift_task_plots(args)
    elif args.analysis_type == "scaling":
        run_scaling(args)
    else:
        raise ValueError(f"Unknown analysis type: {args.analysis_type}")


# Import functions for backward compatibility and external access
from .log_task_counts import analyse_swift_task_counts
from .log_timing import analyse_swift_log_timings
from .mpiuse import analyse_swift_mpiuse
from .task_plots import analyse_swift_task_plots
from .timesteps import analyse_timestep_files

__all__ = [
    "add_arguments",
    "run",
    "run_timestep",
    "run_swift_log_timing",
    "run_swift_task_counts",
    "run_swift_mpiuse",
    "run_swift_task_plots",
    "run_scaling",
    "run_gravity_checks",
    "run_gravity_error_maps",
    "analyse_timestep_files",
    "analyse_swift_log_timings",
    "analyse_swift_task_counts",
    "analyse_swift_mpiuse",
    "analyse_swift_task_plots",
]
