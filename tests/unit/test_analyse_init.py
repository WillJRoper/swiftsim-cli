"""Tests for the analyse mode init module."""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from swiftsim_cli.modes.analyse import add_arguments, run

# CLI argv and the heavy analysis function each subcommand must reach.
_SUBCOMMAND_CASES = {
    "timesteps": (
        ["timesteps", "a.txt", "-l", "a"],
        "swiftsim_cli.modes.analyse.timesteps.analyse_timestep_files",
    ),
    "gravity-checks": (
        ["gravity-checks", "a.dat", "-l", "a"],
        "swiftsim_cli.modes.analyse.gravity_checks.analyse_force_checks",
    ),
    "gravity-error-maps": (
        ["gravity-error-maps", "a.dat", "-l", "a"],
        "swiftsim_cli.modes.analyse.gravity_error_maps."
        "analyse_gravity_error_maps",
    ),
    "log": (
        ["log", "swift.log"],
        "swiftsim_cli.modes.analyse.log_timing.analyse_swift_log_timings",
    ),
    "task-counts": (
        ["task-counts", "swift.log"],
        "swiftsim_cli.modes.analyse.log_task_counts.analyse_swift_task_counts",
    ),
    "mpiuse": (
        ["mpiuse", "reports"],
        "swiftsim_cli.modes.analyse.mpiuse.analyse_swift_mpiuse",
    ),
    "task-plots": (
        ["task-plots", "tasks.dat", "--threadpool", "threadpool.dat"],
        "swiftsim_cli.modes.analyse.task_plots.analyse_swift_task_plots",
    ),
    "scaling": (
        ["scaling", "timers", "a.log", "b.log"],
        "swiftsim_cli.modes.analyse.scaling.analyse_swift_scaling",
    ),
}


def _analyse_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser


def test_every_registered_subcommand_has_a_dispatch_case():
    """New subcommands must be added to the parse->dispatch test."""
    group = _analyse_parser()._subparsers
    assert group is not None
    subparsers = group._group_actions[0]
    assert isinstance(subparsers.choices, dict)
    assert set(subparsers.choices) == set(_SUBCOMMAND_CASES)


@pytest.mark.parametrize("name", sorted(_SUBCOMMAND_CASES))
def test_parse_then_dispatch_reaches_analysis(name, in_inputs_dir):
    """Each subcommand's parser binds the entry point reaching its analysis."""
    argv, target = _SUBCOMMAND_CASES[name]
    args = _analyse_parser().parse_args(argv)

    with patch(target) as mock_analysis:
        run(args)

    mock_analysis.assert_called_once()


def test_scaling_runtime_dispatch(in_inputs_dir):
    """The scaling runtime subcommand reaches its analysis."""
    args = _analyse_parser().parse_args(
        ["scaling", "runtime", "a.log", "b.log", "-o", "out", "-p", "x"]
    )

    target = "swiftsim_cli.modes.analyse.scaling.analyse_swift_runtime_scaling"
    with patch(target) as mock_analysis:
        run(args)

    kwargs = mock_analysis.call_args.kwargs
    assert kwargs["log_files"] == ["a.log", "b.log"]
    assert kwargs["output_path"] == Path("out")
    assert kwargs["prefix"] == "x"
    assert kwargs["show_plot"] is False
