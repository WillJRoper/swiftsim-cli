"""Tests for the analyse mode init module."""

import argparse
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.modes.analyse import add_arguments, run


class TestAnalyseMode:
    """Tests for analyse mode."""

    def test_add_arguments(self):
        """Test that add_arguments adds analysis type argument."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        # Test that the subparser was created
        # We can't parse without required args, so just verify structure
        assert hasattr(parser, "_subparsers")

    @patch("swiftsim_cli.modes.analyse.run_timestep")
    def test_run_timesteps(self, mock_run_timestep):
        """Test running timestep analysis."""
        args = Mock()
        args.analysis_type = "timesteps"

        run(args)

        mock_run_timestep.assert_called_once_with(args)

    @patch("swiftsim_cli.modes.analyse.run_gravity_checks")
    def test_run_gravity_checks(self, mock_run_gravity_checks):
        """Test running gravity checks."""
        args = Mock()
        args.analysis_type = "gravity-checks"

        run(args)

        mock_run_gravity_checks.assert_called_once_with(args)

    @patch("swiftsim_cli.modes.analyse.run_gravity_error_maps")
    def test_run_gravity_error_maps(self, mock_run_gravity_error_maps):
        """Test running gravity error maps."""
        args = Mock()
        args.analysis_type = "gravity-error-maps"

        run(args)

        mock_run_gravity_error_maps.assert_called_once_with(args)

    @patch("swiftsim_cli.modes.analyse.run_swift_log_timing")
    def test_run_log_timing(self, mock_run_swift_log_timing):
        """Test running log timing analysis."""
        args = Mock()
        args.analysis_type = "log"

        run(args)

        mock_run_swift_log_timing.assert_called_once_with(args)

    @patch("swiftsim_cli.modes.analyse.run_swift_task_counts")
    def test_run_task_counts(self, mock_run_swift_task_counts):
        """Test running task counts analysis."""
        args = Mock()
        args.analysis_type = "task-counts"

        run(args)

        mock_run_swift_task_counts.assert_called_once_with(args)

    @patch("swiftsim_cli.modes.analyse.run_swift_task_plots")
    def test_run_task_plots(self, mock_run_swift_task_plots):
        """Test running combined task/threadpool plotting."""
        args = Mock()
        args.analysis_type = "task-plots"

        run(args)

        mock_run_swift_task_plots.assert_called_once_with(args)

    @patch("swiftsim_cli.modes.analyse.run_scaling")
    def test_run_scaling(self, mock_run_scaling):
        """Test running scaling analysis."""
        args = Mock()
        args.analysis_type = "scaling"

        run(args)

        mock_run_scaling.assert_called_once_with(args)

    def test_run_unknown_type(self):
        """Test running with unknown analysis type."""
        args = Mock()
        args.analysis_type = "unknown"

        with pytest.raises(ValueError, match="Unknown analysis type"):
            run(args)


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
        ["task-plots", "tasks.dat", "threadpool.dat"],
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
    subparsers = _analyse_parser()._subparsers._group_actions[0]
    assert set(subparsers.choices) == set(_SUBCOMMAND_CASES)


@pytest.mark.parametrize("name", sorted(_SUBCOMMAND_CASES))
def test_parse_then_dispatch_reaches_analysis(name):
    """Parsed subcommand names must match the real dispatch table."""
    argv, target = _SUBCOMMAND_CASES[name]
    args = _analyse_parser().parse_args(argv)

    with patch(target) as mock_analysis:
        run(args)

    mock_analysis.assert_called_once()


def test_scaling_runtime_dispatch():
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
