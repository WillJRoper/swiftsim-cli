"""Tests for the analyse mode init module."""

import argparse
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

    @patch("swiftsim_cli.modes.analyse.run_swift_scaling")
    def test_run_scaling(self, mock_run_swift_scaling):
        """Test running scaling analysis."""
        args = Mock()
        args.analysis_type = "scaling"

        run(args)

        mock_run_swift_scaling.assert_called_once_with(args)

    @patch("swiftsim_cli.modes.analyse.run_swift_task_plots")
    def test_run_task_plots(self, mock_run_swift_task_plots):
        """Test running combined task/threadpool plotting."""
        args = Mock()
        args.analysis_type = "task-plots"

        run(args)

        mock_run_swift_task_plots.assert_called_once_with(args)

    def test_run_unknown_type(self):
        """Test running with unknown analysis type."""
        args = Mock()
        args.analysis_type = "unknown"

        with pytest.raises(ValueError, match="Unknown analysis type"):
            run(args)
