"""Tests for timestep analysis plotting configuration."""

from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.modes.analyse.timesteps import (
    _get_x_axis_label,
    run_timestep,
)


def test_get_x_axis_label_for_standard_columns():
    """Standard timestep columns get meaningful axis labels."""
    assert _get_x_axis_label(1) == "Scale factor"
    assert _get_x_axis_label(2) == "Time [Internal Units]"


def test_get_x_axis_label_for_other_columns():
    """Non-standard timestep columns fall back to a generic label."""
    assert _get_x_axis_label(4) == "Column 5"


@patch("swiftsim_cli.modes.analyse.timesteps.analyse_timestep_files")
def test_run_timestep_passes_time_column(mock_analyse_timestep_files):
    """CLI arguments are forwarded to the timestep analysis function."""
    args = Mock(
        files=["timesteps.txt"],
        labels=["run"],
        output_path="plot_dir",
        prefix="test",
        show_plot=False,
        time_column=1,
    )

    run_timestep(args)

    mock_analyse_timestep_files.assert_called_once_with(
        ["timesteps.txt"],
        ["run"],
        output_path="plot_dir",
        prefix="test",
        show_plot=False,
        time_column=1,
    )


def test_time_column_must_be_non_negative():
    """Negative x-axis column indices are rejected."""
    from swiftsim_cli.modes.analyse.timesteps import analyse_timestep_files

    with pytest.raises(ValueError, match="time_column must be non-negative"):
        analyse_timestep_files([], [], time_column=-1)
