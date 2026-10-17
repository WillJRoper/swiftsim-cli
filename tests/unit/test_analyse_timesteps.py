"""Tests for timestep analysis plotting configuration."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from swiftsim_cli.modes.analyse.timesteps import (
    _get_x_axis_label,
    analyse_timestep_files,
    run_timestep,
)


def test_get_x_axis_label_for_standard_columns():
    """Standard timestep columns get meaningful axis labels."""
    assert _get_x_axis_label(1) == "Time [Internal Units]"
    assert _get_x_axis_label(2) == "Scale factor"


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
        match_runtimes=True,
        nthreads=[4],
    )

    run_timestep(args)

    mock_analyse_timestep_files.assert_called_once_with(
        ["timesteps.txt"],
        ["run"],
        output_path="plot_dir",
        prefix="test",
        show_plot=False,
        time_column=1,
        match_runtimes=True,
        nthreads=[4],
    )


def test_time_column_must_be_non_negative():
    """Negative x-axis column indices are rejected."""
    with pytest.raises(ValueError, match="time_column must be non-negative"):
        analyse_timestep_files([], [], time_column=-1)


def test_nthreads_must_match_files():
    """Thread counts are provided once per input file."""
    with pytest.raises(
        ValueError, match="Number of files and nthreads entries must match"
    ):
        analyse_timestep_files(["a.txt"], ["run"], nthreads=[1, 2])


def test_nthreads_must_be_positive():
    """Thread counts must be positive integers."""
    with pytest.raises(
        ValueError, match="nthreads entries must be positive integers"
    ):
        analyse_timestep_files(["a.txt"], ["run"], nthreads=[0])


def _make_timestep_line(time_value: float, wallclock_ms: float) -> str:
    """Create one valid timestep data line with 15 numeric columns."""
    parts = [0.0] * 15
    parts[2] = time_value
    parts[12] = wallclock_ms
    parts[14] = wallclock_ms / 10.0
    return "  0 " + " ".join(str(value) for value in parts[1:]) + "\n"


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.tight_layout")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.subplots")
def test_match_runtimes_trims_to_shortest_dataset(
    mock_subplots,
    mock_tight_layout,
    mock_save_figure,
    mock_create_output_path,
    tmp_path: Path,
):
    """When requested, all series stop at the shortest runtime."""
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text(
        "# header\n"
        + _make_timestep_line(1.0, 1000.0)
        + _make_timestep_line(2.0, 1000.0)
        + _make_timestep_line(3.0, 1000.0),
        encoding="utf-8",
    )
    file_b.write_text(
        "# header\n"
        + _make_timestep_line(1.0, 1000.0)
        + _make_timestep_line(2.0, 1000.0),
        encoding="utf-8",
    )

    ax1 = Mock()
    ax2 = Mock()
    fig = Mock()
    mock_subplots.return_value = (fig, (ax1, ax2))
    mock_create_output_path.return_value = tmp_path / "plot.png"

    analyse_timestep_files(
        [str(file_a), str(file_b)],
        ["run a", "run b"],
        show_plot=False,
        match_runtimes=True,
    )

    first_wallclock_x = np.asarray(ax1.plot.call_args_list[0].args[0])
    second_wallclock_x = np.asarray(ax1.plot.call_args_list[2].args[0])

    assert np.array_equal(first_wallclock_x, np.array([1.0, 2.0]))
    assert np.array_equal(second_wallclock_x, np.array([1.0, 2.0]))


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.tight_layout")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.subplots")
def test_shared_extent_reports_speedup_without_trimming(
    mock_subplots,
    mock_tight_layout,
    mock_save_figure,
    mock_create_output_path,
    tmp_path: Path,
    capsys,
):
    """Runs report speedup at their shared extent without plot trimming."""
    slow_file = tmp_path / "slow.txt"
    fast_file = tmp_path / "fast.txt"
    slow_file.write_text(
        _make_timestep_line(1.0, 5000.0) + _make_timestep_line(2.0, 5000.0),
        encoding="utf-8",
    )
    fast_file.write_text(
        _make_timestep_line(1.0, 4000.0) + _make_timestep_line(2.0, 4000.0),
        encoding="utf-8",
    )
    mock_subplots.return_value = (Mock(), (Mock(), Mock()))
    mock_create_output_path.return_value = tmp_path / "plot.png"

    analyse_timestep_files(
        [str(slow_file), str(fast_file)],
        ["slow", "fast"],
        show_plot=False,
    )

    output = capsys.readouterr().out
    assert "RUNTIME COMPARISON AT SHARED Scale factor 2" in output
    assert "Speedup vs slowest" in output
    assert "fast" in output
    assert "1.25x" in output
    assert "25.0" in output
    assert "20.0" in output
    assert output.index("| fast |") < output.index("| slow |")


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.tight_layout")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.subplots")
def test_timestep_axis_label_is_inferred_from_header(
    mock_subplots,
    mock_tight_layout,
    mock_save_figure,
    mock_create_output_path,
    tmp_path: Path,
):
    """The x-axis label follows the timestep table header when available."""
    timestep_file = tmp_path / "timesteps.txt"
    timestep_file.write_text(
        "#   Step           Time Scale-factor     Redshift      Time-step "
        "Time-bins      Updates    g-Updates    s-Updates "
        "sink-Updates    b-Updates  Wall-clock time [ms]  Props    "
        "Dead time [ms]\n" + _make_timestep_line(1.0, 1000.0),
        encoding="utf-8",
    )

    ax1 = Mock()
    ax2 = Mock()
    fig = Mock()
    mock_subplots.return_value = (fig, (ax1, ax2))
    mock_create_output_path.return_value = tmp_path / "plot.png"

    analyse_timestep_files(
        [str(timestep_file)],
        ["run"],
        show_plot=False,
        time_column=2,
    )

    ax2.set_xlabel.assert_called_once_with("Scale factor")


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.tight_layout")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.subplots")
def test_timestep_axis_label_falls_back_without_header(
    mock_subplots,
    mock_tight_layout,
    mock_save_figure,
    mock_create_output_path,
    tmp_path: Path,
):
    """Files without a header keep the generic column-based label."""
    timestep_file = tmp_path / "timesteps.txt"
    timestep_file.write_text(
        _make_timestep_line(1.0, 1000.0),
        encoding="utf-8",
    )

    ax1 = Mock()
    ax2 = Mock()
    fig = Mock()
    mock_subplots.return_value = (fig, (ax1, ax2))
    mock_create_output_path.return_value = tmp_path / "plot.png"

    analyse_timestep_files(
        [str(timestep_file)],
        ["run"],
        show_plot=False,
        time_column=4,
    )

    ax2.set_xlabel.assert_called_once_with("Column 5")


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.tight_layout")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.subplots")
def test_nthreads_converts_main_axis_to_cpu_time(
    mock_subplots,
    mock_tight_layout,
    mock_save_figure,
    mock_create_output_path,
    tmp_path: Path,
):
    """Thread counts scale plotted cumulative times into CPU time."""
    timestep_file = tmp_path / "timesteps.txt"
    timestep_file.write_text(
        _make_timestep_line(1.0, 1000.0),
        encoding="utf-8",
    )

    ax1 = Mock()
    ax2 = Mock()
    fig = Mock()
    mock_subplots.return_value = (fig, (ax1, ax2))
    mock_create_output_path.return_value = tmp_path / "plot.png"

    analyse_timestep_files(
        [str(timestep_file)],
        ["run"],
        show_plot=False,
        nthreads=[4],
    )

    wallclock_y = np.asarray(ax1.plot.call_args_list[0].args[1])
    deadtime_y = np.asarray(ax1.plot.call_args_list[1].args[1])

    assert np.array_equal(wallclock_y, np.array([4.0 / 3600.0]))
    assert np.array_equal(deadtime_y, np.array([0.4 / 3600.0]))
    ax1.set_ylabel.assert_called_once_with("CPU hrs")


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.tight_layout")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.subplots")
def test_non_monotonic_x_skips_runtime_comparison(
    mock_subplots,
    mock_tight_layout,
    mock_save_figure,
    mock_create_output_path,
    tmp_path: Path,
    capsys,
):
    """A decreasing x column (e.g. redshift) must not be interpolated."""
    files = []
    for name, wallclock in (("slow.txt", 5000.0), ("fast.txt", 4000.0)):
        path = tmp_path / name
        path.write_text(
            _make_timestep_line(3.0, wallclock)
            + _make_timestep_line(2.0, wallclock)
            + _make_timestep_line(1.0, wallclock),
            encoding="utf-8",
        )
        files.append(str(path))
    mock_subplots.return_value = (Mock(), (Mock(), Mock()))
    mock_create_output_path.return_value = tmp_path / "plot.png"

    analyse_timestep_files(files, ["slow", "fast"], show_plot=False)

    output = capsys.readouterr().out
    assert "not monotonically increasing" in output
    assert "RUNTIME COMPARISON" not in output


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.tight_layout")
@patch("swiftsim_cli.modes.analyse.timesteps.plt.subplots")
def test_older_13_column_table_uses_header_columns(
    mock_subplots,
    mock_tight_layout,
    mock_save_figure,
    mock_create_output_path,
    tmp_path: Path,
):
    """Tables without sink-/b-Updates locate wall-clock from the header."""
    timestep_file = tmp_path / "timesteps.txt"
    timestep_file.write_text(
        "# Step Properties: Rebuild=1, Redistribute=2\n"
        "#   Step  Time Scale-factor Redshift Time-step Time-bins Updates "
        "g-Updates s-Updates Wall-clock time [ms] Props Dead time [ms]\n"
        "  0 0.0 0.5 1.0 0.0 1 56 0 0 0 3600000.0 265 360000.0\n"
        "  1 0.1 0.6 0.6 0.1 40 40 0 0 0 3600000.0 1 0.0\n",
        encoding="utf-8",
    )
    ax1 = Mock()
    ax2 = Mock()
    mock_subplots.return_value = (Mock(), (ax1, ax2))
    mock_create_output_path.return_value = tmp_path / "plot.png"

    analyse_timestep_files([str(timestep_file)], ["run"], show_plot=False)

    x = np.asarray(ax1.plot.call_args_list[0].args[0])
    wallclock_y = np.asarray(ax1.plot.call_args_list[0].args[1])
    deadtime_y = np.asarray(ax1.plot.call_args_list[1].args[1])
    assert np.array_equal(x, np.array([0.5, 0.6]))
    assert np.allclose(wallclock_y, [1.0, 2.0])
    assert np.allclose(deadtime_y, [0.1, 0.1])
    ax2.set_xlabel.assert_called_once_with("Scale factor")
    mock_save_figure.assert_called_once()


def test_no_parsable_rows_raises(tmp_path: Path):
    """A file with no step rows is an error, not an empty plot."""
    timestep_file = tmp_path / "timesteps.txt"
    timestep_file.write_text("# just a comment\nnot a row\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No step-table rows"):
        analyse_timestep_files([str(timestep_file)], ["run"], show_plot=False)


def test_time_column_out_of_range_raises(tmp_path: Path):
    """Selecting a column the table does not have is a clear error."""
    timestep_file = tmp_path / "timesteps.txt"
    timestep_file.write_text(_make_timestep_line(1.0, 1.0), encoding="utf-8")

    with pytest.raises(ValueError, match="out of range"):
        analyse_timestep_files(
            [str(timestep_file)], ["run"], show_plot=False, time_column=15
        )
