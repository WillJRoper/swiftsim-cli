"""Tests for timestep analysis plotting configuration."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from swiftsim_cli.modes.analyse.timesteps import (
    analyse_timestep_files,
    run_timestep,
)


@patch("swiftsim_cli.modes.analyse.timesteps.analyse_timestep_files")
def test_run_timestep_passes_time_column(mock_analyse_timestep_files):
    """CLI arguments are forwarded to the timestep analysis function."""
    args = Mock(
        files=["timesteps.txt"],
        labels=["run"],
        out="plot_dir",
        prefix="test",
        show=False,
        time_column="time",
        match_runtimes=True,
        threads_per_run=[4],
        steps=[2, 5],
    )

    run_timestep(args)

    mock_analyse_timestep_files.assert_called_once_with(
        ["timesteps.txt"],
        ["run"],
        output_path="plot_dir",
        prefix="test",
        show_plot=False,
        time_column="time",
        match_runtimes=True,
        nthreads=[4],
        step_range=[2, 5],
    )


def test_time_column_must_be_known():
    """Unknown x-axis columns are rejected."""
    with pytest.raises(ValueError, match="time_column must be one of"):
        analyse_timestep_files([], [], time_column="Column 5")


def test_nthreads_must_match_files():
    """Thread counts are provided once per input file."""
    with pytest.raises(
        ValueError,
        match="Number of files and cores-per-run entries must match",
    ):
        analyse_timestep_files(["a.txt"], ["run"], nthreads=[1, 2])


def test_nthreads_must_be_positive():
    """Thread counts must be positive integers."""
    with pytest.raises(
        ValueError, match="cores-per-run entries must be positive integers"
    ):
        analyse_timestep_files(["a.txt"], ["run"], nthreads=[0])


def _make_timestep_line(
    time_value: float, wallclock_ms: float, step: int = 0
) -> str:
    """Create one valid timestep data line with 15 numeric columns.

    ``time_value`` fills both the Time and Scale-factor columns.
    """
    parts = [0.0] * 15
    parts[1] = time_value
    parts[2] = time_value
    parts[12] = wallclock_ms
    parts[14] = wallclock_ms / 10.0
    return f"  {step} " + " ".join(str(value) for value in parts[1:]) + "\n"


def _table(*rows: tuple[float, float]) -> str:
    """Consecutive step rows (from step 0) for ``(x, wallclock_ms)`` pairs."""
    return "".join(
        _make_timestep_line(x, wallclock, step)
        for step, (x, wallclock) in enumerate(rows)
    )


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.subplots")
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
        "# header\n" + _table((1.0, 1000.0), (2.0, 1000.0), (3.0, 1000.0)),
        encoding="utf-8",
    )
    file_b.write_text(
        "# header\n" + _table((1.0, 1000.0), (2.0, 1000.0)),
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
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.subplots")
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
        _table((1.0, 5000.0), (2.0, 5000.0)),
        encoding="utf-8",
    )
    fast_file.write_text(
        _table((1.0, 4000.0), (2.0, 4000.0)),
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
    assert "20.0" in output  # less runtime [%]
    assert "Faster" not in output
    assert output.index("| fast |") < output.index("| slow |")


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.subplots")
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
        time_column="scale-factor",
    )

    ax2.set_xlabel.assert_called_once_with("Scale factor")


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.subplots")
def test_timestep_axis_label_for_redshift(
    mock_subplots,
    mock_tight_layout,
    mock_save_figure,
    mock_create_output_path,
    tmp_path: Path,
):
    """Files without a header still label the selected column."""
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
        time_column="redshift",
    )

    ax2.set_xlabel.assert_called_once_with("Redshift")


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.subplots")
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
    ax1.set_ylabel.assert_called_once_with("Cumulative core time [core-hrs]")


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.subplots")
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
            _table((3.0, wallclock), (2.0, wallclock), (1.0, wallclock)),
            encoding="utf-8",
        )
        files.append(str(path))
    mock_subplots.return_value = (Mock(), (Mock(), Mock()))
    mock_create_output_path.return_value = tmp_path / "plot.png"

    analyse_timestep_files(files, ["slow", "fast"], show_plot=False)

    captured = capsys.readouterr()
    assert "warning: x-axis column is not monotonically" in captured.err
    assert "RUNTIME COMPARISON" not in captured.out


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.subplots")
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


def _numbered_line(step: int, wallclock_ms: float) -> str:
    """A step row for step number ``step``."""
    return _make_timestep_line(float(step), wallclock_ms, step)


@patch("swiftsim_cli.modes.analyse.timesteps.create_output_path")
@patch("swiftsim_cli.modes.analyse.timesteps.save_figure")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.subplots")
def test_step_range_filters_rows(
    mock_subplots,
    mock_tight_layout,
    mock_save_figure,
    mock_create_output_path,
    tmp_path: Path,
):
    """--steps keeps only the rows of the requested steps."""
    timestep_file = tmp_path / "timesteps.txt"
    timestep_file.write_text(
        "".join(_numbered_line(step, 1000.0) for step in range(1, 6)),
        encoding="utf-8",
    )
    ax1 = Mock()
    mock_subplots.return_value = (Mock(), (ax1, Mock()))
    mock_create_output_path.return_value = tmp_path / "plot.png"

    analyse_timestep_files(
        [str(timestep_file)],
        ["run"],
        time_column="step",
        step_range=(2, 4),
    )

    x = np.asarray(ax1.plot.call_args_list[0].args[0])
    assert np.array_equal(x, [2.0, 3.0, 4.0])


def test_step_range_without_rows_raises(tmp_path: Path):
    """A step range that selects nothing names the file and range."""
    timestep_file = tmp_path / "timesteps.txt"
    timestep_file.write_text(_numbered_line(1, 1.0), encoding="utf-8")

    with pytest.raises(ValueError, match="in steps 5-9"):
        analyse_timestep_files(
            [str(timestep_file)], ["run"], step_range=(5, 9)
        )


# ---------------------------------------------------------------------------
# x-axis choice, restarts, gaps and malformed rows
# ---------------------------------------------------------------------------

_HEADER = (
    "#   Step           Time Scale-factor     Redshift      Time-step "
    "Time-bins      Updates    g-Updates    s-Updates "
    "sink-Updates    b-Updates  Wall-clock time [ms]  Props    "
    "Dead time [ms]\n"
)


def _row(step: int, time: float, a: float, wall: float, dead=0.0) -> str:
    """A 15-column step row with the given time, scale factor and times."""
    z = 1.0 / a - 1.0
    return (
        f"{step:8d} {time:.6e} {a:.7f} {z:.7f} 0.0 1 56 10 0 0 0 0 "
        f"{wall:.3f} 0 {dead:.3f}\n"
    )


def _run_plot(files, labels=None, **kwargs):
    """Run the analysis with mocked axes; return (ax1, ax2)."""
    ax1, ax2 = Mock(), Mock()
    with (
        patch("matplotlib.pyplot.subplots", return_value=(Mock(), (ax1, ax2))),
        patch("matplotlib.pyplot.tight_layout"),
        patch("swiftsim_cli.modes.analyse.timesteps.save_figure"),
    ):
        analyse_timestep_files(
            [str(f) for f in files],
            labels or [Path(f).stem for f in files],
            output_path=Path(files[0]).parent,
            **kwargs,
        )
    return ax1, ax2


def test_auto_axis_falls_back_to_time_for_non_cosmological_runs(
    tmp_path: Path, capsys
):
    """A constant scale factor (a=1) means time is plotted instead."""
    path = tmp_path / "sod.txt"
    path.write_text(
        _HEADER + "".join(_row(s, 0.1 * s, 1.0, 1000.0) for s in range(4))
    )

    ax1, ax2 = _run_plot([path])

    assert "using time" in capsys.readouterr().err
    np.testing.assert_allclose(
        ax1.plot.call_args_list[0].args[0], [0.0, 0.1, 0.2, 0.3]
    )
    ax2.set_xlabel.assert_called_once_with("Time [internal units]")


def test_auto_axis_uses_scale_factor_when_it_varies(tmp_path: Path):
    """Cosmological runs keep the scale factor on the x-axis."""
    path = tmp_path / "cosmo.txt"
    path.write_text(
        _HEADER + "".join(_row(s, s, 0.1 * (s + 1), 1000.0) for s in range(3))
    )

    _, ax2 = _run_plot([path])

    ax2.set_xlabel.assert_called_once_with("Scale factor")


def test_non_cosmological_speedup_compares_equal_work(tmp_path: Path, capsys):
    """Sod-like runs: same cost per unit time is 1.00x, not a fake win.

    The longer run took more steps; compared at the shared time both have
    spent the same wall-clock time.
    """
    short = tmp_path / "short.txt"
    long = tmp_path / "long.txt"
    short.write_text(
        _HEADER + "".join(_row(s, 0.1 * s, 1.0, 1000.0) for s in range(5))
    )
    long.write_text(
        _HEADER + "".join(_row(s, 0.1 * s, 1.0, 1000.0) for s in range(9))
    )

    _run_plot([short, long])

    out = capsys.readouterr().out
    assert "SHARED Time [internal units] 0.4" in out
    assert out.count("1.00x") == 2


def test_constant_x_column_refuses_comparison(tmp_path: Path, capsys):
    """An explicitly chosen constant column cannot be compared."""
    files = []
    for name, wall in (("a.txt", 1000.0), ("b.txt", 2000.0)):
        path = tmp_path / name
        path.write_text(
            _HEADER + "".join(_row(s, 0.1 * s, 1.0, wall) for s in range(3))
        )
        files.append(path)

    _run_plot(files, time_column="scale-factor")

    captured = capsys.readouterr()
    assert "x-axis column is constant" in captured.err
    assert "RUNTIME COMPARISON" not in captured.out


def test_restart_keeps_latest_occurrence_of_each_step(tmp_path: Path, capsys):
    """Rows repeated after a restart replace the earlier ones."""
    path = tmp_path / "restarted.txt"
    path.write_text(
        _HEADER
        + "".join(_row(s, s, 1.0, 1000.0) for s in range(4))
        # restarted from step 2: steps 2 and 3 run again, slower
        + _HEADER
        + "".join(_row(s, s, 1.0, 2000.0) for s in range(2, 5))
    )

    ax1, _ = _run_plot([path], time_column="step")

    assert "2 rows superseded by a restart" in capsys.readouterr().err
    np.testing.assert_array_equal(
        ax1.plot.call_args_list[0].args[0], [0, 1, 2, 3, 4]
    )
    hours = np.asarray(ax1.plot.call_args_list[0].args[1]) * 3600
    np.testing.assert_allclose(hours, [1, 2, 4, 6, 8])


def test_malformed_rows_are_counted(tmp_path: Path, capsys):
    """Truncated rows (e.g. a crash mid-write) are skipped with a count."""
    path = tmp_path / "truncated.txt"
    path.write_text(
        _HEADER
        + _row(0, 0.0, 1.0, 1000.0)
        + _row(1, 0.1, 1.0, 1000.0)[:30]
        + "\n"
        + _row(1, 0.1, 1.0, 1000.0)
    )

    _run_plot([path])

    assert "skipped 1 malformed step-table row" in capsys.readouterr().err


def test_step_gaps_are_reported_and_break_the_line(tmp_path: Path, capsys):
    """Missing steps are warned about and drawn as a break, not a line."""
    path = tmp_path / "gappy.txt"
    path.write_text(
        _HEADER + "".join(_row(s, 0.1 * s, 1.0, 1000.0) for s in (0, 1, 5, 6))
    )

    ax1, _ = _run_plot([path])

    assert "3 steps missing in 1 gap" in capsys.readouterr().err
    x = np.asarray(ax1.plot.call_args_list[0].args[0])
    assert x.size == 5 and np.isnan(x[2])


def test_dead_time_column_is_optional(tmp_path: Path, capsys):
    """Older tables without dead time still plot the wall-clock time."""
    path = tmp_path / "old.txt"
    path.write_text(
        "#   Step  Time Scale-factor Redshift Time-step Time-bins Updates "
        "g-Updates s-Updates Wall-clock time [ms] Props\n"
        "  0 0.0 1.0 0.0 0.0 1 56 100 0 0 12.5 25\n"
        "  1 0.1 1.0 0.0 0.1 46 46 100 0 0 10.0 1\n"
    )

    ax1, _ = _run_plot([path])

    assert "no 'Dead time [ms]' column" in capsys.readouterr().err
    assert ax1.plot.call_count == 1  # wall-clock only


def test_missing_wallclock_column_is_named(tmp_path: Path):
    """Only the missing required column is named in the error."""
    path = tmp_path / "odd.txt"
    path.write_text(
        "#   Step  Time Scale-factor Redshift Dead time [ms]\n"
        "  0 0.0 1.0 0.0 1.0\n"
    )

    with pytest.raises(ValueError, match=r"no 'Wall-clock time \[ms\]'"):
        _run_plot([path])


def test_output_file_name(tmp_path: Path):
    """The plot is timesteps_analysis/timesteps_analysis.png."""
    path = tmp_path / "run.txt"
    path.write_text(_HEADER + _row(0, 0.0, 1.0, 1.0) + _row(1, 1.0, 1.0, 1.0))

    analyse_timestep_files([str(path)], ["run"], output_path=tmp_path)

    assert (
        tmp_path / "timesteps_analysis" / "timesteps_analysis.png"
    ).exists()


def test_cli_defaults_and_cores_per_run():
    """--time-column defaults to auto; --threads-per-run is gone."""
    import argparse

    from swiftsim_cli.modes.analyse.timesteps import add_timestep_arguments

    parser = argparse.ArgumentParser()
    add_timestep_arguments(parser.add_subparsers())
    args = parser.parse_args(["timesteps", __file__])
    assert args.time_column == "auto"
    args = parser.parse_args(["timesteps", __file__, "--cores-per-run", "8"])
    assert args.threads_per_run == [8]
    with pytest.raises(SystemExit):
        parser.parse_args(["timesteps", __file__, "--threads-per-run", "8"])
