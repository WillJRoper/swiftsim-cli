"""Tests for the output_times module."""

import argparse

import numpy as np
import pytest

from swiftsim_cli.modes.output_times import (
    _generate_output_list_no_cosmo,
    _get_out_list_log_scale_factor,
    _get_out_list_scale_factor,
    _get_out_list_time,
    _get_out_list_z,
    add_arguments,
    generate_output_list,
    run,
    write_output_list,
)


def test_get_out_list_z():
    """Test generating output list with redshift."""
    # Test with positive delta
    times = _get_out_list_z(1.0, 0.5, 0.0)
    assert np.allclose(times, np.array([1.0, 0.5, 0.0]))

    # Test with negative delta
    times = _get_out_list_z(1.0, -0.5, 0.0)
    assert np.allclose(times, np.array([1.0, 0.5, 0.0]))


def test_get_out_list_time():
    """Test generating output list with time."""
    # Test with positive delta
    times = _get_out_list_time(0.0, 0.5, 1.0)
    assert np.allclose(times, np.array([0.0, 0.5, 1.0]))

    # Test with negative delta
    times = _get_out_list_time(0.0, -0.5, 1.0)
    assert np.allclose(times, np.array([0.0, 0.5, 1.0]))


def test_get_out_list_scale_factor():
    """Test generating output list with scale factor."""
    # Test with positive delta
    times = _get_out_list_scale_factor(0.1, 0.1, 0.3)
    print(f"Times: {times}")
    assert np.allclose(times, np.array([0.1, 0.2, 0.3]))

    # Test with negative delta
    times = _get_out_list_scale_factor(0.1, -0.1, 0.3)
    print(f"Times: {times}")
    assert np.allclose(times, np.array([0.1, 0.2, 0.3]))


def test_get_out_list_log_scale_factor():
    """Test generating output list with log scale factor."""
    # Test with positive delta
    times = _get_out_list_log_scale_factor(0.1, 0.1, 0.3)
    expected = 10 ** np.arange(np.log10(0.1), np.log10(0.3) + 1e-9, 0.1)
    print(f"Times: {times}")
    print(f"Expected: {expected}")
    assert np.allclose(times, expected)

    # Test with negative delta
    times = _get_out_list_log_scale_factor(0.1, -0.1, 0.3)
    expected = 10 ** np.arange(np.log10(0.1), np.log10(0.3) + 1e-9, 0.1)
    print(f"Times: {times}")
    print(f"Expected: {expected}")
    assert np.allclose(times, expected)


def test_write_output_list(tmp_path):
    """Test writing output list to file."""
    # Create a temporary output file
    out_file = tmp_path / "output_list.txt"

    # Test with redshift (mixed)
    snapshot_times = np.array([1.0, 0.5, 0.0])
    snipshot_times = np.array([0.75, 0.25])
    write_output_list(out_file, snapshot_times, snipshot_times, doing_z=True)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert lines[0] == "# Redshift, Select Output\n"
    assert lines[1] == "1.0, Snapshot\n"
    assert lines[2] == "0.75, Snipshot\n"

    # Test with time (mixed)
    snapshot_times = np.array([0.0, 0.5, 1.0])
    snipshot_times = np.array([0.25, 0.75])
    write_output_list(
        out_file, snapshot_times, snipshot_times, doing_time=True
    )
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert lines[0] == "# Time, Select Output\n"
    assert lines[1] == "0.0, Snapshot\n"

    # Test with scale factor (mixed)
    snapshot_times = np.array([0.1, 0.2, 0.3])
    snipshot_times = np.array([0.15, 0.25])
    write_output_list(
        out_file, snapshot_times, snipshot_times, doing_scale_factor=True
    )
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert lines[0] == "# Scale Factor, Select Output\n"
    assert lines[1] == "0.1, Snapshot\n"

    # Test with only snapshots (single column)
    snapshot_times = np.array([0.1, 0.2, 0.3])
    snipshot_times = np.array([])
    write_output_list(
        out_file, snapshot_times, snipshot_times, doing_scale_factor=True
    )
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert lines[0] == "# Scale Factor\n"
    assert lines[1] == "0.1\n"
    assert lines[2] == "0.2\n"
    assert lines[3] == "0.3\n"


def test_write_output_list_error(tmp_path):
    """Test error when no output type is specified."""
    out_file = tmp_path / "output_list.txt"
    snapshot_times = np.array([1.0, 0.5, 0.0])
    snipshot_times = np.array([])

    # Should raise ValueError when none of doing_z, doing_time,
    # doing_scale_factor is True
    with pytest.raises(ValueError, match="something went wrong"):
        write_output_list(out_file, snapshot_times, snipshot_times)


def test_generate_output_list_no_cosmo(tmp_path):
    """Test generating output list without cosmology."""
    # Create a temporary output file
    out_file = tmp_path / "output_list.txt"

    # Test with redshift
    args_redshift = {
        "out": out_file,
        "first_snap_z": 1.0,
        "delta_z": 0.5,
        "final_snap_z": 0.0,
    }
    _generate_output_list_no_cosmo(args_redshift)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert lines[0] == "# Redshift\n"
    assert lines[1] == "1.0\n"
    assert lines[2] == "0.5\n"
    assert lines[3] == "0.0\n"

    # Test with time
    args_time = {
        "out": out_file,
        "first_snap_time": 0.0,
        "delta_time": 0.5,
        "final_snap_time": 1.0,
    }
    _generate_output_list_no_cosmo(args_time)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert lines[0] == "# Time\n"
    assert lines[1] == "0.0\n"
    assert lines[2] == "0.5\n"
    assert lines[3] == "1.0\n"

    # Test with scale factor
    args_scale_factor = {
        "out": out_file,
        "first_snap_scale_factor": 0.1,
        "delta_scale_factor": 0.1,
        "final_snap_scale_factor": 0.3,
    }
    _generate_output_list_no_cosmo(args_scale_factor)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert lines[0] == "# Scale Factor\n"
    assert lines[1] == "0.1\n"
    assert lines[2] == "0.2\n"
    assert lines[3] == "0.3\n"

    # Test with log scale factor
    args_log_scale_factor = {
        "out": out_file,
        "first_snap_scale_factor": 0.1,
        "delta_log_scale_factor": 0.1,
        "final_snap_scale_factor": 0.3,
    }
    _generate_output_list_no_cosmo(args_log_scale_factor)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert lines[0] == "# Scale Factor\n"
    assert np.isclose(float(lines[1].strip()), 0.1)
    assert np.isclose(float(lines[2].strip()), 10 ** (np.log10(0.1) + 0.1))
    assert np.isclose(float(lines[3].strip()), 10 ** (np.log10(0.1) + 0.2))


def test_generate_output_list_with_cosmo(mocker, tmp_path):
    """Test generate_output_list when cosmology is available."""
    mock_cosmo = mocker.Mock()
    mocker.patch(
        "swiftsim_cli.modes.output_times.get_cosmology",
        return_value=mock_cosmo,
    )
    mock_generate = mocker.patch(
        "swiftsim_cli.modes.output_times._generate_output_list_with_cosmo"
    )

    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_z": 10.0,
        "final_snap_z": 0.0,
        "delta_z": 1.0,
        "select": 1,
    }

    generate_output_list(args)

    # Should call the with_cosmo version
    mock_generate.assert_called_once_with(args, mock_cosmo)


def test_generate_output_list_without_cosmo(mocker, tmp_path):
    """Test generate_output_list when cosmology is not available."""
    mocker.patch(
        "swiftsim_cli.modes.output_times.get_cosmology", return_value=None
    )
    mock_generate = mocker.patch(
        "swiftsim_cli.modes.output_times._generate_output_list_no_cosmo"
    )

    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_scale_factor": 0.1,
        "final_snap_scale_factor": 1.0,
        "delta_scale_factor": 0.1,
        "select": 1,
    }

    generate_output_list(args)

    # Should call the no_cosmo version
    mock_generate.assert_called_once_with(args)


def test_generate_output_list_no_cosmo_error_no_first_snap(tmp_path):
    """Test error when no first snapshot is specified."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "delta_z": 1.0,
        "final_snap_z": 0.0,
    }
    with pytest.raises(
        ValueError, match="You must specify the first snapshot"
    ):
        _generate_output_list_no_cosmo(args)


def test_generate_output_list_no_cosmo_error_no_delta(tmp_path):
    """Test error when no delta is specified."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_z": 10.0,
        "final_snap_z": 0.0,
    }
    with pytest.raises(ValueError, match="You must specify the delta"):
        _generate_output_list_no_cosmo(args)


def test_generate_output_list_no_cosmo_error_z_without_delta_z(tmp_path):
    """Test error when first snap is z but delta_z not specified."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_z": 10.0,
        "delta_time": 1.0,
        "final_snap_z": 0.0,
    }
    with pytest.raises(
        ValueError,
        match="you must also specify the delta in terms of redshift",
    ):
        _generate_output_list_no_cosmo(args)


def test_generate_output_list_no_cosmo_error_time_without_delta_time(tmp_path):
    """Test error when first snap is time but delta_time not specified."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_time": 0.0,
        "delta_z": 1.0,
        "final_snap_time": 1.0,
    }
    with pytest.raises(
        ValueError, match="you must also specify the delta in terms of time"
    ):
        _generate_output_list_no_cosmo(args)


def test_generate_output_list_no_cosmo_error_scale_factor_without_delta(
    tmp_path,
):
    """Test error when first snap is scale factor but no delta specified."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_scale_factor": 0.1,
        "delta_z": 1.0,
        "final_snap_scale_factor": 1.0,
    }
    with pytest.raises(
        ValueError,
        match="you must also specify the delta in terms of scale factor",
    ):
        _generate_output_list_no_cosmo(args)


def test_generate_output_list_no_cosmo_error_no_final_snap(tmp_path):
    """Test error when no final snapshot is specified."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_z": 10.0,
        "delta_z": 1.0,
    }
    with pytest.raises(
        ValueError, match="You must specify the final snapshot"
    ):
        _generate_output_list_no_cosmo(args)


def test_generate_output_list_no_cosmo_snipshot_z_mismatch(tmp_path):
    """Test error when snipshot delta type doesn't match snapshot type."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_z": 10.0,
        "delta_z": 1.0,
        "final_snap_z": 0.0,
        "snipshot_delta_time": 0.5,
    }
    with pytest.raises(
        ValueError, match="Snipshot delta must be given in terms of redshift"
    ):
        _generate_output_list_no_cosmo(args)


def test_generate_output_list_no_cosmo_snipshot_time_mismatch(tmp_path):
    """Test error when snipshot delta type doesn't match (time)."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_time": 0.0,
        "delta_time": 0.5,
        "final_snap_time": 1.0,
        "snipshot_delta_z": 0.25,
    }
    with pytest.raises(
        ValueError, match="Snipshot delta must be given in terms of time"
    ):
        _generate_output_list_no_cosmo(args)


def test_generate_output_list_no_cosmo_snipshot_scale_factor_mismatch(
    tmp_path,
):
    """Test error when snipshot delta type doesn't match for scale factor."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_scale_factor": 0.1,
        "delta_scale_factor": 0.1,
        "final_snap_scale_factor": 0.3,
        "snipshot_delta_z": 0.05,
    }
    with pytest.raises(
        ValueError,
        match="Snipshot delta must be given in terms of scale factor",
    ):
        _generate_output_list_no_cosmo(args)


def test_generate_output_list_no_cosmo_with_snipshots(tmp_path):
    """Test generating output list with snipshots."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_z": 2.0,
        "delta_z": 1.0,
        "final_snap_z": 0.0,
        "snipshot_delta_z": 0.5,
    }
    _generate_output_list_no_cosmo(args)

    # Check that file was created
    assert out_file.exists()

    # Check contents
    with open(out_file, "r") as f:
        lines = f.readlines()

    # Should have header + 3 snapshots + 2 snipshots = 6 lines
    assert len(lines) == 6
    assert "# Redshift, Select Output" in lines[0]
    # Verify we have both snapshots and snipshots
    content = "".join(lines)
    assert "Snapshot" in content
    assert "Snipshot" in content


def test_generate_output_list_no_cosmo_time_with_snipshots(tmp_path):
    """Test generating output list with time and snipshots."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_time": 0.0,
        "delta_time": 0.5,
        "final_snap_time": 1.5,
        "snipshot_delta_time": 0.25,
    }
    _generate_output_list_no_cosmo(args)

    # Check that file was created
    assert out_file.exists()

    # Check contents
    with open(out_file, "r") as f:
        lines = f.readlines()

    assert "# Time, Select Output" in lines[0]
    # Verify we have both snapshots and snipshots
    content = "".join(lines)
    assert "Snapshot" in content
    assert "Snipshot" in content


def test_generate_output_list_no_cosmo_log_scale_factor(tmp_path):
    """Test generating output list with logarithmic scale factor."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_scale_factor": 0.1,
        "delta_log_scale_factor": 0.1,
        "final_snap_scale_factor": 0.3,
    }
    _generate_output_list_no_cosmo(args)

    # Check that file was created
    assert out_file.exists()

    # Check contents
    with open(out_file, "r") as f:
        lines = f.readlines()

    assert "# Scale Factor\n" == lines[0]
    # Check values (approximate due to floating point)
    assert np.isclose(float(lines[1].strip()), 0.1)
    assert np.isclose(float(lines[2].strip()), 10 ** (np.log10(0.1) + 0.1))
    assert np.isclose(float(lines[3].strip()), 10 ** (np.log10(0.1) + 0.2))


def test_generate_output_list_no_cosmo_scale_factor_with_snipshots(tmp_path):
    """Test generating output list with scale factor and snipshots."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_scale_factor": 0.1,
        "delta_scale_factor": 0.1,
        "final_snap_scale_factor": 0.3,
        "snipshot_delta_log_scale_factor": 0.05,
    }
    _generate_output_list_no_cosmo(args)

    # Check that file was created
    assert out_file.exists()

    # Check contents
    with open(out_file, "r") as f:
        lines = f.readlines()

    assert "# Scale Factor, Select Output" in lines[0]
    # Verify we have both snapshots and snipshots
    content = "".join(lines)
    assert "Snapshot" in content
    assert "Snipshot" in content


def test_add_arguments():
    """Test that add_arguments adds all output-times arguments."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)

    # Test with defaults
    args = parser.parse_args([])
    assert args.out == "output_list.txt"
    assert args.params is None
    assert args.first_snap_z is None

    # Test with some arguments
    args = parser.parse_args(["--out", "test.txt", "--first-snap-z", "10.0"])
    assert args.out == "test.txt"
    assert args.first_snap_z == 10.0


def test_run(mocker, tmp_path):
    """Test that run() calls generate_output_list."""
    from unittest.mock import Mock

    mock_generate = mocker.patch(
        "swiftsim_cli.modes.output_times.generate_output_list"
    )

    args = Mock()
    args.out = tmp_path / "output.txt"
    args.first_snap_z = 10.0
    args.final_snap_z = 0.0
    args.delta_z = 1.0
    args.select = 1

    run(args)

    # Should call generate_output_list with vars(args)
    mock_generate.assert_called_once()
    call_args = mock_generate.call_args[0][0]
    assert isinstance(call_args, dict)
