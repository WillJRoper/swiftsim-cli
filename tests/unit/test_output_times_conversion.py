"""Tests for the output_times conversion functionality."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from swiftsim_cli.modes.output_times import (
    _generate_output_list_no_cosmo,
    _generate_output_list_with_cosmo,
    add_arguments,
    unify_snapshot_times,
)

# --- Fixtures & Mocks ---


@pytest.fixture
def mock_cosmo(mocker):
    """Mock cosmology object."""
    cosmo = MagicMock()
    # Simple linear relationships for testing logic, not accuracy
    # z = 1/a - 1
    # t = 1 - z/10  (just a dummy linear relation)
    # We mock the conversion functions directly usually, but some internals use
    # cosmo.age

    mocker.patch("swiftsim_cli.cosmology.get_cosmology", return_value=cosmo)
    mocker.patch(
        "swiftsim_cli.modes.output_times.load_parameters",
        return_value={"Cosmology": {"a_begin": 0.01, "a_end": 1.0}},
    )
    return cosmo


@pytest.fixture
def mock_conversions(mocker):
    """Mock conversion functions to simple linear ops for predictability."""
    mocker.patch(
        "swiftsim_cli.modes.output_times.convert_time_to_redshift",
        side_effect=lambda t: 10.0 * (1.0 - t),
    )
    mocker.patch(
        "swiftsim_cli.modes.output_times.convert_redshift_to_time",
        side_effect=lambda z: 1.0 - z / 10.0,
    )
    mocker.patch(
        "swiftsim_cli.modes.output_times.convert_scale_factor_to_redshift",
        side_effect=lambda a: 1.0 / a - 1.0,
    )
    mocker.patch(
        "swiftsim_cli.modes.output_times.convert_redshift_to_scale_factor",
        side_effect=lambda z: 1.0 / (1.0 + z),
    )
    # Note: unify_snapshot_times imports these locally? No, it imports from
    # swiftsim_cli.cosmology
    # So we should patch where they are defined or used.
    # In output_times.py: from swiftsim_cli.cosmology import ...
    # So patching swiftsim_cli.modes.output_times.convert_... works.


# --- Tests ---


def test_add_arguments_includes_list_units():
    """Test that the list-units argument is added."""
    import argparse

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["--list-units", "redshift"])
    assert args.list_units == "redshift"


def test_no_cosmo_raises_on_list_units(tmp_path):
    """Test that providing list-units without cosmology raises ValueError."""
    out_file = tmp_path / "output.txt"
    args = {
        "out": out_file,
        "first_snap_z": 1.0,
        "delta_z": 0.5,
        "final_snap_z": 0.0,
        "list_units": "time",
    }
    with pytest.raises(
        ValueError, match="Cannot convert output units without cosmology"
    ):
        _generate_output_list_no_cosmo(args)


def test_unify_snapshot_times_doing_z(mock_conversions):
    """Test unify_snapshot_times when doing_z is True."""
    # Input z, should pass through
    first, final = unify_snapshot_times(
        first_snap_z=1.0, final_snap_z=0.0, doing_z=True
    )
    assert first == 1.0
    assert final == 0.0

    # Input time, should convert to z
    # t=1.0 -> z=0.0, t=0.5 -> z=5.0
    first, final = unify_snapshot_times(
        first_snap_time=0.5, final_snap_time=1.0, doing_z=True
    )
    assert np.isclose(first, 5.0)
    assert np.isclose(final, 0.0)

    # Input scale factor, should convert to z
    # a=0.5 -> z=1.0, a=1.0 -> z=0.0
    first, final = unify_snapshot_times(
        first_snap_scale_factor=0.5, final_snap_scale_factor=1.0, doing_z=True
    )
    assert np.isclose(first, 1.0)
    assert np.isclose(final, 0.0)


def test_unify_snapshot_times_doing_time(mock_conversions):
    """Test unify_snapshot_times when doing_time is True."""
    # Input time
    first, final = unify_snapshot_times(
        first_snap_time=0.0, final_snap_time=1.0, doing_time=True
    )
    assert first == 0.0
    assert final == 1.0

    # Input z -> time
    # z=5.0 -> t=0.5
    first, final = unify_snapshot_times(
        first_snap_z=5.0, final_snap_z=0.0, doing_time=True
    )
    assert np.isclose(first, 0.5)

    # Input a -> z -> time
    # a=0.5 -> z=1.0 -> t=0.9
    first, final = unify_snapshot_times(
        first_snap_scale_factor=0.5,
        final_snap_scale_factor=1.0,
        doing_time=True,
    )
    assert np.isclose(first, 0.9)  # 1 - 1.0/10 = 0.9


def test_unify_snapshot_times_doing_scale_factor(mock_conversions):
    """Test unify_snapshot_times when doing_scale_factor is True."""
    # Input a
    first, final = unify_snapshot_times(
        first_snap_scale_factor=0.1,
        final_snap_scale_factor=1.0,
        doing_scale_factor=True,
    )
    assert first == 0.1

    # Input z -> a
    # z=1.0 -> a=0.5
    first, final = unify_snapshot_times(
        first_snap_z=1.0, final_snap_z=0.0, doing_scale_factor=True
    )
    assert np.isclose(first, 0.5)

    # Input t -> z -> a
    # t=0.9 -> z=1.0 -> a=0.5
    first, final = unify_snapshot_times(
        first_snap_time=0.9, final_snap_time=1.0, doing_scale_factor=True
    )
    assert np.isclose(first, 0.5)


def test_unify_snapshot_times_errors():
    """Test error handling in unify_snapshot_times."""
    # No first snap
    with pytest.raises(
        ValueError, match="You must specify the first snapshot"
    ):
        unify_snapshot_times(final_snap_z=0.0, doing_z=True)

    # No final snap
    with pytest.raises(
        ValueError, match="You must specify the final snapshot"
    ):
        unify_snapshot_times(first_snap_z=1.0, doing_z=True)

    # No mode selected (shouldn't happen via CLI but possible direct call)
    with pytest.raises(ValueError, match="Found no valid snapshot type"):
        unify_snapshot_times(first_snap_z=1.0, final_snap_z=0.0)


def test_list_units_redshift_conversion(
    mock_cosmo, mock_conversions, tmp_path
):
    """Test converting output to redshift."""
    out_file = tmp_path / "out_z.txt"
    # Start in time: 0.9 -> z=1.0, 1.0 -> z=0.0
    args = {
        "out": out_file,
        "first_snap_time": 0.9,
        "final_snap_time": 1.0,
        "delta_time": 0.1,
        "list_units": "redshift",
    }
    _generate_output_list_with_cosmo(args, mock_cosmo)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert "# Redshift, Select Output" in lines[0]
    # Check first value is roughly 1.0
    assert np.isclose(float(lines[1].split(",")[0]), 1.0)


def test_list_units_scale_factor_conversion(
    mock_cosmo, mock_conversions, tmp_path
):
    """Test converting output to scale factor."""
    out_file = tmp_path / "out_a.txt"
    # Start in z: 1.0 -> a=0.5
    args = {
        "out": out_file,
        "first_snap_z": 1.0,
        "final_snap_z": 0.0,
        "delta_z": 1.0,
        "list_units": "scale-factor",
    }
    _generate_output_list_with_cosmo(args, mock_cosmo)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert "# Scale Factor, Select Output" in lines[0]
    # Check first value is 0.5
    assert np.isclose(float(lines[1].split(",")[0]), 0.5)


def test_list_units_time_conversion(mock_cosmo, mock_conversions, tmp_path):
    """Test converting output to time."""
    out_file = tmp_path / "out_t.txt"
    # Start in a: 0.5 -> z=1.0 -> t=0.9
    args = {
        "out": out_file,
        "first_snap_scale_factor": 0.5,
        "final_snap_scale_factor": 1.0,
        "delta_scale_factor": 0.5,
        "list_units": "time",
    }
    _generate_output_list_with_cosmo(args, mock_cosmo)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert "# Time, Select Output" in lines[0]
    assert np.isclose(float(lines[1].split(",")[0]), 0.9)


def test_snipshot_conversion_logic(mock_cosmo, mock_conversions, tmp_path):
    """Test that snipshots are also converted correctly."""
    out_file = tmp_path / "out_snip.txt"
    # Working in Z, converting to Time
    # Snap: z=1.0 -> t=0.9
    # Snip: z=0.5 -> t=0.95
    args = {
        "out": out_file,
        "first_snap_z": 1.0,
        "final_snap_z": 0.0,
        "delta_z": 1.0,
        "snipshot_delta_z": 0.5,
        "list_units": "time",
    }
    _generate_output_list_with_cosmo(args, mock_cosmo)
    with open(out_file, "r") as f:
        content = f.read()

    # We expect t=0.95 to be a snipshot
    # 0.95 -> Snipshot
    assert "0.95, Snipshot" in content


@pytest.mark.parametrize(
    "main_type,snip_type,main_arg,snip_arg",
    [
        # doing_z and snip_doing_time
        (
            "z",
            "time",
            {"first_snap_z": 1.0, "final_snap_z": 0.0, "delta_z": 1.0},
            {"snipshot_delta_time": 0.05},
        ),
        # doing_z and snip_doing_scale_factor
        (
            "z",
            "scale_factor",
            {"first_snap_z": 1.0, "final_snap_z": 0.0, "delta_z": 1.0},
            {"snipshot_delta_scale_factor": 0.1},
        ),
        # doing_time and snip_doing_z
        (
            "time",
            "z",
            {
                "first_snap_time": 0.9,
                "final_snap_time": 1.0,
                "delta_time": 0.1,
            },
            {"snipshot_delta_z": 0.5},
        ),
        # doing_time and snip_doing_scale_factor
        (
            "time",
            "scale_factor",
            {
                "first_snap_time": 0.9,
                "final_snap_time": 1.0,
                "delta_time": 0.1,
            },
            {"snipshot_delta_scale_factor": 0.1},
        ),
        # doing_scale_factor and snip_doing_z
        (
            "scale_factor",
            "z",
            {
                "first_snap_scale_factor": 0.5,
                "final_snap_scale_factor": 1.0,
                "delta_scale_factor": 0.5,
            },
            {"snipshot_delta_z": 0.1},
        ),
        # doing_scale_factor and snip_doing_time
        (
            "scale_factor",
            "time",
            {
                "first_snap_scale_factor": 0.5,
                "final_snap_scale_factor": 1.0,
                "delta_scale_factor": 0.5,
            },
            {"snipshot_delta_time": 0.01},
        ),
        # doing_log_scale_factor and snip_doing_scale_factor
        (
            "log_a",
            "a",
            {
                "first_snap_scale_factor": 0.1,
                "final_snap_scale_factor": 1.0,
                "delta_log_scale_factor": 0.1,
            },
            {"snipshot_delta_scale_factor": 0.1},
        ),
        # doing_scale_factor and snip_doing_log_scale_factor
        (
            "a",
            "log_a",
            {
                "first_snap_scale_factor": 0.1,
                "final_snap_scale_factor": 1.0,
                "delta_scale_factor": 0.1,
            },
            {"snipshot_delta_log_scale_factor": 0.1},
        ),
    ],
)
def test_mixed_snipshot_units_coverage(
    mock_cosmo,
    mock_conversions,
    tmp_path,
    main_type,
    snip_type,
    main_arg,
    snip_arg,
):
    """Test various combinations of snapshot and snipshot units."""
    out_file = tmp_path / f"out_mix_{main_type}_{snip_type}.txt"
    args = {"out": out_file, **main_arg, **snip_arg}

    # Just ensure it runs without error
    _generate_output_list_with_cosmo(args, mock_cosmo)
    assert out_file.exists()


def test_mixed_snipshot_units_logic(mock_cosmo, mock_conversions, tmp_path):
    """Test case where snipshots and snapshots have different units."""
    out_file = tmp_path / "out_mixed.txt"
    # Snap in Z: 1.0 to 0.0
    # Snip in Time: 0.95 (which is z=0.5)
    args = {
        "out": out_file,
        "first_snap_z": 1.0,
        "final_snap_z": 0.0,
        "delta_z": 1.0,
        "snipshot_delta_time": 0.05,  # small delta to hit something
        "list_units": "scale-factor",  # Convert all to a
    }
    # This just ensures no crash in the complex snipshot conversion block
    _generate_output_list_with_cosmo(args, mock_cosmo)
    with open(out_file, "r") as f:
        line1 = f.readline()
    assert "# Scale Factor, Select Output" in line1


def test_list_units_conversion_permutations(
    mock_cosmo, mock_conversions, tmp_path
):
    """Test remaining conversion permutations for list_units."""
    # Case 1: Start in Scale Factor (or Log), Convert to Redshift
    # test_list_units_redshift_conversion covered Time -> Redshift.
    # We need Scale Factor -> Redshift.
    out_file = tmp_path / "out_a_to_z.txt"
    args = {
        "out": out_file,
        "first_snap_scale_factor": 0.5,  # z=1.0
        "final_snap_scale_factor": 1.0,  # z=0.0
        "delta_scale_factor": 0.5,
        "list_units": "redshift",
    }
    _generate_output_list_with_cosmo(args, mock_cosmo)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert "# Redshift, Select Output" in lines[0]
    assert np.isclose(
        float(lines[1].split(",")[0]), 1.0
    )  # Sorted descending for Z

    # Case 2: Start in Redshift, Convert to Time
    # Wait, test_with_cosmo_converts_z_to_time covered this? Yes.
    # But let's verify coverage.

    # Case 3: Start in Time, Convert to Scale Factor
    # test_with_cosmo_converts_time_to_scale_factor covered this.

    # Case 4: Start in Redshift, Convert to Scale Factor
    # test_list_units_scale_factor_conversion covered this.

    # Case 5: Start in Time, Convert to Redshift (Covered by
    # test_list_units_redshift_conversion)

    # Case 6: Start in Scale Factor, Convert to Time
    # test_list_units_time_conversion covered this.

    # What about doing_z -> time conversion inside the loop?
    # Logic:
    # if list_units == "time":
    #    if doing_z: ...
    #    elif doing_scale_factor: ...
    # Ensure both branches are hit.

    # Case: Z -> Time
    out_file_z_t = tmp_path / "out_z_to_t.txt"
    args_z_t = {
        "out": out_file_z_t,
        "first_snap_z": 1.0,
        "final_snap_z": 0.0,
        "delta_z": 1.0,
        "list_units": "time",
    }
    _generate_output_list_with_cosmo(args_z_t, mock_cosmo)

    # Case: Time -> Scale Factor (already checked?)
    # Logic:
    # if list_units == "scale-factor":
    #    if doing_z: ...
    #    elif doing_time: ...

    # We need Time -> Scale Factor test explicitly here just to be sure
    out_file_t_a = tmp_path / "out_t_to_a.txt"
    args_t_a = {
        "out": out_file_t_a,
        "first_snap_time": 0.0,  # z=1 -> a=0.5
        "final_snap_time": 1.0,  # z=0 -> a=1
        "delta_time": 1.0,
        "list_units": "scale-factor",
    }
    _generate_output_list_with_cosmo(args_t_a, mock_cosmo)


def test_regression_log_scale_factor(mock_cosmo, mock_conversions, tmp_path):
    """Regression test for the log scale factor bug."""
    out_file = tmp_path / "out_log.txt"
    args = {
        "out": out_file,
        "first_snap_scale_factor": 0.1,
        "final_snap_scale_factor": 1.0,
        "delta_log_scale_factor": 0.1,
    }
    # Should run without error
    _generate_output_list_with_cosmo(args, mock_cosmo)
