"""Tests for the cosmology module."""

import numpy as np
import pytest

from swiftsim_cli.cosmology import (
    clear_cosmology_cache,
    convert_redshift_to_scale_factor,
    convert_redshift_to_time,
    convert_scale_factor_to_redshift,
    convert_time_to_redshift,
    get_cosmology,
)


def test_convert_scale_factor_to_redshift():
    """Test converting scale factor to redshift."""
    # Test scalar conversion
    assert convert_scale_factor_to_redshift(1.0) == pytest.approx(0.0)
    assert convert_scale_factor_to_redshift(0.5) == pytest.approx(1.0)
    assert convert_scale_factor_to_redshift(0.1) == pytest.approx(9.0)

    # Test array conversion
    scale_factors = np.array([1.0, 0.5, 0.1])
    redshifts = convert_scale_factor_to_redshift(scale_factors)
    assert np.allclose(redshifts, np.array([0.0, 1.0, 9.0]))


def test_convert_redshift_to_scale_factor():
    """Test converting redshift to scale factor."""
    # Test scalar conversion
    assert convert_redshift_to_scale_factor(0.0) == pytest.approx(1.0)
    assert convert_redshift_to_scale_factor(1.0) == pytest.approx(0.5)
    assert convert_redshift_to_scale_factor(9.0) == pytest.approx(0.1)

    # Test array conversion
    redshifts = np.array([0.0, 1.0, 9.0])
    scale_factors = convert_redshift_to_scale_factor(redshifts)
    assert np.allclose(scale_factors, np.array([1.0, 0.5, 0.1]))


def test_get_cosmology(mocker):
    """Test getting cosmology object from parameters."""
    # Mock the load_parameters function
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value={
            "Cosmology": {
                "h": 0.6774,
                "Omega_m": 0.3089,
                "Omega_b": 0.0486,
            }
        },
    )

    # Get the cosmology object
    cosmo = get_cosmology()

    # Check that the cosmology object is correct
    assert cosmo.H0.value == 67.74
    assert cosmo.Om0 == 0.3089
    assert cosmo.Ob0 == 0.0486


def test_convert_redshift_to_time(mocker):
    """Test converting redshift to time."""
    # Mock the load_parameters function
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value={
            "Cosmology": {
                "h": 0.6774,
                "Omega_m": 0.3089,
                "Omega_b": 0.0486,
            }
        },
    )
    clear_cosmology_cache()

    # Test scalar conversion
    time = convert_redshift_to_time(1.0)
    assert time == pytest.approx(5.9, rel=1e-2)

    # Test array conversion
    redshifts = np.array([0.0, 1.0, 9.0])
    times = convert_redshift_to_time(redshifts)
    assert np.allclose(times, np.array([13.8, 5.9, 0.6]), rtol=1e-1)


def test_convert_time_to_redshift(mocker):
    """Test converting time to redshift."""
    # Mock the load_parameters function
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value={
            "Cosmology": {
                "h": 0.6774,
                "Omega_m": 0.3089,
                "Omega_b": 0.0486,
            }
        },
    )
    clear_cosmology_cache()

    # Test scalar conversion
    redshift = convert_time_to_redshift(5.9)
    assert redshift == pytest.approx(1.0, rel=1e-2)

    # Test array conversion
    times = np.array([13.8, 5.9, 0.6])
    redshifts = convert_time_to_redshift(times)
    assert np.allclose(
        redshifts, np.array([0.0, 1.0, 9.0]), rtol=1e-1, atol=1e-1
    )


def test_get_cosmology_no_parameters(mocker):
    """Test get_cosmology with no parameters."""
    # Mock load_parameters to return empty dict
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value={},
    )
    clear_cosmology_cache()

    # Should return None when no parameters
    cosmo = get_cosmology()
    assert cosmo is None


def test_convert_redshift_to_time_no_cosmology(mocker):
    """Test convert_redshift_to_time with no cosmology."""
    # Mock load_parameters to return empty dict (no cosmology)
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value={},
    )
    clear_cosmology_cache()

    # Should raise ValueError when cosmology not initialized
    with pytest.raises(ValueError, match="Cosmology not initialized"):
        convert_redshift_to_time(1.0)


def test_convert_time_to_redshift_out_of_range(mocker):
    """Test convert_time_to_redshift with out of range time."""
    # Mock the load_parameters function
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value={
            "Cosmology": {
                "h": 0.6774,
                "Omega_m": 0.3089,
                "Omega_b": 0.0486,
            }
        },
    )
    clear_cosmology_cache()

    # Times at or beyond the present-day age are clamped to z=0
    redshift = convert_time_to_redshift(1000.0)
    assert redshift == 0.0


PLANCK_PARAMS = {
    "Cosmology": {
        "h": 0.6774,
        "Omega_m": 0.3089,
        "Omega_b": 0.0486,
    }
}


def test_get_cosmology_without_cosmology_section(mocker):
    """A params file with no Cosmology section gives no cosmology."""
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value={"TimeIntegration": {"time_begin": 0.0}},
    )
    clear_cosmology_cache()

    assert get_cosmology() is None


def test_convert_time_to_redshift_no_cosmology(mocker):
    """convert_time_to_redshift raises clearly without a cosmology."""
    mocker.patch("swiftsim_cli.cosmology.load_parameters", return_value={})
    clear_cosmology_cache()

    with pytest.raises(ValueError, match="Cosmology not initialized"):
        convert_time_to_redshift(1.0)


def test_convert_time_to_redshift_too_early_raises(mocker):
    """Times before z=140 raise rather than silently becoming z=0."""
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value=PLANCK_PARAMS,
    )
    clear_cosmology_cache()

    with pytest.raises(ValueError, match="Cannot convert time"):
        convert_time_to_redshift(1e-6)
