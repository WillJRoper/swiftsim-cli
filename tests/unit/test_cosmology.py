"""Tests for the cosmology module."""

import warnings

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
                "Omega_lambda": 0.6911,
                "Omega_b": 0.0486,
            }
        },
    )

    # Get the cosmology object
    cosmo = get_cosmology()
    assert cosmo is not None

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
                "Omega_lambda": 0.6911,
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
                "Omega_lambda": 0.6911,
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
                "Omega_lambda": 0.6911,
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
        "Omega_lambda": 0.6911,
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


def test_incomplete_cosmology_section_is_a_value_error(mocker):
    """A Cosmology section missing a parameter is a clear error."""
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value={"Cosmology": {"a_begin": 0.01}},
    )
    clear_cosmology_cache()

    with pytest.raises(ValueError, match="Cosmology section has no"):
        get_cosmology()


def test_convert_time_to_redshift_no_cosmology(mocker):
    """convert_time_to_redshift raises clearly without a cosmology."""
    mocker.patch("swiftsim_cli.cosmology.load_parameters", return_value={})
    clear_cosmology_cache()

    with pytest.raises(ValueError, match="Cosmology not initialized"):
        convert_time_to_redshift(1.0)


def test_convert_time_to_redshift_too_early_raises(mocker):
    """Times before z=1000 raise rather than silently becoming z=0."""
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value=PLANCK_PARAMS,
    )
    clear_cosmology_cache()

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the astropy warning is suppressed
        with pytest.raises(ValueError, match="Cannot convert time"):
            convert_time_to_redshift(1e-6)


def test_internal_time_unit_in_gyr():
    """Mpc / (km/s) is ~977.8 Gyr; missing units give a clear error."""
    from swiftsim_cli.cosmology import internal_time_unit_in_gyr

    params = {
        "InternalUnitSystem": {
            "UnitLength_in_cgs": 3.08567758e24,
            "UnitVelocity_in_cgs": 1e5,
        }
    }
    assert internal_time_unit_in_gyr(params) == pytest.approx(977.79, 1e-4)
    with pytest.raises(ValueError, match="InternalUnitSystem"):
        internal_time_unit_in_gyr({"InternalUnitSystem": {}})


# Present-day ages printed by SWIFT ("Universe age at present day", in
# Mpc / (km/s)) from `swift --cosmology --self-gravity --dry-run` with
# these Cosmology sections (EAGLE, FLAMINGO L1_m10, and EAGLE with
# Omega_r and w0/wa)
SWIFT_AGES = [
    (
        {
            "h": 0.6777,
            "Omega_cdm": 0.2587481,
            "Omega_lambda": 0.693,
            "Omega_b": 0.0482519,
        },
        1.413442e-02,
    ),
    (
        {
            "h": 0.681,
            "Omega_cdm": 0.256011,
            "Omega_lambda": 0.693922,
            "Omega_b": 0.0486,
            "N_ur": 2.0308,
            "N_nu": 1,
            "M_nu_eV": 0.06,
            "T_nu_0": 1.95176,
        },
        1.407376e-02,
    ),
    (
        {
            "h": 0.6777,
            "Omega_cdm": 0.2587481,
            "Omega_lambda": 0.693,
            "Omega_b": 0.0482519,
            "Omega_r": 8.5e-5,
            "w_0": -0.9,
            "w_a": 0.1,
        },
        1.384442e-02,
    ),
]


@pytest.mark.parametrize("cosmo_params, swift_age", SWIFT_AGES)
def test_ages_match_swift(cosmo_params, swift_age):
    """The astropy cosmology reproduces SWIFT's age to < 1e-5."""
    from swiftsim_cli.cosmology import _GYR_IN_S, build_cosmology

    unit_gyr = 3.08567758e24 / 1e5 / _GYR_IN_S
    age = build_cosmology(cosmo_params).age(0).value
    assert age == pytest.approx(swift_age * unit_gyr, rel=1e-5)


def test_neutrinos_follow_swift():
    """N_eff and the massive species follow SWIFT's definitions."""
    from swiftsim_cli.cosmology import build_cosmology

    cosmo = build_cosmology(SWIFT_AGES[1][0])
    assert cosmo.Neff == pytest.approx(3.044006, rel=1e-6)  # SWIFT's value
    assert list(cosmo.m_nu.value) == [0.06, 0.0, 0.0]
    assert (
        build_cosmology(
            {**SWIFT_AGES[1][0], "M_nu_eV": "0.05, 0.01", "N_nu": 2}
        ).m_nu.value.max()
        == 0.05
    )
    # No neutrinos and no Omega_r: no radiation, as in SWIFT
    assert build_cosmology(SWIFT_AGES[0][0]).Tcmb0.value == 0.0


def test_time_redshift_round_trip_is_vectorised(mocker):
    """Arrays convert in one go and round-trip to machine precision."""
    mocker.patch(
        "swiftsim_cli.cosmology.load_parameters",
        return_value={"Cosmology": SWIFT_AGES[1][0]},
    )
    clear_cosmology_cache()
    z = np.linspace(0, 127, 3000)
    back = convert_time_to_redshift(convert_redshift_to_time(z))
    assert np.max(np.abs(back - z) / (1 + z)) < 1e-12
    cosmo = get_cosmology()
    assert cosmo is not None
    assert convert_redshift_to_time(3.0) == pytest.approx(
        cosmo.age(3.0).value, rel=1e-9
    )
