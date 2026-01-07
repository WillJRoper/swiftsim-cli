"""A module defining machinery related to cosmology in SWIFT."""

from functools import lru_cache
from typing import Union

import astropy.cosmology.core  # type: ignore[import-untyped]
import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.cosmology import (  # type: ignore[import-untyped]
    FlatLambdaCDM,
    z_at_value,
)

from swiftsim_cli.params import load_parameters


def _get_cosmology() -> FlatLambdaCDM:
    """Get the cosmology parameters from the SWIFT configuration.

    Returns:
        FlatLambdaCDM: An astropy cosmology object with the parameters
        defined in the SWIFT configuration.
    """
    # Get the parameters
    params = load_parameters()

    # If we have no parameters we can't set up a cosmology
    if len(params) == 0:
        return None

    # Extract cosmology parameters
    cosmo_params = params["Cosmology"]

    # Calculate Omega_m if not provided (Omega_m = Omega_cdm + Omega_b)
    if "Omega_m" in cosmo_params:
        omega_m = cosmo_params["Omega_m"]
    else:
        omega_m = cosmo_params["Omega_cdm"] + cosmo_params["Omega_b"]

    # Create a FlatLambdaCDM cosmology object
    cosmo = FlatLambdaCDM(
        H0=cosmo_params["h"] * 100,
        Om0=omega_m,
        Ob0=cosmo_params["Omega_b"],
    )

    return cosmo


@lru_cache(maxsize=1)
def get_cosmology() -> FlatLambdaCDM | None:
    """Get the cached cosmology object.

    If no parameters are defined, this will return None.

    Returns:
        FlatLambdaCDM: The cached cosmology object.
    """
    return _get_cosmology()


def convert_redshift_to_time(
    redshift: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Convert redshift to time in seconds.

    Args:
        redshift: The redshift value(s) to convert.

    Returns:
        The time(s) in seconds corresponding to the given redshift(s).
    """
    cosmo = get_cosmology()
    if cosmo is None:
        raise ValueError("Cosmology not initialized")
    return cosmo.age(redshift).value


def convert_time_to_redshift(
    time: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Convert time in seconds to redshift.

    Args:
        time: The time(s) in seconds to convert.

    Returns:
        The redshift(s) corresponding to the given time(s).
    """
    cosmo = get_cosmology()

    def _convert_single_time(t):
        try:
            return z_at_value(cosmo.age, t * u.Gyr, zmax=140, zmin=0.0).value
        except astropy.cosmology.core.CosmologyError:
            return 0.0

    if np.isscalar(time):
        return _convert_single_time(time)
    else:
        return np.array([_convert_single_time(t) for t in np.asarray(time)])


def convert_scale_factor_to_redshift(
    scale_factor: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Convert scale factor to redshift.

    Args:
        scale_factor: The scale factor(s) to convert.

    Returns:
        The redshift(s) corresponding to the given scale factor(s).
    """
    return 1.0 / scale_factor - 1.0


def convert_redshift_to_scale_factor(
    redshift: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Convert redshift to scale factor.

    Args:
        redshift: The redshift value(s) to convert.

    Returns:
        The scale factor(s) corresponding to the given redshift(s).
    """
    return 1.0 / (1.0 + redshift)
