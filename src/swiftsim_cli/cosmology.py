"""A module defining machinery related to cosmology in SWIFT."""

from functools import lru_cache
from typing import Union

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.cosmology import (  # type: ignore[import-untyped]
    CosmologyError,
    FlatLambdaCDM,
    z_at_value,
)

from swiftsim_cli.params import get_parameter_cache_version, load_parameters


def _get_cosmology() -> FlatLambdaCDM | None:
    """Get the cosmology parameters from the SWIFT configuration.

    Returns:
        FlatLambdaCDM | None: An astropy cosmology object with the parameters
        defined in the SWIFT configuration, or None if no parameters are
        loaded or they have no Cosmology section.
    """
    # Get the parameters
    params = load_parameters()

    # Without a Cosmology section we can't set up a cosmology
    cosmo_params = params.get("Cosmology")
    if not cosmo_params:
        return None

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
def _get_cached_cosmology(_params_version: int) -> FlatLambdaCDM | None:
    """Get the cached cosmology object for a parameter cache version."""
    return _get_cosmology()


def get_cosmology() -> FlatLambdaCDM | None:
    """Get the cached cosmology object.

    If no parameters (or no Cosmology section) are defined, this will return
    None.

    Returns:
        FlatLambdaCDM | None: The cached cosmology object.
    """
    return _get_cached_cosmology(get_parameter_cache_version())


def clear_cosmology_cache() -> None:
    """Clear the cached cosmology object."""
    _get_cached_cosmology.cache_clear()


def convert_redshift_to_time(
    redshift: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Convert redshift to the age of the Universe in Gyr.

    Args:
        redshift: The redshift value(s) to convert.

    Returns:
        The age(s) in Gyr corresponding to the given redshift(s).

    Raises:
        ValueError: If no cosmology is available.
    """
    cosmo = get_cosmology()
    if cosmo is None:
        raise ValueError("Cosmology not initialized")
    return cosmo.age(redshift).value


def convert_time_to_redshift(
    time: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Convert the age of the Universe in Gyr to redshift.

    Ages at or beyond the present-day age are clamped to z=0.

    Args:
        time: The age(s) in Gyr to convert.

    Returns:
        The redshift(s) corresponding to the given age(s).

    Raises:
        ValueError: If no cosmology is available or an age cannot be
            converted (e.g. it is earlier than the age at z=140).
    """
    cosmo = get_cosmology()
    if cosmo is None:
        raise ValueError("Cosmology not initialized")
    age_now = cosmo.age(0).value

    def _convert_single_time(t):
        if t >= age_now:
            return 0.0
        try:
            return z_at_value(cosmo.age, t * u.Gyr, zmax=140, zmin=0.0).value
        except CosmologyError as e:
            raise ValueError(
                f"Cannot convert time {t} Gyr to a redshift: {e}"
            ) from e

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
