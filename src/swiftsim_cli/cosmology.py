"""A module defining machinery related to cosmology in SWIFT.

astropy is imported inside the functions that need it, keeping it off the
CLI's start-up path.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping, Union

import numpy as np

from swiftsim_cli.params import get_parameter_cache_version, load_parameters

if TYPE_CHECKING:
    from astropy.cosmology import FLRW

# One Gyr in seconds (Julian years, as astropy's u.Gyr)
_GYR_IN_S = 365.25 * 86400.0 * 1e9

# SWIFT's present-day CMB temperature (const_T_CMB_0_cgs) in K
_T_CMB_0_K = 2.7255

# Earliest redshift the time -> redshift conversion covers
_Z_MAX = 1000.0


def _float_list(value: Any) -> list[float]:
    """Read a SWIFT array parameter ("0.05, 0.01", [0.05, 0.01] or 0.05)."""
    if isinstance(value, str):
        return [float(v) for v in value.strip("[] ").split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def build_cosmology(cosmo_params: Mapping[str, Any]) -> FLRW:
    """Build an astropy cosmology matching SWIFT's for a Cosmology section.

    Follows SWIFT's cosmology_init: Omega_m = Omega_cdm + Omega_b
    (massive neutrinos are extra), Omega_lambda is used as given and any
    remainder is curvature. Without neutrinos (N_ur = N_nu = 0) the only
    radiation is Omega_r (photons, default 0); with neutrinos the CMB
    temperature is SWIFT's 2.7255 K (or implied by Omega_r) and the massive
    species are added with their masses and degeneracies.

    Args:
        cosmo_params: The parameter file's Cosmology section.

    Returns:
        The astropy cosmology.

    Raises:
        ValueError: If a required parameter is missing.
    """
    import astropy.constants as const
    import astropy.units as u
    from astropy.cosmology import w0waCDM

    try:
        h = float(cosmo_params["h"])
        omega_b = float(cosmo_params["Omega_b"])
        if "Omega_m" in cosmo_params:
            omega_m = float(cosmo_params["Omega_m"])
        else:
            omega_m = float(cosmo_params["Omega_cdm"]) + omega_b
        omega_lambda = float(cosmo_params["Omega_lambda"])
    except KeyError as e:
        raise ValueError(
            f"The parameter file's Cosmology section has no {e.args[0]}"
        ) from e

    omega_r = float(cosmo_params.get("Omega_r", 0.0))
    n_ur = float(cosmo_params.get("N_ur", 0.0))
    n_nu = int(cosmo_params.get("N_nu", 0))

    # The CMB temperature implied by a photon density Omega_r
    h0 = (100 * h * u.km / u.s / u.Mpc).to(1 / u.s)
    rho_crit = 3 * h0**2 / (8 * np.pi * const.G)
    t_from_omega_r = (
        (omega_r * rho_crit * const.c**3 / (4 * const.sigma_sb)) ** 0.25
    ).to_value(u.K)

    neff = 0.0
    m_nu: list[float] = [0.0]
    if n_ur == 0 and n_nu == 0:
        t_cmb = t_from_omega_r
    else:
        t_cmb = t_from_omega_r if omega_r != 0 else _T_CMB_0_K
        decoupling = (4.0 / 11.0) ** (1.0 / 3.0)
        t_nu = float(cosmo_params.get("T_nu_0", 0.0)) or t_cmb * decoupling
        masses = [0.0] * n_nu
        degeneracies = [1.0] * n_nu
        if n_nu > 0:
            given = _float_list(cosmo_params.get("M_nu_eV", masses))
            masses[: len(given)] = given[:n_nu]
            given = _float_list(cosmo_params.get("deg_nu", degeneracies))
            degeneracies[: len(given)] = given[:n_nu]
        neff = n_ur + sum(degeneracies) * (t_nu / t_cmb) ** 4 / decoupling**4
        # astropy shares Neff equally between floor(Neff) species, the
        # massive ones listed first
        n_species = int(np.floor(neff))
        massive = [
            m
            for m, d in zip(masses, degeneracies)
            for _ in range(max(1, round(d)))
        ]
        m_nu = (massive + [0.0] * n_species)[:n_species] or [0.0]

    return w0waCDM(
        H0=100 * h,
        Om0=omega_m,
        Ode0=omega_lambda,
        w0=float(cosmo_params.get("w_0", -1.0)),
        wa=float(cosmo_params.get("w_a", 0.0)),
        Tcmb0=t_cmb,
        Neff=neff,
        m_nu=m_nu * u.eV,
        Ob0=omega_b,
    )


def _get_cosmology() -> FLRW | None:
    """Get the cosmology defined by the loaded parameter file.

    Returns:
        The astropy cosmology, or None if no parameters are loaded or they
        have no Cosmology section.
    """
    cosmo_params = load_parameters().get("Cosmology")
    if not cosmo_params:
        return None
    return build_cosmology(cosmo_params)


@lru_cache(maxsize=1)
def _get_cached_cosmology(_params_version: int) -> FLRW | None:
    """Get the cached cosmology object for a parameter cache version."""
    return _get_cosmology()


def get_cosmology() -> FLRW | None:
    """Get the cached cosmology object.

    If no parameters (or no Cosmology section) are defined, this will return
    None.

    Returns:
        FLRW | None: The cached cosmology object.
    """
    return _get_cached_cosmology(get_parameter_cache_version())


def clear_cosmology_cache() -> None:
    """Clear the cached cosmology object."""
    _get_cached_cosmology.cache_clear()
    _age_grid.cache_clear()


def convert_redshift_to_time(
    redshift: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Convert redshift to the age of the Universe in Gyr.

    Redshifts up to 1000 use the same age grid as
    `convert_time_to_redshift`, so the two round-trip exactly; higher ones
    use astropy's age integral.

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
    z = np.asarray(redshift, dtype=float)
    if np.any(z > _Z_MAX) or np.any(z < 0):
        return cosmo.age(redshift).value
    _, _, age_of_log_a = _age_grid(get_parameter_cache_version())
    ages = age_of_log_a(-np.log1p(z))
    return float(ages) if np.ndim(redshift) == 0 else ages


def convert_time_to_redshift(
    time: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Convert the age of the Universe in Gyr to redshift.

    The age is interpolated on a fine grid in log(a), so arrays convert in
    one vectorised step. Ages at or beyond the present-day age are clamped
    to z=0.

    Args:
        time: The age(s) in Gyr to convert.

    Returns:
        The redshift(s) corresponding to the given age(s).

    Raises:
        ValueError: If no cosmology is available or an age is earlier than
            the age at z=1000.
    """
    cosmo = get_cosmology()
    if cosmo is None:
        raise ValueError("Cosmology not initialized")
    ages, log_a_of_age, _ = _age_grid(get_parameter_cache_version())
    times = np.asarray(time, dtype=float)
    if np.any(times < ages[0]):
        raise ValueError(
            f"Cannot convert time {times.min():.6g} Gyr to a redshift: it is "
            f"before z={_Z_MAX:g} ({ages[0]:.6g} Gyr)"
        )
    log_a_out = np.minimum(log_a_of_age(np.minimum(times, ages[-1])), 0.0)
    redshift = np.expm1(-log_a_out)
    return float(redshift) if np.ndim(time) == 0 else redshift


@lru_cache(maxsize=1)
def _age_grid(_params_version: int) -> tuple[np.ndarray, Any, Any]:
    """Return the age grid (Gyr) and splines of ln(a) <-> age.

    The grid runs from z=1000 to z=0 in ln(a); dt = dln(a) / H(a) is
    integrated up from the age at z=1000, which is far cheaper than an
    age integral per grid point.
    """
    from scipy.integrate import cumulative_simpson
    from scipy.interpolate import CubicSpline

    cosmo = get_cosmology()
    assert cosmo is not None
    log_a = np.linspace(-np.log1p(_Z_MAX), 0.0, 20001)
    integrand = cosmo.hubble_time.value / cosmo.efunc(np.expm1(-log_a))
    ages = cosmo.age(_Z_MAX).value + cumulative_simpson(
        integrand, x=log_a, initial=0.0
    )
    return ages, CubicSpline(ages, log_a), CubicSpline(log_a, ages)


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


def internal_time_unit_in_gyr(params: dict | None = None) -> float:
    """Return SWIFT's internal time unit in Gyr.

    SWIFT derives its time unit from the internal length and velocity units
    (U_t = U_L / U_V), so this is read from the parameter file's
    InternalUnitSystem section.

    Args:
        params: The parameters to read. Defaults to the loaded parameters.

    Returns:
        The length of one internal time unit in Gyr.

    Raises:
        ValueError: If the InternalUnitSystem section or either unit is
            missing.
    """
    if params is None:
        params = load_parameters()
    units = params.get("InternalUnitSystem") or {}
    try:
        length_cgs = float(units["UnitLength_in_cgs"])
        velocity_cgs = float(units["UnitVelocity_in_cgs"])
    except KeyError as e:
        raise ValueError(
            "The parameter file needs InternalUnitSystem:UnitLength_in_cgs "
            "and UnitVelocity_in_cgs to write a time-based output list "
            "for a cosmological run."
        ) from e
    return length_cgs / velocity_cgs / _GYR_IN_S
