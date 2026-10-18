"""Pytest configuration and fixtures for swiftsim-cli tests."""

import os

# Render headless in tests; the package itself must not force a backend.
os.environ.setdefault("MPLBACKEND", "Agg")

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from swiftsim_cli import profile as profile_mod
from swiftsim_cli.timing.timer_db import TimerDB, build_timer_db

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def isolated_profile_file(tmp_path, monkeypatch):
    """Keep every test away from the real ~/.swiftsim-utils directory."""
    monkeypatch.setattr(
        profile_mod,
        "PROFILE_FILE",
        tmp_path / ".swiftsim-utils" / "profiles.yaml",
    )
    profile_mod.set_profile_override(None)
    profile_mod._load_store_cached.cache_clear()
    yield profile_mod.PROFILE_FILE
    profile_mod.set_profile_override(None)
    profile_mod._load_store_cached.cache_clear()


@pytest.fixture(scope="session")
def swift_src_dir() -> Path:
    """Synthetic SWIFT-like C tree matching the log fixtures."""
    return DATA_DIR / "swift_src"


@pytest.fixture(scope="session")
def logs_dir() -> Path:
    """Directory of the (scrubbed) SWIFT log fixtures."""
    return DATA_DIR / "logs"


@pytest.fixture(scope="session")
def synthetic_db(swift_src_dir) -> TimerDB:
    """Timer database parsed from the synthetic C tree."""
    return TimerDB(build_timer_db(swift_src_dir))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def _clear_param_and_cosmology_caches():
    """Stop cached parameters/cosmology leaking between tests."""
    from swiftsim_cli.cosmology import clear_cosmology_cache
    from swiftsim_cli.params import clear_parameter_cache

    clear_parameter_cache()
    clear_cosmology_cache()
    yield
    clear_parameter_cache()
    clear_cosmology_cache()


@pytest.fixture
def write_tiny_ics():
    """Return a writer for a tiny gas-only IC file with a Units group.

    The file has no dark matter and is complete enough for swiftsimio.
    """

    def _write(path, time=0.0, redshift=None, ngas=64):
        counts = np.array([ngas, 0, 0, 0, 0, 0])
        with h5py.File(path, "w") as f:
            header = f.create_group("Header")
            header.attrs["BoxSize"] = np.array([100.0, 100.0, 100.0])
            header.attrs["NumPart_Total"] = counts
            header.attrs["NumPart_ThisFile"] = counts
            header.attrs["Time"] = np.array([time])
            if redshift is not None:
                header.attrs["Redshift"] = np.array([redshift])
            units = f.create_group("Units")
            units.attrs["Unit mass in cgs (U_M)"] = np.array([1.989e43])
            units.attrs["Unit length in cgs (U_L)"] = np.array([3.085e24])
            units.attrs["Unit time in cgs (U_t)"] = np.array([3.085e19])
            units.attrs["Unit current in cgs (U_I)"] = np.array([1.0])
            units.attrs["Unit temperature in cgs (U_T)"] = np.array([1.0])
            f.create_dataset(
                "PartType0/Coordinates", data=np.full((ngas, 3), 50.0)
            )
        return path

    return _write
