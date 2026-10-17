"""Pytest configuration and fixtures for swiftsim-cli tests."""

import os

# Render headless in tests; the package itself must not force a backend.
os.environ.setdefault("MPLBACKEND", "Agg")

import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from swiftsim_cli.src_parser import TimerDef, TimerInstance


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_timer_db() -> Dict[str, TimerDef]:
    """Create a sample timer database for testing."""
    return {
        "space_split.c:100": TimerDef(
            timer_id="space_split.c:100",
            function="space_split",
            log_pattern=r"^.*space_split:\s+Zoom cell tree.*took"
            r"\s+([\d.]+)\s+ms",
            start_line=95,
            end_line=100,
            label_text="Zoom cell tree and multipole construction "
            "took %.3f %s.",
            timer_type="operation",
        ),
        "space_split.c:105": TimerDef(
            timer_id="space_split.c:105",
            function="space_split",
            log_pattern=r"^.*space_split:\s+Background cell tree.*"
            r"took\s+([\d.]+)\s+ms",
            start_line=102,
            end_line=105,
            label_text="Background cell tree and multipole construction "
            "took %.3f %s.",
            timer_type="operation",
        ),
        "engine_launch.c:50": TimerDef(
            timer_id="engine_launch.c:50",
            function="engine_launch",
            log_pattern=r"^.*engine_launch:\s+took\s+([\d.]+)\s+ms",
            start_line=45,
            end_line=50,
            label_text="took %.3f %s.",
            timer_type="function",
        ),
        "engine_launch.c:50|(tasks)": TimerDef(
            timer_id="engine_launch.c:50|(tasks)",
            function="engine_launch",
            log_pattern=r"^.*engine_launch:\s+took\s+([\d.]+)\s+ms",
            start_line=45,
            end_line=50,
            label_text="(tasks) took %.3f %s.",
            timer_type="operation",
        ),
    }


@pytest.fixture
def sample_nesting_db() -> Dict:
    """Create a sample nesting database for testing."""
    return {
        "space_split": {
            "function_timer": "took %.3f %s.",
            "file": "space_split.c",
            "nested_operations": [
                "Zoom cell tree and multipole construction took %.3f %s.",
                "Background cell tree and multipole construction took "
                "%.3f %s.",
            ],
            "nested_functions": [],
        },
        "engine_launch": {
            "function_timer": "took %.3f %s.",
            "file": "engine_launch.c",
            "nested_operations": ["(tasks) took %.3f %s."],
            "nested_functions": ["space_split"],
        },
    }


@pytest.fixture
def sample_timer_instances() -> Dict[int, List[TimerInstance]]:
    """Create sample timer instances for testing."""
    return {
        0: [
            TimerInstance(
                timer_id="space_split.c:100",
                function="space_split",
                step=0,
                time_ms=609.098,
                line_index=100,
                timer_type="operation",
            ),
            TimerInstance(
                timer_id="space_split.c:105",
                function="space_split",
                step=0,
                time_ms=15.180,
                line_index=101,
                timer_type="operation",
            ),
            TimerInstance(
                timer_id="engine_launch.c:50",
                function="engine_launch",
                step=0,
                time_ms=800.0,
                line_index=50,
                timer_type="function",
            ),
            TimerInstance(
                timer_id="engine_launch.c:50|(tasks)",
                function="engine_launch",
                step=0,
                time_ms=9.54,
                line_index=51,
                timer_type="operation",
            ),
        ]
    }


@pytest.fixture
def sample_log_content() -> str:
    """Create sample SWIFT log content for testing."""
    return (
        "[00009.0] space_split: Zoom cell tree and multipole construction "
        "took 609.098 ms."
        "[00009.1] space_split: Background cell tree and multipole "
        "construction took 15.180 ms."
        "[00009.9] engine_launch: (tasks) took 9.540 ms."
        "[00010.0] engine_launch: took 800.000 ms."
    )


@pytest.fixture
def mock_swift_profile():
    """Create a mock SWIFT profile for testing."""
    mock_profile = Mock()
    mock_profile.swiftsim_dir = "/mock/swift/dir"
    mock_profile.data_dir = "/mock/data/dir"
    return mock_profile


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
