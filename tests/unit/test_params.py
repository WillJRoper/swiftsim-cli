"""Tests for the params module."""

import os
from pathlib import Path

import pytest

from swiftsim_cli import params
from swiftsim_cli.params import load_parameters


def teardown_function():
    """Reset PARAMS cache after each test."""
    params.PARAMS = None
    params.PARAMS_SOURCE = None


def test_load_parameters_valid_file(tmp_path):
    """Test loading parameters from a valid YAML file."""
    # Create a temporary YAML file
    param_file = tmp_path / "params.yml"
    param_file.write_text("Cosmology:\n    h: 0.6774\n    Omega_m: 0.3089")

    # Load the parameters
    loaded_params = load_parameters(param_file)

    # Check that the parameters are correct
    assert loaded_params["Cosmology"]["h"] == 0.6774
    assert loaded_params["Cosmology"]["Omega_m"] == 0.3089


def test_load_parameters_file_not_found():
    """Test error when parameter file not found."""
    with pytest.raises(FileNotFoundError):
        load_parameters(Path("non_existent_file.yml"))


def test_load_parameters_invalid_yaml(tmp_path):
    """Test error when YAML is invalid."""
    # Create a temporary YAML file with invalid YAML
    param_file = tmp_path / "params.yml"
    param_file.write_text("Cosmology: {h: 0.6774, Omega_m: 0.3089")

    with pytest.raises(ValueError):
        load_parameters(param_file)


def test_load_parameters_with_tabs(tmp_path):
    """Test loading parameters with tabs in YAML."""
    # Create a temporary YAML file with tabs
    param_file = tmp_path / "params.yml"
    param_file.write_text("Cosmology:\n\th: 0.6774\n\tOmega_m: 0.3089")

    # Load the parameters
    loaded_params = load_parameters(param_file)

    # Check that the parameters are correct
    assert loaded_params["Cosmology"]["h"] == 0.6774
    assert loaded_params["Cosmology"]["Omega_m"] == 0.3089


def test_load_parameters_caching(tmp_path):
    """Test that parameters are cached."""
    # Create a temporary YAML file
    param_file = tmp_path / "params.yml"
    param_file.write_text("Cosmology:\n    h: 0.6774\n    Omega_m: 0.3089")

    # Load the parameters twice
    params1 = load_parameters(param_file)
    params2 = load_parameters(param_file)

    # Check that the parameters are the same object
    assert params1 is params2


def test_load_parameters_reloads_for_different_files(tmp_path):
    """Test that a different parameter file replaces the cached state."""
    param_file_1 = tmp_path / "params_1.yml"
    param_file_1.write_text("Cosmology:\n    h: 0.6774\n")
    param_file_2 = tmp_path / "params_2.yml"
    param_file_2.write_text("Cosmology:\n    h: 0.7010\n")

    params1 = load_parameters(param_file_1)
    params2 = load_parameters(param_file_2)

    assert params1 is not params2
    assert params2["Cosmology"]["h"] == 0.7010


def test_load_parameters_no_file():
    """Test loading parameters with no file specified."""
    params = load_parameters()
    assert params == {}


def test_load_parameters_permission_error(tmp_path):
    """Test error when file has no read permissions."""
    # Create a temporary file and remove read permissions
    param_file = tmp_path / "params.yml"
    param_file.write_text("Cosmology:\n    h: 0.6774\n    Omega_m: 0.3089")
    os.chmod(param_file, 0o000)

    with pytest.raises(IOError):
        load_parameters(param_file)
