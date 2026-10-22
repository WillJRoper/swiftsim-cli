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


def test_load_parameters_no_arg_returns_cached(tmp_path):
    """No-arg calls return what the CLI already loaded, not an empty dict."""
    param_file = tmp_path / "params.yml"
    param_file.write_text("Cosmology:\n  a_begin: 0.1\n")

    loaded = load_parameters(param_file)

    assert load_parameters() is loaded
    assert load_parameters()["Cosmology"]["a_begin"] == 0.1


@pytest.mark.parametrize("text", ["- a\n- b\n", "just a string\n", "42\n"])
def test_non_mapping_file_is_a_value_error(tmp_path, text):
    """A parameter file must be a mapping of sections."""
    param_file = tmp_path / "params.yml"
    param_file.write_text(text)
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_parameters(param_file)


def test_duplicate_keys_warn_and_keep_the_first(tmp_path, capsys):
    """Duplicate keys (as in SWIFT's parameter_example.yml) are a warning."""
    param_file = tmp_path / "params.yml"
    param_file.write_text("Gravity:\n  eta: 0.1\n  eta: 0.2\nA:\n  b: 1\n")
    loaded = load_parameters(param_file)
    assert loaded["Gravity"]["eta"] == 0.1
    assert loaded["A"]["b"] == 1
    err = capsys.readouterr().err
    assert err.startswith(f"warning: {param_file}:3: found duplicate key")
    assert "SWIFT itself rejects duplicate keys" in err


def test_empty_file_gives_no_parameters(tmp_path):
    """An empty parameter file is an empty mapping."""
    param_file = tmp_path / "params.yml"
    param_file.write_text("")
    assert load_parameters(param_file) == {}


def test_round_trip_yaml_never_wraps(tmp_path):
    """Long values stay on their key's line (SWIFT reads line by line)."""
    from io import StringIO

    from swiftsim_cli.params import round_trip_yaml

    value = "/a/very/long/path/" + "x" * 150
    buffer = StringIO()
    round_trip_yaml().dump({"Snapshots": {"subdir": value}}, buffer)
    assert f"    subdir: {value}\n" in buffer.getvalue()


def test_check_swift_parameter_text_finds_every_problem():
    """Parser limits: line length, duplicates, placeholders, counts."""
    from swiftsim_cli.params import check_swift_parameter_text

    text = (
        "# comment line\n---\nA:\n  x: 1   # trailing comment\n  x: 2\n"
        "  empty:\n  big: FLT_MAX\nA:\n  y: 3\n"
        f"B:\n  long: {'v' * 260}\n"
    )
    problems = check_swift_parameter_text(text)
    assert any("duplicate parameter A:x" in p for p in problems)
    assert any("duplicate section A" in p for p in problems)
    assert any("A:empty has no value" in p for p in problems)
    assert any("placeholder FLT_MAX" in p for p in problems)
    assert any("characters long" in p for p in problems)
    assert check_swift_parameter_text("A:\n  x: 1\nB:\n  y: [1, 2]\n") == []

    many = "".join(f"S{i}:\n  p: 1\n" for i in range(65))
    assert any("65 sections" in p for p in check_swift_parameter_text(many))
    params = "A:\n" + "".join(f"  p{i}: 1\n" for i in range(701))
    assert any(
        "701 parameters" in p for p in check_swift_parameter_text(params)
    )
