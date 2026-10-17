"""Tests for output_times run function."""

from unittest.mock import Mock

import numpy as np
import pytest

from swiftsim_cli.modes.output_times import run


def test_run_calls_generate_output_list(mocker, tmp_path):
    """Test that run() calls generate_output_list with args."""
    mock_generate = mocker.patch(
        "swiftsim_cli.modes.output_times.generate_output_list"
    )

    # Create mock args
    args = Mock()
    args.out = tmp_path / "output.txt"
    args.first_snap_z = 10.0
    args.final_snap_z = 0.0
    args.delta_z = 1.0
    args.select = 1

    run(args)

    # Should call generate_output_list with vars(args)
    mock_generate.assert_called_once()
    call_args = mock_generate.call_args[0][0]
    assert isinstance(call_args, dict)
    assert call_args["out"] == tmp_path / "output.txt"


COSMO_PARAMS_YML = """\
Cosmology:
  h: 0.6777
  a_begin: 0.0078125
  a_end: 0.5
  Omega_cdm: 0.2587481
  Omega_lambda: 0.693
  Omega_b: 0.0482519
"""


def _run_cli(tmp_path, mocker, extra_args):
    """Run output-times through the real CLI with a cosmological params file.

    Only the user profile is mocked; parameters and cosmology are real.
    """
    from swiftsim_cli.cli import main

    params = tmp_path / "params.yml"
    params.write_text(COSMO_PARAMS_YML)
    out = tmp_path / "output_list.txt"
    main(["output-times", "-p", str(params), "-o", str(out), *extra_args])
    return out.read_text().splitlines()


def test_cli_output_times_uses_cosmology_from_params(tmp_path, mocker):
    """A redshift start with a time delta needs the params cosmology."""
    lines = _run_cli(tmp_path, mocker, ["-fz", "10", "-dt", "0.5"])

    assert lines[0] == "# Time"
    times = [float(line) for line in lines[1:]]
    # Age at z=10 is ~0.47 Gyr, the final output is bounded by a_end=0.5
    # (age ~5.9 Gyr at z=1)
    assert times[0] == pytest.approx(0.47, abs=0.02)
    assert np.allclose(np.diff(times), 0.5)
    assert 5.4 < times[-1] < 6.0


def test_cli_output_times_final_defaults_to_params_a_end(tmp_path, mocker):
    """With no final snapshot given, a_end from the params file is used."""
    lines = _run_cli(tmp_path, mocker, ["-fz", "3", "-dz", "1"])

    assert lines[0] == "# Redshift"
    assert [float(line) for line in lines[1:]] == [3.0, 2.0, 1.0]
