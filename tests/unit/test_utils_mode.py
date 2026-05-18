"""Tests for the utils mode."""

import argparse
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from unyt import unyt_array

from swiftsim_cli.modes.utils import (
    add_arguments,
    compute_softening_rows,
    format_softening_report,
    run,
)


class TestAddArguments:
    """Test the utils mode argument parser."""

    def test_softenings_arguments(self):
        """Test parsing the softenings subcommand."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        args = parser.parse_args(
            [
                "softenings",
                "--ics",
                "/tmp/ics.hdf5",
                "--params",
                "/tmp/params.yml",
            ]
        )

        assert args.utils_type == "softenings"
        assert str(args.ics) == "/tmp/ics.hdf5"
        assert str(args.params) == "/tmp/params.yml"
        assert args.softening_coeff is None
        assert args.pivot_redshift is None


class TestComputeSofteningRows:
    """Test softening calculations from a SWIFT dataset."""

    @patch("swiftsim_cli.modes.utils.load")
    def test_compute_softening_rows(self, mock_load):
        """Test per-species softenings are computed from particle counts."""
        mock_load.return_value = SimpleNamespace(
            metadata=SimpleNamespace(
                boxsize=unyt_array([100.0, 100.0, 100.0], "Mpc"),
                n_gas=64,
                n_dark_matter=8,
                n_boundary=0,
                n_sinks=0,
                n_stars=0,
                n_black_holes=0,
                n_neutrinos=0,
            )
        )

        box_size, rows = compute_softening_rows(
            ics_file=Mock(),
            softening_coeff=0.04,
            pivot_redshift=3.0,
        )

        assert box_size.value == 100.0
        assert [row["species"] for row in rows] == ["gas", "dark_matter"]
        assert rows[0]["count"] == 64
        assert rows[1]["count"] == 8
        assert np.isclose(rows[0]["comoving_softening"].value, 1.0)
        assert np.isclose(rows[0]["maximal_softening"].value, 0.25)
        assert np.isclose(rows[1]["comoving_softening"].value, 2.0)


class TestFormatSofteningReport:
    """Test softening report formatting."""

    def test_format_softening_report(self):
        """Test the report includes the key summary fields."""
        report = format_softening_report(
            ics_file=Mock(__str__=Mock(return_value="ics.hdf5")),
            param_file=Mock(__str__=Mock(return_value="params.yml")),
            softening_coeff=0.04,
            pivot_redshift=2.7,
            a_begin=0.1,
            box_size=unyt_array(100.0, "Mpc"),
            rows=[
                {
                    "label": "Gas",
                    "count": 64,
                    "mean_separation": unyt_array(25.0, "Mpc"),
                    "comoving_softening": unyt_array(1.0, "Mpc"),
                    "maximal_softening": unyt_array(0.27027, "Mpc"),
                }
            ],
        )

        assert "Softening report for ics.hdf5" in report
        assert "Parameter file: params.yml" in report
        assert "a_begin=0.1, z_begin=9" in report
        assert "softening_coeff=0.04, pivot_redshift=2.7" in report
        assert "Gas" in report


class TestRun:
    """Test mode dispatch."""

    @patch("swiftsim_cli.modes.utils.load_parameters")
    @patch("swiftsim_cli.modes.utils.load_swift_profile")
    @patch("swiftsim_cli.modes.utils.compute_softening_rows")
    @patch("swiftsim_cli.modes.utils.format_softening_report")
    def test_run_softenings(
        self,
        mock_format_report,
        mock_compute_rows,
        mock_load_profile,
        mock_load_parameters,
        capsys,
    ):
        """Test softenings dispatch uses profile defaults and prints report."""
        mock_load_profile.return_value = SimpleNamespace(
            softening_coeff=0.05,
            softening_pivot_z=4.0,
        )
        mock_load_parameters.return_value = {"Cosmology": {"a_begin": "0.2"}}
        mock_compute_rows.return_value = ("box", [{"label": "Gas"}])
        mock_format_report.return_value = "report text"

        args = SimpleNamespace(
            utils_type="softenings",
            ics=Mock(),
            params=Mock(),
            softening_coeff=None,
            pivot_redshift=None,
        )

        run(args)

        mock_compute_rows.assert_called_once_with(
            ics_file=args.ics,
            softening_coeff=0.05,
            pivot_redshift=4.0,
        )
        mock_format_report.assert_called_once()
        assert capsys.readouterr().out == "report text\n"
