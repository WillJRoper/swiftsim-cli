"""Tests for the show mode."""

import argparse
from unittest.mock import patch

import pytest

from swiftsim_cli.modes import show


class TestShowMode:
    """Tests for the show mode."""

    def test_add_arguments(self):
        """Show takes an optional NAME and --cosmo."""
        parser = argparse.ArgumentParser()
        show.add_arguments(parser)

        args = parser.parse_args([])
        assert args.name is None
        assert args.cosmo is False

        args = parser.parse_args(["dev", "--cosmo"])
        assert args.name == "dev"
        assert args.cosmo is True

    @patch("swiftsim_cli.modes.show.display_profile")
    def test_run_default(self, mock_display):
        """By default show matches 'profile --show' (profile + cosmology)."""
        show.run(argparse.Namespace(name=None, cosmo=False))
        mock_display.assert_called_once_with(name=None)

    @patch("swiftsim_cli.modes.show.display_profile")
    def test_run_cosmo_only(self, mock_display):
        """--cosmo shows only the cosmology, of the named profile."""
        show.run(argparse.Namespace(name="dev", cosmo=True))
        mock_display.assert_called_once_with(
            print_header=False, show_profile=False, name="dev"
        )


def test_cosmology_shows_derived_quantities(tmp_path, capsys):
    """A full Cosmology section also gets SWIFT's derived quantities."""
    from swiftsim_cli.modes.profile import display_cosmology

    params = tmp_path / "params.yml"
    params.write_text(
        "Cosmology:\n  h: 0.6777\n  Omega_cdm: 0.2587481\n"
        "  Omega_lambda: 0.693\n  Omega_b: 0.0482519\n  a_begin: 0.5\n"
    )
    display_cosmology(params)
    out = capsys.readouterr().out
    rows = {
        cells[0]: cells[1]
        for line in out.split("DERIVED COSMOLOGY")[1].splitlines()
        if len(cells := [c.strip() for c in line.strip("|").split("|")]) == 2
    }
    assert rows["Omega_m"] == "0.307"
    assert rows["Omega_k"] == "0"  # no neutrinos or radiation: closed
    assert rows["z_begin"] == "1"
    assert float(rows["age today [Gyr]"]) == pytest.approx(13.82, abs=0.01)


def test_incomplete_cosmology_warns(tmp_path, capsys):
    """An incomplete Cosmology section is shown, with a warning."""
    from swiftsim_cli.modes.profile import display_cosmology

    params = tmp_path / "params.yml"
    params.write_text("Cosmology:\n  h: 0.7\n")
    display_cosmology(params)
    out, err = capsys.readouterr()
    assert "| h " in out and "DERIVED" not in out
    assert "unusable Cosmology section" in err and "Omega_b" in err
