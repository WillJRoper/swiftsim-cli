"""Tests for the show mode."""

import argparse
from unittest.mock import patch

from swiftsim_cli.modes import show


class TestShowMode:
    """Tests for the show mode."""

    def test_add_arguments(self):
        """Show takes an optional NAME, --cosmo and --all."""
        parser = argparse.ArgumentParser()
        show.add_arguments(parser)

        args = parser.parse_args([])
        assert args.name is None
        assert args.cosmo is False
        assert args.all is False

        args = parser.parse_args(["dev", "--cosmo"])
        assert args.name == "dev"
        assert args.cosmo is True

        args = parser.parse_args(["--all"])
        assert args.all is True

    @patch("swiftsim_cli.modes.show.display_profile")
    def test_run_default(self, mock_display):
        """By default show matches 'profile --show' (profile + cosmology)."""
        show.run(argparse.Namespace(name=None, cosmo=False, all=False))
        mock_display.assert_called_once_with(name=None)

    @patch("swiftsim_cli.modes.show.display_profile")
    def test_run_cosmo_only(self, mock_display):
        """--cosmo shows only the cosmology, of the named profile."""
        show.run(argparse.Namespace(name="dev", cosmo=True, all=False))
        mock_display.assert_called_once_with(
            print_header=False, show_profile=False, name="dev"
        )

    @patch("swiftsim_cli.modes.show.display_profile")
    def test_run_all(self, mock_display):
        """--all is the default behaviour (kept for compatibility)."""
        show.run(argparse.Namespace(name=None, cosmo=False, all=True))
        mock_display.assert_called_once_with(name=None)
