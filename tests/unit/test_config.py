"""Tests for the config mode."""

import argparse
from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.modes.config import add_arguments, merge_flags, run


class TestConfigMode:
    """Tests for config mode."""

    def test_add_arguments(self):
        """Test that add_arguments adds all config arguments."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        # Test default values
        args = parser.parse_args([])
        assert args.list_options is False
        assert args.debug_build is False
        assert args.gravity is False
        assert args.eagle is False
        assert args.eaglexl is False

        # Test --list-options flag
        args = parser.parse_args(["--list-options"])
        assert args.list_options is True

        # Test --debug-build flag
        args = parser.parse_args(["--debug-build"])
        assert args.debug_build is True

        # Test --gravity flag
        args = parser.parse_args(["--gravity"])
        assert args.gravity is True

        # Test --eagle flag
        args = parser.parse_args(["--eagle"])
        assert args.eagle is True

        # Test --eaglexl flag
        args = parser.parse_args(["--eaglexl"])
        assert args.eaglexl is True

        # The presets have no short forms (and --debug is the global flag;
        # swift-cli builds every mode parser with allow_abbrev=False)
        parser.allow_abbrev = False
        for flag in ("-d", "-g", "-e", "-x", "--debug"):
            _, unknown = parser.parse_known_args([flag])
            assert unknown == [flag]

    @patch("swiftsim_cli.modes.config.show_config_options")
    def test_run_with_show(self, mock_show, tmp_path):
        """Test the run function with --list-options flag."""
        # Create mock args
        args = Mock()
        args.list_options = True

        # Call run
        run(args)

        # Verify show_config_options was called
        mock_show.assert_called_once_with()

    @patch("swiftsim_cli.modes.config.config_swiftsim")
    def test_run_with_debug_build(self, mock_config, tmp_path):
        """Test the run function with --debug-build flag."""
        # Create mock args
        args = Mock()
        args.list_options = False
        args.debug_build = True
        args.gravity = False
        args.eagle = False
        args.eaglexl = False
        args.options = []

        # Call run
        run(args)

        # Verify config_swiftsim was called with debug options
        mock_config.assert_called_once()
        call_args = mock_config.call_args
        opts = call_args[1]["opts"]
        assert "--enable-debug" in opts
        assert "--enable-debugging-checks" in opts
        assert "--disable-optimization" in opts

    @patch("swiftsim_cli.modes.config.config_swiftsim")
    def test_run_with_gravity(self, mock_config, tmp_path):
        """Test the run function with --gravity flag."""
        # Create mock args
        args = Mock()
        args.list_options = False
        args.debug_build = False
        args.gravity = True
        args.eagle = False
        args.eaglexl = False
        args.options = []

        # Call run
        run(args)

        # Verify config_swiftsim was called with gravity options
        mock_config.assert_called_once()
        call_args = mock_config.call_args
        opts = call_args[1]["opts"]
        assert "--enable-ipo" in opts
        assert "--with-tbbmalloc" in opts
        assert "--with-parmetis" in opts

    @patch("swiftsim_cli.modes.config.config_swiftsim")
    def test_run_with_eagle(self, mock_config, tmp_path):
        """Test the run function with --eagle flag."""
        # Create mock args
        args = Mock()
        args.list_options = False
        args.debug_build = False
        args.gravity = False
        args.eagle = True
        args.eaglexl = False
        args.options = []

        # Call run
        run(args)

        # Verify config_swiftsim was called with EAGLE options
        mock_config.assert_called_once()
        call_args = mock_config.call_args
        opts = call_args[1]["opts"]
        assert "--with-subgrid=EAGLE" in opts
        assert "--with-hydro=sphenix" in opts
        assert "--with-kernel=wendland-C2" in opts

    @patch("swiftsim_cli.modes.config.config_swiftsim")
    def test_run_with_eaglexl(self, mock_config, tmp_path):
        """Test the run function with --eaglexl flag."""
        # Create mock args
        args = Mock()
        args.list_options = False
        args.debug_build = False
        args.gravity = False
        args.eagle = False
        args.eaglexl = True
        args.options = []

        # Call run
        run(args)

        # Verify config_swiftsim was called with EAGLE-XL options
        mock_config.assert_called_once()
        call_args = mock_config.call_args
        opts = call_args[1]["opts"]
        assert "--with-subgrid=EAGLE-XL" in opts
        assert "--with-hydro=sphenix" in opts
        assert "--with-kernel=wendland-C2" in opts

    @patch("swiftsim_cli.modes.config.config_swiftsim")
    def test_run_with_multiple_presets(self, mock_config, tmp_path):
        """Test the run function with multiple preset flags."""
        # Create mock args
        args = Mock()
        args.list_options = False
        args.debug_build = True
        args.gravity = True
        args.eagle = False
        args.eaglexl = False
        args.options = []

        # Call run
        run(args)

        # Verify config_swiftsim was called with combined options
        mock_config.assert_called_once()
        call_args = mock_config.call_args
        opts = call_args[1]["opts"]
        # Should have debug options
        assert "--enable-debug" in opts
        assert "--enable-debugging-checks" in opts
        assert "--disable-optimization" in opts
        # Should have gravity options
        assert "--enable-ipo" in opts
        assert "--with-tbbmalloc" in opts
        assert "--with-parmetis" in opts

    @patch("swiftsim_cli.modes.config.config_swiftsim")
    def test_run_removes_duplicates(self, mock_config, tmp_path):
        """Test that duplicate options are removed."""
        # Create mock args with presets that have overlapping options
        args = Mock()
        args.list_options = False
        args.debug_build = False
        args.gravity = True
        args.eagle = (
            True  # EAGLE also has --enable-ipo, --with-tbbmalloc, etc.
        )
        args.eaglexl = False
        args.options = []

        # Call run
        run(args)

        # Verify config_swiftsim was called
        mock_config.assert_called_once()
        call_args = mock_config.call_args
        opts_list = call_args[1]["opts"]

        # Count occurrences of overlapping options
        assert opts_list.count("--enable-ipo") == 1
        assert opts_list.count("--with-tbbmalloc") == 1
        assert opts_list.count("--with-parmetis") == 1

    @patch("swiftsim_cli.modes.config.config_swiftsim")
    def test_run_with_additional_options(self, mock_config, tmp_path):
        """Test the run function with additional options."""
        # Create mock args
        args = Mock()
        args.list_options = False
        args.debug_build = False
        args.gravity = False
        args.eagle = False
        args.eaglexl = False
        args.options = ["--custom-opt1", "--custom-opt2"]

        # Call run
        run(args)

        # Verify config_swiftsim was called with additional options
        mock_config.assert_called_once()
        call_args = mock_config.call_args
        opts = call_args[1]["opts"]
        assert "--custom-opt1" in opts
        assert "--custom-opt2" in opts


class TestPresetsAndMerging:
    """Preset exclusivity and flag merging."""

    def test_eagle_presets_are_exclusive(self, capsys):
        """--eagle and --eaglexl set the subgrid model differently."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--eagle", "--eaglexl"])
        assert "not allowed with" in capsys.readouterr().err

    def test_help_lists_preset_options(self):
        """Preset help is generated from the one preset table."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        helps = {a.dest: a.help for a in parser._actions}
        assert "--with-subgrid=EAGLE-XL" in str(helps["eaglexl"])

    def test_later_value_wins_with_warning(self, capsys):
        """A later value of an option overrides a preset's, with a warning."""
        merged = merge_flags(
            [
                "--with-hydro=sphenix",
                "--enable-ipo",
                "--enable-ipo",
                "--with-hydro=gadget2",
                "CC=gcc",
            ]
        )
        assert merged == ["--enable-ipo", "--with-hydro=gadget2", "CC=gcc"]
        err = capsys.readouterr().err
        assert "--with-hydro=gadget2 overrides --with-hydro=sphenix" in err
