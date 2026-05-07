"""Tests for the config mode."""

import argparse
from unittest.mock import Mock, patch

from swiftsim_cli.modes.config import (
    add_arguments,
    config_swiftsim,
    run,
    show_config_options,
)


class TestConfigMode:
    """Tests for config mode."""

    def test_add_arguments(self):
        """Test that add_arguments adds all config arguments."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        # Test default values
        args = parser.parse_args([])
        assert args.show is False
        assert args.debug is False
        assert args.gravity is False
        assert args.eagle is False
        assert args.eaglexl is False

        # Test --show flag
        args = parser.parse_args(["--show"])
        assert args.show is True

        # Test --debug flag
        args = parser.parse_args(["--debug"])
        assert args.debug is True

        # Test --gravity flag
        args = parser.parse_args(["--gravity"])
        assert args.gravity is True

        # Test --eagle flag
        args = parser.parse_args(["--eagle"])
        assert args.eagle is True

        # Test --eaglexl flag
        args = parser.parse_args(["--eaglexl"])
        assert args.eaglexl is True

        # Test short forms
        args = parser.parse_args(["-s"])
        assert args.show is True

        args = parser.parse_args(["-d"])
        assert args.debug is True

        args = parser.parse_args(["-g"])
        assert args.gravity is True

        args = parser.parse_args(["-e"])
        assert args.eagle is True

        args = parser.parse_args(["-x"])
        assert args.eaglexl is True

    @patch("swiftsim_cli.modes.config.show_config_options")
    def test_run_with_show(self, mock_show, tmp_path):
        """Test the run function with --show flag."""
        # Create mock args
        args = Mock()
        args.show = True

        # Call run
        run(args)

        # Verify show_config_options was called
        mock_show.assert_called_once_with()

    @patch("swiftsim_cli.modes.config.config_swiftsim")
    def test_run_with_debug(self, mock_config, tmp_path):
        """Test the run function with --debug flag."""
        # Create mock args
        args = Mock()
        args.show = False
        args.debug = True
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
        args.show = False
        args.debug = False
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
        args.show = False
        args.debug = False
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
        args.show = False
        args.debug = False
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
        args.show = False
        args.debug = True
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
        args.show = False
        args.debug = False
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
        args.show = False
        args.debug = False
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


class TestConfigSwiftsim:
    """Tests for config_swiftsim function."""

    @patch("swiftsim_cli.modes.config._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.config.get_swiftsim_dir")
    def test_config_swiftsim(self, mock_get_dir, mock_run_command, tmp_path):
        """Test configuring SWIFT with options."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return the directory
        mock_get_dir.return_value = swift_dir

        # Call config_swiftsim
        config_swiftsim(["--enable-debug"], swift_dir)

        # Verify get_swiftsim_dir was called
        mock_get_dir.assert_called_once_with(swift_dir)

        # Verify the configure command was run
        mock_run_command.assert_called_once_with(
            ["./configure", "--enable-debug"], swift_dir
        )

    @patch("swiftsim_cli.modes.config._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.config.get_swiftsim_dir")
    def test_config_swiftsim_without_dir(
        self, mock_get_dir, mock_run_command, tmp_path
    ):
        """Test configuring SWIFT without explicit directory (uses config)."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return a directory
        mock_get_dir.return_value = swift_dir

        # Call config_swiftsim without directory
        config_swiftsim(["--enable-ipo"], None)

        # Verify get_swiftsim_dir was called with None
        mock_get_dir.assert_called_once_with(None)

        # Verify the configure command was run
        mock_run_command.assert_called_once_with(
            ["./configure", "--enable-ipo"], swift_dir
        )


class TestShowConfigOptions:
    """Tests for show_config_options function."""

    @patch("swiftsim_cli.modes.config._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.config.get_swiftsim_dir")
    def test_show_config_options(
        self, mock_get_dir, mock_run_command, tmp_path
    ):
        """Test showing configuration options."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return the directory
        mock_get_dir.return_value = swift_dir

        # Call show_config_options
        show_config_options(swift_dir)

        # Verify get_swiftsim_dir was called
        mock_get_dir.assert_called_once_with(swift_dir)

        # Verify the configure --help command was run
        mock_run_command.assert_called_once_with(
            ["./configure", "--help"], swift_dir
        )

    @patch("swiftsim_cli.modes.config._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.config.get_swiftsim_dir")
    def test_show_config_options_without_dir(
        self, mock_get_dir, mock_run_command, tmp_path
    ):
        """Test showing options without explicit directory (uses config)."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return a directory
        mock_get_dir.return_value = swift_dir

        # Call show_config_options without directory
        show_config_options(None)

        # Verify get_swiftsim_dir was called with None
        mock_get_dir.assert_called_once_with(None)

        # Verify the configure --help command was run
        mock_run_command.assert_called_once_with(
            ["./configure", "--help"], swift_dir
        )
