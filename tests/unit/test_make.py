"""Tests for the make mode."""

import argparse
from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.modes.make import add_arguments, make_swift, run


class TestMakeMode:
    """Tests for make mode."""

    def test_add_arguments(self):
        """Test that add_arguments adds the nr-threads argument."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        # Test default value
        args = parser.parse_args([])
        assert args.nr_threads == 1

        # Test with explicit value
        args = parser.parse_args(["-j", "8"])
        assert args.nr_threads == 8

        # Test with long form
        args = parser.parse_args(["--nr-threads", "16"])
        assert args.nr_threads == 16

    @patch("swiftsim_cli.modes.make.make_swift")
    def test_run(self, mock_make_swift, tmp_path):
        """Test the run function calls make_swift with correct arguments."""
        # Create mock args
        args = Mock()
        args.nr_threads = 8

        # Call run
        run(args)

        # Verify make_swift was called with correct arguments
        mock_make_swift.assert_called_once_with(nr_threads=8)


class TestMakeSwift:
    """Tests for make_swift function."""

    @patch("swiftsim_cli.modes.make._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.make.get_swiftsim_dir")
    def test_make_swift_single_thread(
        self, mock_get_dir, mock_run_command, tmp_path
    ):
        """Test compiling SWIFT with a single thread."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return the directory
        mock_get_dir.return_value = swift_dir

        # Call make_swift with default (single thread)
        make_swift(swift_dir, nr_threads=1)

        # Verify get_swiftsim_dir was called
        mock_get_dir.assert_called_once_with(swift_dir)

        # Verify the make command was run without -j flag
        mock_run_command.assert_called_once_with(["make"], swift_dir)

    @patch("swiftsim_cli.modes.make._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.make.get_swiftsim_dir")
    def test_make_swift_multiple_threads(
        self, mock_get_dir, mock_run_command, tmp_path
    ):
        """Test compiling SWIFT with multiple threads."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return the directory
        mock_get_dir.return_value = swift_dir

        # Call make_swift with multiple threads
        make_swift(swift_dir, nr_threads=8)

        # Verify get_swiftsim_dir was called
        mock_get_dir.assert_called_once_with(swift_dir)

        # Verify the make command was run with -j flag
        mock_run_command.assert_called_once_with(
            ["make", "-j", "8"], swift_dir
        )

    @patch("swiftsim_cli.modes.make._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.make.get_swiftsim_dir")
    def test_make_swift_without_dir(
        self, mock_get_dir, mock_run_command, tmp_path
    ):
        """Test compiling SWIFT without explicit directory (uses config)."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return a directory
        mock_get_dir.return_value = swift_dir

        # Call make_swift without directory
        make_swift(None, nr_threads=4)

        # Verify get_swiftsim_dir was called with None
        mock_get_dir.assert_called_once_with(None)

        # Verify the make command was run with -j flag
        mock_run_command.assert_called_once_with(
            ["make", "-j", "4"], swift_dir
        )

    @patch("swiftsim_cli.modes.make._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.make.get_swiftsim_dir")
    def test_make_swift_invalid_thread_count(
        self, mock_get_dir, mock_run_command, tmp_path
    ):
        """Test that invalid thread count raises ValueError."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return the directory
        mock_get_dir.return_value = swift_dir

        # Should raise ValueError for thread count < 1
        with pytest.raises(
            ValueError, match="Number of threads must be at least 1"
        ):
            make_swift(swift_dir, nr_threads=0)

        with pytest.raises(
            ValueError, match="Number of threads must be at least 1"
        ):
            make_swift(swift_dir, nr_threads=-1)

        # Command should not be run
        mock_run_command.assert_not_called()

    @patch("swiftsim_cli.modes.make._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.make.get_swiftsim_dir")
    def test_make_swift_propagates_errors(
        self, mock_get_dir, mock_run_command
    ):
        """Test that errors from get_swiftsim_dir are propagated."""
        # Mock get_swiftsim_dir to raise an error
        mock_get_dir.side_effect = ValueError("SWIFT directory not set")

        # Should raise the same error
        with pytest.raises(ValueError, match="SWIFT directory not set"):
            make_swift(None, nr_threads=4)

        # Command should not be run
        mock_run_command.assert_not_called()
