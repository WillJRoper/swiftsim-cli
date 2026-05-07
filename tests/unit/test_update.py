"""Tests for the update mode."""

import argparse
from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.modes.update import add_arguments, run, update_swift


class TestUpdateMode:
    """Tests for update mode."""

    def test_add_arguments(self):
        """Test that add_arguments does nothing (no arguments needed)."""
        parser = argparse.ArgumentParser()
        # Should not raise any errors
        add_arguments(parser)
        # Parser should not have any new actions beyond the default help
        assert len(parser._actions) == 1  # Only help action

    @patch("swiftsim_cli.modes.update.update_swift")
    def test_run(self, mock_update_swift, tmp_path):
        """Test the run function calls update_swift with correct arguments."""
        # Create mock args
        args = Mock()

        # Call run
        run(args)

        # Verify update_swift was called without arguments
        mock_update_swift.assert_called_once_with()


class TestUpdateSwift:
    """Tests for update_swift function."""

    @patch("swiftsim_cli.modes.update._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.update.get_swiftsim_dir")
    def test_update_swift_with_explicit_dir(
        self, mock_get_dir, mock_run_command, tmp_path
    ):
        """Test updating SWIFT with an explicit directory."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return the directory
        mock_get_dir.return_value = swift_dir

        # Call update_swift
        update_swift(swift_dir)

        # Verify get_swiftsim_dir was called
        mock_get_dir.assert_called_once_with(swift_dir)

        # Verify the git pull command was run
        mock_run_command.assert_called_once_with(["git", "pull"], swift_dir)

    @patch("swiftsim_cli.modes.update._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.update.get_swiftsim_dir")
    def test_update_swift_without_dir(
        self, mock_get_dir, mock_run_command, tmp_path
    ):
        """Test updating SWIFT without explicit directory (uses config)."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return a directory
        mock_get_dir.return_value = swift_dir

        # Call update_swift without directory
        update_swift(None)

        # Verify get_swiftsim_dir was called with None
        mock_get_dir.assert_called_once_with(None)

        # Verify the git pull command was run
        mock_run_command.assert_called_once_with(["git", "pull"], swift_dir)

    @patch("swiftsim_cli.modes.update._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.update.get_swiftsim_dir")
    def test_update_swift_propagates_errors(
        self, mock_get_dir, mock_run_command
    ):
        """Test that errors from get_swiftsim_dir are propagated."""
        # Mock get_swiftsim_dir to raise an error
        mock_get_dir.side_effect = ValueError("SWIFT directory not set")

        # Should raise the same error
        with pytest.raises(ValueError, match="SWIFT directory not set"):
            update_swift(None)

        # Command should not be run
        mock_run_command.assert_not_called()
