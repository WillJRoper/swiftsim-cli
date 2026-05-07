"""Tests for the switch mode."""

import argparse
from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.modes.switch import add_arguments, run, switch_swift_branch


class TestSwitchMode:
    """Tests for switch mode."""

    def test_add_arguments(self):
        """Test that add_arguments adds the branch argument."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        # Parser should have the branch positional argument
        # Test by parsing some args
        args = parser.parse_args(["my-branch"])
        assert args.branch == "my-branch"

    @patch("swiftsim_cli.modes.switch.switch_swift_branch")
    def test_run(self, mock_switch, tmp_path):
        """Test the run function calls switch_swift_branch correctly."""
        # Create mock args
        args = Mock()
        args.branch = "feature-branch"

        # Call run
        run(args)

        # Verify switch_swift_branch was called with correct arguments
        mock_switch.assert_called_once_with(branch="feature-branch")


class TestSwitchSwiftBranch:
    """Tests for switch_swift_branch function."""

    @patch("swiftsim_cli.modes.switch.update_current_profile_value")
    @patch("swiftsim_cli.modes.switch._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.switch.get_swiftsim_dir")
    def test_switch_branch_with_explicit_dir(
        self, mock_get_dir, mock_run_command, mock_update_profile, tmp_path
    ):
        """Test switching SWIFT branch with an explicit directory."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return the directory
        mock_get_dir.return_value = swift_dir

        # Call switch_swift_branch
        switch_swift_branch("develop", swift_dir)

        # Verify get_swiftsim_dir was called
        mock_get_dir.assert_called_once_with(swift_dir)

        # Verify the git checkout command was run
        mock_run_command.assert_called_once_with(
            ["git", "checkout", "develop"], swift_dir
        )

        # Verify the profile was updated
        mock_update_profile.assert_called_once_with("branch", "develop")

    @patch("swiftsim_cli.modes.switch.update_current_profile_value")
    @patch("swiftsim_cli.modes.switch._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.switch.get_swiftsim_dir")
    def test_switch_branch_without_dir(
        self, mock_get_dir, mock_run_command, mock_update_profile, tmp_path
    ):
        """Test switching SWIFT branch without explicit directory."""
        # Create a temp directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return a directory
        mock_get_dir.return_value = swift_dir

        # Call switch_swift_branch without directory
        switch_swift_branch("master", None)

        # Verify get_swiftsim_dir was called with None
        mock_get_dir.assert_called_once_with(None)

        # Verify the git checkout command was run
        mock_run_command.assert_called_once_with(
            ["git", "checkout", "master"], swift_dir
        )

        # Verify the profile was updated
        mock_update_profile.assert_called_once_with("branch", "master")

    @patch("swiftsim_cli.modes.switch.update_current_profile_value")
    @patch("swiftsim_cli.modes.switch._run_command_in_swift_dir")
    @patch("swiftsim_cli.modes.switch.get_swiftsim_dir")
    def test_switch_branch_propagates_errors(
        self, mock_get_dir, mock_run_command, mock_update_profile
    ):
        """Test that errors from get_swiftsim_dir are propagated."""
        # Mock get_swiftsim_dir to raise an error
        mock_get_dir.side_effect = ValueError("SWIFT directory not set")

        # Should raise the same error
        with pytest.raises(ValueError, match="SWIFT directory not set"):
            switch_swift_branch("main", None)

        # Command should not be run
        mock_run_command.assert_not_called()
        mock_update_profile.assert_not_called()
