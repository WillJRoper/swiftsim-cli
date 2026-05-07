"""Tests for the swiftsim_dir module."""

from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.swiftsim_dir import (
    _run_command_in_swift_dir,
    get_swiftsim_dir,
)


class TestGetSwiftsimDir:
    """Tests for get_swiftsim_dir function."""

    @patch("swiftsim_cli.swiftsim_dir.load_swift_profile")
    def test_get_swiftsim_dir_with_explicit_path(
        self, mock_load_profile, tmp_path
    ):
        """Test getting SWIFT dir when explicit path is provided and exists."""
        # Create a temporary directory to use as swift_dir
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock the config
        mock_config = Mock()
        mock_load_profile.return_value = mock_config

        # Call with explicit path
        result = get_swiftsim_dir(swift_dir)

        assert result == swift_dir

    @patch("swiftsim_cli.swiftsim_dir.load_swift_profile")
    def test_get_swiftsim_dir_from_config(self, mock_load_profile, tmp_path):
        """Test getting SWIFT dir from config when not explicitly provided."""
        # Create a temporary directory
        swift_dir = tmp_path / "swift_from_config"
        swift_dir.mkdir()

        # Mock the config with a swift directory
        mock_config = Mock()
        mock_config.swiftsim_dir = swift_dir
        mock_load_profile.return_value = mock_config

        # Call without explicit path
        result = get_swiftsim_dir(None)

        assert result == swift_dir

    @patch("swiftsim_cli.swiftsim_dir.load_swift_profile")
    def test_get_swiftsim_dir_not_set_in_config(self, mock_load_profile):
        """Test error when SWIFT dir not set in config and not provided."""
        # Mock config with no swift directory
        mock_config = Mock()
        mock_config.swiftsim_dir = None
        mock_load_profile.return_value = mock_config

        # Should raise ValueError
        with pytest.raises(ValueError, match="SWIFT directory not passed"):
            get_swiftsim_dir(None)

    @patch("swiftsim_cli.swiftsim_dir.load_swift_profile")
    def test_get_swiftsim_dir_does_not_exist(
        self, mock_load_profile, tmp_path
    ):
        """Test error when SWIFT directory doesn't exist."""
        # Use a path that doesn't exist
        swift_dir = tmp_path / "nonexistent"

        # Mock the config
        mock_config = Mock()
        mock_load_profile.return_value = mock_config

        # Should raise FileNotFoundError
        with pytest.raises(
            FileNotFoundError, match="SWIFT directory does not exist"
        ):
            get_swiftsim_dir(swift_dir)


class TestRunCommandInSwiftDir:
    """Tests for _run_command_in_swift_dir function."""

    @patch("swiftsim_cli.swiftsim_dir.run_command_in_dir")
    @patch("swiftsim_cli.swiftsim_dir.get_swiftsim_dir")
    def test_run_command_in_swift_dir(
        self, mock_get_dir, mock_run_command, tmp_path
    ):
        """Test running a command in SWIFT directory."""
        # Create a temporary directory
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()

        # Mock get_swiftsim_dir to return our temp directory
        mock_get_dir.return_value = swift_dir

        # Call the function
        _run_command_in_swift_dir(["make"], swift_dir)

        # Verify get_swiftsim_dir was called with the path
        mock_get_dir.assert_called_once_with(swift_dir)

        # Verify run_command_in_dir was called with command and path
        mock_run_command.assert_called_once_with(["make"], swift_dir)

    @patch("swiftsim_cli.swiftsim_dir.run_command_in_dir")
    @patch("swiftsim_cli.swiftsim_dir.get_swiftsim_dir")
    def test_run_command_propagates_errors(
        self, mock_get_dir, mock_run_command
    ):
        """Test that errors from get_swiftsim_dir are propagated."""
        # Mock get_swiftsim_dir to raise an error
        mock_get_dir.side_effect = ValueError("SWIFT directory not set")

        # Should raise the same error
        with pytest.raises(ValueError, match="SWIFT directory not set"):
            _run_command_in_swift_dir(["make"], None)

        # run_command_in_dir should not be called
        mock_run_command.assert_not_called()
