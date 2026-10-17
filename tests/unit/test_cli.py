"""Unit tests for CLI module."""

from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.cli import main


class TestCLI:
    """Test CLI functionality."""

    @patch("swiftsim_cli.cli.MultiModeCLIArgs")
    @patch("swiftsim_cli.cli.load_parameters")
    def test_main_basic(self, mock_load_params, mock_multi_args):
        """Test basic CLI main function."""
        # Mock the multi-mode args
        mock_args = Mock()
        mock_multi_args_instance = Mock()
        mock_multi_args_instance.modes = [("analyse", mock_args)]
        mock_multi_args.return_value = mock_multi_args_instance

        # Mock the mode module
        mock_module = Mock()

        with patch("swiftsim_cli.cli.MODE_MODULES", {"analyse": mock_module}):
            # Should not raise exception
            main(["analyse", "timesteps", "test.log"])

            mock_multi_args.assert_called_once_with(
                ["analyse", "timesteps", "test.log"]
            )
            mock_module.run.assert_called_once_with(mock_args)

    @patch("swiftsim_cli.cli.MultiModeCLIArgs")
    @patch("swiftsim_cli.cli.load_parameters")
    def test_main_no_args(self, mock_load_params, mock_multi_args):
        """Test CLI main function with no arguments."""
        mock_multi_args_instance = Mock()
        mock_multi_args_instance.modes = []
        mock_multi_args.return_value = mock_multi_args_instance

        main()
        mock_multi_args.assert_called_once_with(None)

    @patch("swiftsim_cli.cli.MultiModeCLIArgs")
    @patch("swiftsim_cli.cli.load_parameters")
    def test_main_with_args(self, mock_load_params, mock_multi_args):
        """Test CLI main function with specific arguments."""
        mock_multi_args_instance = Mock()
        mock_multi_args_instance.modes = []
        mock_multi_args.return_value = mock_multi_args_instance

        test_args = ["config", "--enable-debug"]
        main(test_args)
        mock_multi_args.assert_called_once_with(test_args)

    @patch("swiftsim_cli.cli.MultiModeCLIArgs")
    @patch("swiftsim_cli.cli.load_parameters")
    def test_main_multiple_modes(self, mock_load_params, mock_multi_args):
        """Test CLI with multiple modes."""
        # Mock args for multiple modes
        mock_config_args = Mock()
        mock_make_args = Mock()
        mock_multi_args_instance = Mock()
        mock_multi_args_instance.modes = [
            ("config", mock_config_args),
            ("make", mock_make_args),
        ]
        mock_multi_args.return_value = mock_multi_args_instance

        # Mock the mode modules
        mock_config_module = Mock()
        mock_make_module = Mock()

        with patch(
            "swiftsim_cli.cli.MODE_MODULES",
            {"config": mock_config_module, "make": mock_make_module},
        ):
            main(["config", "--enable-debug", "make", "-j", "4"])

            mock_multi_args.assert_called_once()
            mock_config_module.run.assert_called_once_with(mock_config_args)
            mock_make_module.run.assert_called_once_with(mock_make_args)

    @patch("swiftsim_cli.cli.MultiModeCLIArgs")
    @patch("swiftsim_cli.cli.load_parameters")
    def test_main_mode_execution_error(
        self, mock_load_params, mock_multi_args
    ):
        """Test CLI main function with mode execution error."""
        mock_args = Mock()
        mock_multi_args_instance = Mock()
        mock_multi_args_instance.modes = [("analyse", mock_args)]
        mock_multi_args.return_value = mock_multi_args_instance

        mock_module = Mock()
        mock_module.run.side_effect = Exception("Mode execution failed")

        with patch("swiftsim_cli.cli.MODE_MODULES", {"analyse": mock_module}):
            with pytest.raises(Exception):
                main(["analyse", "timesteps", "test.log"])
