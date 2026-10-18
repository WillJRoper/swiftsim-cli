"""Unit tests for CLI module."""

import argparse
import subprocess
from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.cli import main
from swiftsim_cli.profile import ProfileError


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


def _run_main_raising(exc, monkeypatch, debug=False):
    """Run main() with a single mode whose run() raises ``exc``."""
    multi_args = Mock()
    multi_args.global_args.debug = debug
    multi_args.modes = [("make", Mock(params=None))]
    module = Mock()
    module.run.side_effect = exc
    monkeypatch.setattr(
        "swiftsim_cli.cli.MultiModeCLIArgs", lambda _: multi_args
    )
    monkeypatch.setattr("swiftsim_cli.cli.MODE_MODULES", {"make": module})
    main(["make"])


@pytest.mark.parametrize(
    ("exc", "code", "message"),
    [
        (ValueError("bad value"), 1, "bad value"),
        (ProfileError("no profile"), 1, "no profile"),
        (FileNotFoundError("missing.txt"), 1, "missing.txt"),
        (FileExistsError("exists.txt"), 1, "exists.txt"),
        (NotADirectoryError("notdir"), 1, "notdir"),
        (PermissionError("denied"), 1, "denied"),
        (argparse.ArgumentTypeError("bad arg"), 1, "bad arg"),
        (subprocess.CalledProcessError(3, ["make"]), 3, "'['make']'"),
        (subprocess.CalledProcessError(-9, ["make"]), 137, "SIGKILL"),
        (KeyboardInterrupt(), 130, "interrupted"),
    ],
)
def test_main_reports_user_errors(exc, code, message, monkeypatch, capsys):
    """User-facing errors print 'error: <message>' and set the exit code."""
    monkeypatch.delenv("SWIFT_CLI_DEBUG", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        _run_main_raising(exc, monkeypatch)
    assert excinfo.value.code == code
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert message in err


def test_main_debug_flag_reraises(monkeypatch):
    """--debug shows the original exception instead of a message."""
    monkeypatch.delenv("SWIFT_CLI_DEBUG", raising=False)
    with pytest.raises(ValueError, match="bad value"):
        _run_main_raising(ValueError("bad value"), monkeypatch, debug=True)


def test_main_debug_env_reraises(monkeypatch):
    """SWIFT_CLI_DEBUG=1 shows the original exception too."""
    monkeypatch.setenv("SWIFT_CLI_DEBUG", "1")
    with pytest.raises(KeyboardInterrupt):
        _run_main_raising(KeyboardInterrupt(), monkeypatch)


def test_main_debug_flag_end_to_end(tmp_path, capsys):
    """The real parser wires --debug through to main()."""
    missing = tmp_path / "missing.yml"
    with pytest.raises(SystemExit) as excinfo:
        main(["output-times", "-P", str(missing)])
    assert excinfo.value.code == 1
    assert "error: Parameter file not found" in capsys.readouterr().err

    with pytest.raises(FileNotFoundError):
        main(["--debug", "output-times", "-P", str(missing)])
