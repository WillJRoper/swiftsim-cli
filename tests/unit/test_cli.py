"""Unit tests for CLI module."""

import argparse
import subprocess
import sys
from unittest.mock import Mock

import pytest

from swiftsim_cli.cli import main
from swiftsim_cli.profile import ProfileError


def _multi_args(*modes):
    """A parsed command line running ``modes`` with default globals."""
    multi_args = Mock()
    multi_args.global_args.debug = False
    multi_args.global_args.profile = None
    multi_args.modes = list(modes)
    return multi_args


def test_main_runs_each_mode_in_order(monkeypatch):
    """main() loads each parsed mode's module and runs it in turn."""
    calls = []
    config_args, make_args = Mock(params=None), Mock(params=None)
    multi_args = _multi_args(("config", config_args), ("make", make_args))
    parsed_argv = []

    def parse(argv):
        parsed_argv.append(argv)
        return multi_args

    def load_mode(name):
        return Mock(run=lambda args: calls.append((name, args)))

    monkeypatch.setattr("swiftsim_cli.cli.MultiModeCLIArgs", parse)
    monkeypatch.setattr("swiftsim_cli.cli.load_mode", load_mode)

    main(["config", "--enable-debug", "make", "-j", "4"])

    assert parsed_argv == [["config", "--enable-debug", "make", "-j", "4"]]
    assert calls == [("config", config_args), ("make", make_args)]


def test_main_without_argv_parses_sys_argv(monkeypatch):
    """main() hands None to the parser, which then reads sys.argv."""
    seen: list[object] = []

    def parse(argv):
        seen.append(argv)
        return _multi_args()

    monkeypatch.setattr("swiftsim_cli.cli.MultiModeCLIArgs", parse)
    main()
    assert seen == [None]


def test_main_sets_the_profile_override(monkeypatch):
    """--profile NAME is applied by main(), not by the parser."""
    multi_args = _multi_args()
    multi_args.global_args.profile = "zoom"
    seen: list[str | None] = []
    monkeypatch.setattr(
        "swiftsim_cli.cli.MultiModeCLIArgs", lambda argv: multi_args
    )
    monkeypatch.setattr("swiftsim_cli.cli.set_profile_override", seen.append)
    main(["--profile", "zoom", "show"])
    assert seen == ["zoom"]


def test_unexpected_errors_propagate(monkeypatch):
    """Bugs (non-user errors) are not turned into one-line messages."""
    with pytest.raises(RuntimeError):
        _run_main_raising(RuntimeError("bug"), monkeypatch)


def _run_main_raising(exc, monkeypatch, debug=False):
    """Run main() with a single mode whose run() raises ``exc``."""
    multi_args = _multi_args(("make", Mock(params=None)))
    multi_args.global_args.debug = debug
    module = Mock()
    module.run.side_effect = exc
    monkeypatch.setattr(
        "swiftsim_cli.cli.MultiModeCLIArgs", lambda _: multi_args
    )
    monkeypatch.setattr("swiftsim_cli.cli.load_mode", lambda name: module)
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
        (EOFError(), 1, "error: no input"),
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
    # Only one-line user errors point at --debug
    assert ("--debug for the full traceback" in err) == (
        not isinstance(exc, (subprocess.CalledProcessError, KeyboardInterrupt))
    )


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


def test_startup_does_not_import_heavy_dependencies():
    """Building the parsers for help or a light mode stays cheap."""
    code = (
        "import sys\n"
        "from swiftsim_cli.cli import main\n"
        "from swiftsim_cli.multi_mode_args import MultiModeCLIArgs\n"
        "for argv in (['--help'], ['analyse', '--help'], "
        "['analyse', 'log', '--help'], ['output-times', '--help'], "
        "['new', '--help'], ['utils', 'softenings', '--help']):\n"
        "    try:\n"
        "        MultiModeCLIArgs(argv)\n"
        "    except SystemExit:\n"
        "        pass\n"
        "MultiModeCLIArgs(['make', '-j', '2'])\n"
        "heavy = ('matplotlib', 'astropy', 'swiftsimio', 'h5py', 'pandas',\n"
        "         'tree_sitter')\n"
        "print(sorted(m for m in heavy if m in sys.modules))\n"
        "print('swiftsim_cli.modes.new' in sys.modules and "
        "'swiftsim_cli.modes.profile' not in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines()[-2:] == ["[]", "True"]
