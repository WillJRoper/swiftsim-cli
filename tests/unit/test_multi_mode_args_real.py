"""Tests for the multi_mode_args module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from swiftsim_cli import __version__
from swiftsim_cli.multi_mode_args import MultiModeCLIArgs, top_level_help

# Input paths are checked at parse time; run in a directory that has them
pytestmark = pytest.mark.usefixtures("in_inputs_dir")


def test_single_mode_make():
    """Test parsing a single mode command."""
    argv = ["make", "-j", "8"]
    parser = MultiModeCLIArgs(argv)
    assert len(parser.modes) == 1
    mode_name, mode_args = parser.modes[0]
    assert mode_name == "make"
    assert mode_args.nr_threads == 8


def test_multiple_modes_config_make():
    """Test parsing multiple chained modes."""
    argv = ["config", "--hydro-scheme=gizmo-mfv", "make", "-j", "16"]
    parser = MultiModeCLIArgs(argv)
    assert len(parser.modes) == 2

    mode1_name, mode1_args = parser.modes[0]
    assert mode1_name == "config"
    assert "--hydro-scheme=gizmo-mfv" in mode1_args.options

    mode2_name, mode2_args = parser.modes[1]
    assert mode2_name == "make"
    assert mode2_args.nr_threads == 16


def test_global_args():
    """Global options before the first mode are parsed separately."""
    argv = ["--debug", "--profile", "other", "make", "-j", "8"]
    parser = MultiModeCLIArgs(argv)
    assert parser.global_args.debug is True
    assert parser.global_args.profile == "other"
    assert len(parser.modes) == 1
    mode_name, mode_args = parser.modes[0]
    assert mode_name == "make"
    assert mode_args.nr_threads == 8


def test_debug_after_mode_is_not_global():
    """--debug after a mode is not the global flag (config passes it on)."""
    parser = MultiModeCLIArgs(["config", "--debug"])
    assert parser.global_args.debug is False
    assert parser.modes[0][1].options == ["--debug"]


def test_help_arg(capsys):
    """Test help argument handling."""
    with pytest.raises(SystemExit):
        MultiModeCLIArgs(["--help"])
    captured = capsys.readouterr()
    assert captured.out == top_level_help() + "\n"

    with pytest.raises(SystemExit):
        MultiModeCLIArgs(["make", "--help"])
    captured = capsys.readouterr()
    assert "usage: swift-cli make" in captured.out


def test_argv_none():
    """Test when argv is None, it should use sys.argv."""
    test_argv = ["swift-cli", "make", "-j", "4"]
    with patch.object(sys, "argv", test_argv):
        parser = MultiModeCLIArgs(None)
        assert len(parser.modes) == 1
        mode_name, mode_args = parser.modes[0]
        assert mode_name == "make"
        assert mode_args.nr_threads == 4


@pytest.mark.parametrize(
    "argv", [["unknown_command"], ["--verbose", "make"], ["-x"]]
)
def test_unknown_leading_argument_is_a_usage_error(argv, capsys):
    """Unknown leading arguments print usage and exit 2 like argparse."""
    with pytest.raises(SystemExit) as excinfo:
        MultiModeCLIArgs(argv)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "usage: swift-cli" in err
    assert f"unknown mode or option '{argv[0]}'" in err


@pytest.mark.parametrize("argv", [[], ["--debug"], ["--profile", "x"]])
def test_no_mode_prints_help_and_exits_1(argv, capsys):
    """Without a mode the top-level help is printed and the exit code is 1."""
    with pytest.raises(SystemExit) as excinfo:
        MultiModeCLIArgs(argv)
    assert excinfo.value.code == 1
    assert "Available modes:" in capsys.readouterr().out


def test_version(capsys):
    """--version prints the package version and exits 0."""
    with pytest.raises(SystemExit) as excinfo:
        MultiModeCLIArgs(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"swift-cli {__version__}"


def _modes(argv):
    """Return the (mode, args) pairs parsed from argv."""
    return MultiModeCLIArgs(argv).modes


def test_mode_name_as_option_value():
    """A mode name given as an option's value does not start a mode."""
    (name, args), (name2, args2) = _modes(
        ["new", "--out", "make", "--ics", "ics.hdf5", "make", "-j", "2"]
    )
    assert name == "new"
    assert args.out == Path("make")
    assert args.ics == Path("ics.hdf5")
    assert (name2, args2.nr_threads) == ("make", 2)


def test_mode_name_as_positional_value():
    """A mode name filling a positional argument does not start a mode."""
    ((name, args),) = _modes(["switch", "show"])
    assert (name, args.name) == ("switch", "show")


def test_mode_name_as_short_option_value():
    """Short options with values are recognised too."""
    ((name, args),) = _modes(["new", "-o", "new", "--ics", "ics.hdf5"])
    assert (name, args.out) == ("new", Path("new"))


def test_mode_name_as_equals_value_and_profile_value():
    """--opt=value forms and global option values are handled."""
    modes = _modes(["--profile", "make", "make", "--nr-threads=3", "show"])
    assert [name for name, _ in modes] == ["make", "show"]
    assert modes[0][1].nr_threads == 3


def test_mode_name_as_nargs_option_value():
    """Fixed multi-value options (--steps START END) take all values."""
    ((name, args),) = _modes(
        ["analyse", "mpiuse", "reports", "--steps", "1", "2"]
    )
    assert args.steps == [1, 2]


def test_real_chain_still_splits():
    """A real chain of modes is split into its sections."""
    modes = _modes(
        [
            "config",
            "--enable-debug",
            "make",
            "-j",
            "4",
            "new",
            "-o",
            "x",
            "--ics",
            "ics.hdf5",
        ]
    )
    assert [name for name, _ in modes] == ["config", "make", "new"]
    assert modes[0][1].options == ["--enable-debug"]
    assert modes[1][1].nr_threads == 4
    assert (modes[2][1].out, modes[2][1].ics) == (Path("x"), Path("ics.hdf5"))


def test_mode_name_ends_variable_length_arguments():
    """A mode name after nargs='+' values starts the next mode."""
    modes = _modes(
        [
            "analyse",
            "timesteps",
            "a.txt",
            "b.txt",
            "-l",
            "x",
            "y",
            "make",
            "-j",
            "2",
        ]
    )
    assert [name for name, _ in modes] == ["analyse", "make"]
    assert modes[0][1].files == [Path("a.txt"), Path("b.txt")]
    assert modes[0][1].labels == ["x", "y"]


def test_mode_name_ends_optional_value():
    """A mode name after an option with an optional value starts a mode."""
    modes = _modes(["profile", "--show", "make"])
    assert [name for name, _ in modes] == ["profile", "make"]
    assert modes[0][1].show == ""


def test_nested_subcommands_are_followed():
    """Subcommands of analyse (and scaling) are scanned with their parser."""
    modes = _modes(
        [
            "analyse",
            "scaling",
            "timers",
            "a.log",
            "b.log",
            "-o",
            "show",
            "show",
        ]
    )
    assert [name for name, _ in modes] == ["analyse", "show"]
    assert modes[0][1].scaling_type == "timers"
    assert modes[0][1].out == Path("show")


def test_terminator_ends_a_mode():
    """'--' ends the current mode's arguments explicitly."""
    modes = _modes(["config", "--enable-debug", "--", "make", "-j", "2"])
    assert [name for name, _ in modes] == ["config", "make"]
    assert modes[0][1].options == ["--enable-debug"]
    assert modes[1][1].nr_threads == 2


def test_terminator_must_be_followed_by_a_mode(capsys):
    """A non-mode token after '--' is a usage error."""
    with pytest.raises(SystemExit) as excinfo:
        MultiModeCLIArgs(["make", "--", "-j", "2"])
    assert excinfo.value.code == 2
    assert "expected a mode after '--', got '-j'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["analyse", "gravity-error-maps", "a.dat", "--res", "5"],
        ["make", "--nr", "2"],
        ["analyse", "log", "swift.log", "--top", "5"],
    ],
)
def test_abbreviated_options_are_rejected(argv):
    """Long options must be spelled out (allow_abbrev=False everywhere)."""
    with pytest.raises(SystemExit) as excinfo:
        MultiModeCLIArgs(argv)
    assert excinfo.value.code == 2
