"""Tests for the multi_mode_args module."""

import argparse
import sys
from unittest.mock import patch

import pytest

from swiftsim_cli.multi_mode_args import MultiModeCLIArgs


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
    """Test parsing global arguments."""
    argv = ["--verbose", "make", "-j", "8"]
    parser = MultiModeCLIArgs(argv)
    assert parser.global_args.verbose is True
    assert len(parser.modes) == 1
    mode_name, mode_args = parser.modes[0]
    assert mode_name == "make"
    assert mode_args.nr_threads == 8
    assert mode_args.verbose is True


def test_help_arg(capsys):
    """Test help argument handling."""
    with pytest.raises(SystemExit):
        MultiModeCLIArgs(["--help"])
    captured = capsys.readouterr()
    assert (
        "swift-cli: Utilities for Swift development workflows" in captured.out
    )

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


def test_unknown_argument():
    """Test when an unknown argument is encountered."""
    with pytest.raises(argparse.ArgumentTypeError, match="Unknown argument"):
        MultiModeCLIArgs(["unknown_command"])
