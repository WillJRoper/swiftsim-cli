"""Tests for the top-level scaling mode."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from swiftsim_cli.modes import scaling


class TestScalingArguments:
    """Tests for scaling CLI argument setup."""

    def test_add_arguments_for_timers(self):
        """The timers subcommand accepts the expected options."""
        parser = ArgumentParser()
        scaling.add_arguments(parser)

        args = parser.parse_args(
            [
                "timers",
                "a.log",
                "b.log",
                "--steps",
                "1",
                "20",
                "--rank-source",
                "all",
            ]
        )

        assert args.scaling_type == "timers"
        assert args.log_files == [Path("a.log"), Path("b.log")]
        assert args.steps == [1, 20]
        assert args.rank_source == "all"

    def test_add_arguments_for_runtime(self):
        """The runtime subcommand accepts the shared scaling options."""
        parser = ArgumentParser()
        scaling.add_arguments(parser)

        args = parser.parse_args(["runtime", "a.log", "b.log", "--show"])

        assert args.scaling_type == "runtime"
        assert args.log_files == [Path("a.log"), Path("b.log")]
        assert args.show is True
        assert args.steps is None


class TestScalingRun:
    """Tests for scaling subcommand dispatch."""

    @patch("swiftsim_cli.modes.scaling.run_swift_scaling")
    def test_run_timers(self, mock_run_timers):
        """Timer scaling dispatches to the detailed timer implementation."""
        args = Namespace(scaling_type="timers")

        scaling.run(args)

        mock_run_timers.assert_called_once_with(args)

    @patch("swiftsim_cli.modes.scaling.run_swift_runtime_scaling")
    def test_run_runtime(self, mock_run_runtime):
        """Runtime scaling dispatches to the runtime implementation."""
        args = Namespace(scaling_type="runtime")

        scaling.run(args)

        mock_run_runtime.assert_called_once_with(args)

    def test_run_unknown_type(self):
        """Unknown scaling subcommands are rejected."""
        args = Namespace(scaling_type="unknown")

        with pytest.raises(ValueError, match="Unknown scaling type"):
            scaling.run(args)


def test_extract_total_runtime_from_log(tmp_path: Path):
    """Runtime extraction sums the step-table wallclock column."""
    log_file = tmp_path / "swift.log"
    log_file.write_text(
        "\n".join(
            [
                "[0000] main: MPI is up and running with 4 node(s).",
                "1 0.0 0.0 0.0 0.0 0 0 0 0 0 0 0 10.0",
                "2 0.0 0.0 0.0 0.0 0 0 0 0 0 0 0 20.0",
                "3 0.0 0.0 0.0 0.0 0 0 0 0 0 0 0 30.0",
            ]
        ),
        encoding="utf-8",
    )

    total_runtime_ms, step_count = scaling._extract_total_runtime_from_log(
        str(log_file), (1, 2)
    )

    assert total_runtime_ms == pytest.approx(30.0)
    assert step_count == 2


def test_resolve_log_files_expands_glob_patterns(tmp_path: Path):
    """Scaling input resolution expands shell-style patterns itself."""
    first = tmp_path / "log_a.txt"
    second = tmp_path / "log_b.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    resolved = scaling._resolve_log_files([tmp_path / "log_*.txt"])

    assert resolved == [str(first), str(second)]


def test_resolve_log_files_keeps_explicit_existing_files(tmp_path: Path):
    """Explicit file paths are preserved without glob expansion."""
    log_file = tmp_path / "swift.log"
    log_file.write_text("content", encoding="utf-8")

    resolved = scaling._resolve_log_files([log_file])

    assert resolved == [str(log_file)]
