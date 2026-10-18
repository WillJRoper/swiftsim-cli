"""Tests for the analyse scaling subcommand CLI."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import pytest

from swiftsim_cli.modes.analyse import scaling


def _scaling_parser() -> ArgumentParser:
    """Return a parser accepting ``scaling <timers|runtime> ...``."""
    parser = ArgumentParser()
    scaling.add_scaling_arguments(parser.add_subparsers(dest="analysis"))
    return parser


class TestScalingArguments:
    """Tests for scaling CLI argument setup."""

    def test_add_arguments_for_timers(self):
        """The timers subcommand accepts the expected options."""
        parser = _scaling_parser()

        args = parser.parse_args(
            [
                "scaling",
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
        parser = _scaling_parser()

        args = parser.parse_args(
            ["scaling", "runtime", "a.log", "b.log", "--show"]
        )

        assert args.scaling_type == "runtime"
        assert args.log_files == [Path("a.log"), Path("b.log")]
        assert args.show is True
        assert args.steps is None


@pytest.mark.parametrize(
    ("kind", "target"),
    [
        ("timers", "run_swift_scaling"),
        ("runtime", "run_swift_runtime_scaling"),
    ],
)
def test_subcommands_bind_their_entry_points(kind, target, in_inputs_dir):
    """Each scaling subcommand binds its entry point as ``func``."""
    with patch(f"swiftsim_cli.modes.analyse.scaling.{target}") as mock_run:
        args = _scaling_parser().parse_args(
            ["scaling", kind, "a.log", "b.log"]
        )
        args.func(args)

    mock_run.assert_called_once_with(args)


def test_unmatched_log_pattern_is_an_error(in_inputs_dir):
    """A log file or pattern matching nothing is a clear error."""
    with pytest.raises(ValueError, match="no log file matches 'c\\*.log'"):
        scaling._resolve_log_files([Path("a.log"), Path("c*.log")])


def test_extract_total_runtime_from_log(tmp_path: Path):
    """Runtime extraction sums the header-located wallclock column."""
    log_file = tmp_path / "swift.log"
    # An older 13-column table (no sink-/b-Updates), so the wall-clock
    # column is the 11th, not the 13th.
    log_file.write_text(
        "\n".join(
            [
                "[0000] main: MPI is up and running with 4 node(s).",
                "#   Step  Time Scale-factor Redshift Time-step Time-bins "
                "Updates g-Updates s-Updates Wall-clock time [ms] Props "
                "Dead time [ms]",
                "  1 0.0 0.0 0.0 0.0 1 56 0 0 0 10.0 265 99.0",
                "  2 0.0 0.0 0.0 0.0 1 56 0 0 0 20.0 265 99.0",
                "  3 0.0 0.0 0.0 0.0 1 56 0 0 0 30.0 265 99.0",
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
