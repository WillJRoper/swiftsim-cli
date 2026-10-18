"""Tests for the ``analyse log`` mode."""

from pathlib import Path
from unittest.mock import patch

import pytest

from swiftsim_cli.cli import main
from swiftsim_cli.modes.analyse.log_timing import analyse_swift_log_timings
from swiftsim_cli.multi_mode_args import MultiModeCLIArgs
from swiftsim_cli.timing import timing_plots

PLOTS = [
    "01_exclusive_time.png",
    "02_step_breakdown.png",
    "03_mean_time_per_call.png",
    "04_timer_distribution.png",
    "05_task_categories.png",
    "06_call_patterns.png",
    "07_time_series.png",
    "08_timing_variability.png",
]


def _parse(argv):
    ((_, args),) = MultiModeCLIArgs(["analyse", "log", *argv]).modes
    return args


@pytest.fixture
def fast_plots(monkeypatch):
    """Skip rendering PNGs (slow) but record which would be written."""

    def fake_save(fig, path, show=False):
        import matplotlib.pyplot as plt

        plt.close(fig)
        Path(path).touch()
        return path

    monkeypatch.setattr(timing_plots, "save_figure", fake_save)


class TestArguments:
    """CLI flags of ``analyse log``."""

    def test_defaults(self):
        """Defaults: rank 0, revision matching on, cached DB."""
        args = _parse(["run.log"])
        assert args.log_file == Path("run.log")
        assert args.rank == 0
        assert args.top_n == 20
        assert args.steps is None
        assert args.hierarchy_functions is None
        assert args.match_revision is True
        assert args.regenerate_db is False
        assert args.swift_src is None

    def test_flags(self):
        """All flags parse."""
        args = _parse(
            [
                "run.log",
                "--rank",
                "all",
                "--top-n",
                "5",
                "--steps",
                "2",
                "9",
                "--hierarchy-functions",
                "engine_rebuild",
                "space_rebuild",
                "--no-match-revision",
                "--regenerate-db",
                "--swift-src",
                "swift",
            ]
        )
        assert args.rank == "all"
        assert args.top_n == 5
        assert args.steps == [2, 9]
        assert args.hierarchy_functions == ["engine_rebuild", "space_rebuild"]
        assert args.match_revision is False
        assert args.regenerate_db is True
        assert args.swift_src == Path("swift")
        assert _parse(["run.log", "--rank", "3"]).rank == 3

    def test_bad_rank(self):
        """--rank must be a number or 'all'."""
        with pytest.raises(SystemExit):
            _parse(["run.log", "--rank", "some"])

    def test_run_forwards_arguments(self):
        """The mode entry passes every option on."""
        target = (
            "swiftsim_cli.modes.analyse.log_timing.analyse_swift_log_timings"
        )
        with patch(target) as analyse:
            main(["analyse", "log", "run.log", "--steps", "1", "3"])
        kwargs = analyse.call_args.kwargs
        assert kwargs["log_file"] == "run.log"
        assert kwargs["step_range"] == (1, 3)
        assert kwargs["rank"] == 0
        assert kwargs["match_revision"] is True


class TestEndToEnd:
    """``swift-cli analyse log`` on the log fixtures."""

    def test_sample_log(self, logs_dir, swift_src_dir, tmp_path, capsys):
        """Writes the tables and charts and prints the summary."""
        main(
            [
                "analyse",
                "log",
                str(logs_dir / "sample.txt"),
                "--swift-src",
                str(swift_src_dir),
                "-o",
                str(tmp_path),
                "-p",
                "run",
            ]
        )
        out_dir = tmp_path / "log_analysis"
        for name in ["analysis_tables.txt", *PLOTS]:
            assert (out_dir / f"run_{name}").is_file(), name

        console = capsys.readouterr().out
        assert "WHERE THE STEP TIME GOES" in console
        assert "302 matched, 0 unmatched (0.0%)" in console
        assert "Steps analysed: 4 (steps 1-4)" in console
        assert "min 99.4%" in console
        assert "initialisation 23.5 s" in console

        tables = (out_dir / "run_analysis_tables.txt").read_text()
        for title in [
            "CALL TREE (rank 0, all paths)",
            "TIMERS BY EXCLUSIVE TIME",
            "CALLERS OF engine_rebuild",
            "INSIDE engine_maketasks",
            "PER-STEP COVERAGE",
            "INITIALISATION",
            "TASK CATEGORY CPU TIME",
            "TASK COUNTS",
        ]:
            assert title in tables, title
        # The old tool's failure modes are gone
        assert "SYNTHETIC" not in tables
        assert "engine_step [" not in tables

    def test_two_ranks(
        self, logs_dir, swift_src_dir, tmp_path, capsys, fast_plots
    ):
        """Default rank 0 warns; --rank all adds an imbalance table."""
        log = str(logs_dir / "mpi_v2_two_rank.txt")
        analyse_swift_log_timings(
            log, output_path=tmp_path, swift_src=swift_src_dir
        )
        assert "2 ranks printed timers" in capsys.readouterr().out

        analyse_swift_log_timings(
            log, output_path=tmp_path, swift_src=swift_src_dir, rank="all"
        )
        tables = (
            tmp_path / "log_analysis" / "analysis_tables.txt"
        ).read_text()
        assert "RANK IMBALANCE (2 ranks)" in tables
        assert "CALL TREE (rank 1, all paths)" in tables

        analyse_swift_log_timings(
            log, output_path=tmp_path, swift_src=swift_src_dir, rank=1
        )
        assert "(rank 1, top-level timers)" in capsys.readouterr().out

    def test_rank_without_timers(self, logs_dir, swift_src_dir, tmp_path):
        """Asking for a rank that printed no timers is an error."""
        with pytest.raises(ValueError, match="Rank 3 printed no timers"):
            analyse_swift_log_timings(
                str(logs_dir / "mpi4_v1.txt"),
                output_path=tmp_path,
                swift_src=swift_src_dir,
                rank=3,
            )

    def test_hierarchy_function_warnings(
        self, logs_dir, swift_src_dir, tmp_path, capsys, fast_plots
    ):
        """Unknown --hierarchy-functions names are reported."""
        analyse_swift_log_timings(
            str(logs_dir / "sample.txt"),
            output_path=tmp_path,
            swift_src=swift_src_dir,
            hierarchy_functions=["engine_unskip", "not_a_timer"],
            step_range=(2, 3),
            top_n=3,
        )
        console = capsys.readouterr().out
        assert "no timers named not_a_timer" in console
        assert "Steps analysed: 2 (steps 2-3)" in console
        tables = (
            tmp_path / "log_analysis" / "analysis_tables.txt"
        ).read_text()
        assert "CALLERS OF engine_unskip" in tables
        assert "CALLERS OF engine_rebuild" not in tables

    def test_zero_times(self, logs_dir, swift_src_dir, tmp_path, fast_plots):
        """A log of 0.000 ms timers and wall-clock times does not crash."""
        analyse_swift_log_timings(
            str(logs_dir / "zero_time.txt"),
            output_path=tmp_path,
            swift_src=swift_src_dir,
        )
        tables = (
            tmp_path / "log_analysis" / "analysis_tables.txt"
        ).read_text()
        assert "No complete steps to analyse." in tables

    def test_restart_log(
        self, logs_dir, swift_src_dir, tmp_path, capsys, fast_plots
    ):
        """Restarted runs are flagged and the 0.000 row is not a step."""
        analyse_swift_log_timings(
            str(logs_dir / "fos_restart.txt"),
            output_path=tmp_path,
            swift_src=swift_src_dir,
        )
        console = capsys.readouterr().out
        assert "[restarted run]" in console
        assert "Steps analysed: 4 (steps 47242-47245)" in console

    def test_top_n_validated(self, logs_dir, tmp_path):
        """--top-n must be positive."""
        with pytest.raises(ValueError, match="top-n"):
            analyse_swift_log_timings(
                str(logs_dir / "sample.txt"), output_path=tmp_path, top_n=0
            )
