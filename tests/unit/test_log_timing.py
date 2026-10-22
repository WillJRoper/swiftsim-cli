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

    @pytest.fixture(autouse=True)
    def _inputs(self, tmp_path, monkeypatch):
        """Input paths are checked at parse time: provide them."""
        (tmp_path / "run.log").touch()
        (tmp_path / "swift").mkdir()
        monkeypatch.chdir(tmp_path)

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
        assert "warning: 2 ranks printed timers" in capsys.readouterr().err

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
        console = capsys.readouterr()
        assert "warning: no timers named not_a_timer" in console.err
        assert "Steps analysed: 2 (steps 2-3)" in console.out
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

    def test_log_without_timers_is_an_error(self, swift_src_dir, tmp_path):
        """No timer lines at all points at -v 1."""
        log = tmp_path / "quiet.log"
        log.write_text("nothing to see\n")
        with pytest.raises(ValueError, match="run SWIFT with -v 1"):
            analyse_swift_log_timings(
                str(log), output_path=tmp_path, swift_src=swift_src_dir
            )
        assert not (tmp_path / "log_analysis").exists()

    def test_top_n_validated(self, logs_dir, tmp_path):
        """--top-n must be positive."""
        with pytest.raises(ValueError, match="top-n"):
            analyse_swift_log_timings(
                str(logs_dir / "sample.txt"), output_path=tmp_path, top_n=0
            )


ROW = (
    "{n:8d}   1.0e-05    0.0078125  127.0   0.0e+00    1   56            0"
    "     100            0            0            0             {wall}"
    "    265           495.008\n"
)


class TestChecks:
    """Validation, warnings and outputs added in the second review."""

    def test_step_range_outside_the_log(
        self, logs_dir, swift_src_dir, tmp_path
    ):
        """--steps matching no step names the available range."""
        with pytest.raises(ValueError, match="the log has steps 1-4"):
            analyse_swift_log_timings(
                str(logs_dir / "sample.txt"),
                output_path=tmp_path,
                swift_src=swift_src_dir,
                step_range=(10, 20),
            )
        with pytest.raises(ValueError, match="START must not exceed END"):
            analyse_swift_log_timings(
                str(logs_dir / "sample.txt"),
                output_path=tmp_path,
                swift_src=swift_src_dir,
                step_range=(3, 2),
            )

    def test_stale_outputs_removed(
        self, logs_dir, swift_src_dir, tmp_path, fast_plots
    ):
        """Charts an earlier run wrote but this one does not are removed."""
        out_dir = tmp_path / "log_analysis"
        out_dir.mkdir()
        for name in PLOTS:
            (out_dir / name).touch()
        analyse_swift_log_timings(
            str(logs_dir / "zero_time.txt"),
            output_path=tmp_path,
            swift_src=swift_src_dir,
        )
        assert not (out_dir / "07_time_series.png").exists()
        assert not (out_dir / "per_step_coverage.csv").exists()

    def test_coverage_csv_and_outliers(
        self, logs_dir, swift_src_dir, tmp_path, fast_plots
    ):
        """Every step goes to the CSV; the text has only outliers."""
        analyse_swift_log_timings(
            str(logs_dir / "sample.txt"),
            output_path=tmp_path,
            swift_src=swift_src_dir,
        )
        out_dir = tmp_path / "log_analysis"
        rows = (out_dir / "per_step_coverage.csv").read_text().splitlines()
        assert len(rows) == 5
        tables = (out_dir / "analysis_tables.txt").read_text()
        assert "PER-STEP COVERAGE OUTLIERS (rank 0: 0 of 4" in tables

    def test_warnings(self, swift_src_dir, tmp_path, capsys, fast_plots):
        """Unmatched lines, damaged lines and >105% coverage warn."""
        log = tmp_path / "w.log"
        log.write_text(
            ROW.format(n=0, wall="1.000")
            + "[00002.0] engine_launch: (tasks) took 900.000 ms.\n"
            + "[00002.0] mystery_fn: took 300.000 ms.\n"
            + "[00002.0] engine_collect_end_of_step: took 5.0[00002.1] "
            "x: y\n"
            + "[00002.5] task_dump_all: took 400.000 ms.\n"
            + ROW.format(n=1, wall="1000.000")
        )
        analyse_swift_log_timings(
            str(log), output_path=tmp_path, swift_src=swift_src_dir
        )
        captured = capsys.readouterr()
        err = captured.err
        assert "match no timer in the SWIFT source" in err
        assert "had another line appended" in err
        assert "more than 105% of the wall-clock time in 1 step" in err
        assert "Timers between steps (not in the step time): 0.4 s" in (
            captured.out
        )

    def test_no_elapsed_stamps(self, swift_src_dir, tmp_path, fast_plots):
        """Without rank-0 elapsed stamps there is no run-time split."""
        log = tmp_path / "r1.log"
        log.write_text(
            ROW.format(n=0, wall="1.000")
            + "[0001] [00002.0] engine_launch: (tasks) took 9.000 ms.\n"
            + ROW.format(n=1, wall="10.000")
        )
        analyse_swift_log_timings(
            str(log), output_path=tmp_path, swift_src=swift_src_dir, rank=1
        )
        tables = (
            tmp_path / "log_analysis" / "analysis_tables.txt"
        ).read_text()
        assert "Run time" not in tables


def test_dense_time_series(tmp_path, fast_plots):
    """Over DENSE_STEPS steps the time series uses rolling medians."""
    import numpy as np

    from swiftsim_cli.timing.log_scan import Record, Step
    from swiftsim_cli.timing.timer_nesting import Node
    from swiftsim_cli.timing.timing_hierarchy import StepTree, aggregate

    trees = []
    for n in range(1, timing_plots.DENSE_STEPS + 50):
        step = Step(number=n, kind="step", wallclock_ms=10.0 + n % 7)
        rec = Record(0, 1.0, 9.0, "f", "function", "f", None, n)
        trees.append(StepTree(step, 0, [Node(rec)]))
    h = aggregate(trees, 0, sum(t.step.wallclock_ms or 0.0 for t in trees))
    path = timing_plots.step_time_series(trees, h, tmp_path / "t.png", False)
    assert path is not None and path.is_file()
    x, y = timing_plots._rolling_median(np.arange(10.0), np.arange(10.0), 5)
    assert list(x) == [2, 3, 4, 5, 6, 7] and list(y) == [2, 3, 4, 5, 6, 7]
    x, y = timing_plots._rolling_median(np.arange(3.0), np.arange(3.0), 5)
    assert len(y) == 3
