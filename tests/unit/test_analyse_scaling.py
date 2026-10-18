"""Tests for ``analyse scaling`` (timers and runtime)."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import pytest

from swiftsim_cli.modes.analyse import scaling
from swiftsim_cli.modes.analyse.scaling import data, plots, timers
from swiftsim_cli.modes.analyse.scaling.data import (
    LogTimers,
    describe_slope,
    fit_slope,
    format_time,
    load_log_runtime,
    load_log_timers,
    require_rank_counts,
    resolve_log_files,
)

ROW = (
    "{n:8d}   1.0e-05    0.0078125  127.0   0.0e+00    1   56            0"
    "     100            0            0            0             {wall}"
    "    265           495.008\n"
)


@pytest.fixture
def synthetic_scaling_db(monkeypatch, synthetic_db):
    """Analyse the fixtures with the synthetic timer database."""
    monkeypatch.setattr(
        data, "resolve_timer_db", lambda *a, **k: (synthetic_db, "test")
    )


@pytest.fixture
def fast_plots(monkeypatch):
    """Record the plots instead of rendering PNGs."""

    def fake_save(fig, path, show=False):
        import matplotlib.pyplot as plt

        plt.close(fig)
        Path(path).touch()
        return path

    monkeypatch.setattr(plots, "save_figure", fake_save)


def _write_log(path: Path, ranks: int, step_ms: list[float], extra="") -> Path:
    """A log with an MPI header, row 0 and one engine_launch per step."""
    lines = [
        f"[0000] [00000.0] main: MPI is up and running with {ranks} node(s).",
        ROW.format(n=0, wall="5000.000").rstrip(),
    ]
    for i, ms in enumerate(step_ms, 1):
        lines.append(
            f"[0000] [{i:07.1f}] engine_launch: (tasks) took {ms:.3f} ms."
        )
        lines.append(extra.format(i=i, ms=ms)) if extra else None
        lines.append(ROW.format(n=i, wall=f"{ms * 1.25:.3f}").rstrip())
    path.write_text("\n".join(lines) + "\n")
    return path


def _logs(tmp_path, extra=""):
    return [
        str(_write_log(tmp_path / "a.log", 2, [800.0, 800.0], extra)),
        str(_write_log(tmp_path / "b.log", 4, [400.0, 400.0], extra)),
        str(_write_log(tmp_path / "c.log", 8, [200.0, 200.0], extra)),
    ]


class TestLogTimers:
    """Per-log statistics."""

    def test_exclusive_totals(self, synthetic_scaling_db, logs_dir):
        """Totals are exclusive times of the steps analysed (no step 0)."""
        log = load_log_timers(str(logs_dir / "sample.txt"))
        assert log.n_ranks == 1 and log.n_steps == 4
        assert log.step_time == pytest.approx(21399.138)
        rank0 = log.exclusive[0]
        assert rank0["engine_rebuild"] == pytest.approx(2.8, abs=0.1)
        assert sum(rank0.values()) <= log.step_time
        assert log.calls[0]["engine_launch (tasks)"] == 4
        assert 90 < log.total("sum", "root") / log.step_time * 100 <= 100
        assert log.unmatched == 0.0

    def test_steps_and_ranks(self, synthetic_scaling_db, logs_dir):
        """--steps limits the steps; ranks are kept apart."""
        log = load_log_timers(str(logs_dir / "sample.txt"), (2, 2))
        assert log.calls[0]["engine_launch (tasks)"] == 1
        assert log.step_time == pytest.approx(7477.584)
        v2 = load_log_timers(str(logs_dir / "mpi_v2_two_rank.txt"))
        assert set(v2.exclusive) == {0, 1}
        key = "engine_launch (tasks)"
        e0, e1 = v2.exclusive[0][key], v2.exclusive[1][key]
        # 'all' is the mean over ranks, so 1/N stays the ideal
        assert v2.series("sum", "all")[key] == pytest.approx((e0 + e1) / 2)
        assert v2.series("sum", "root")[key] == pytest.approx(e0)
        calls = v2.calls[0][key] + v2.calls[1][key]
        assert v2.series("mean", "all")[key] == pytest.approx(
            (e0 + e1) / calls
        )
        assert v2.total("mean", "all") > 0
        with pytest.raises(ValueError, match="no complete steps within"):
            load_log_timers(str(logs_dir / "sample.txt"), (50, 60))

    def test_view_helpers(self):
        """Percent, totals and per-call means of a hand-made log."""
        log = LogTimers(
            "x",
            "x",
            2,
            1,
            100.0,
            {0: {"a": 10.0}, 1: {"a": 30.0}},
            {0: {"a": 2}, 1: {"a": 2}},
        )
        assert log.percent("a", "root") == 10.0
        assert log.percent("a", "all") == 20.0
        assert log.total("mean", "root") == 5.0
        empty = LogTimers("y", "y", 1, 0, 0.0)
        assert empty.percent("a", "root") == 0.0
        assert empty.total("mean", "root") == 0.0


class TestHelpers:
    """Small pure helpers."""

    def test_slopes(self):
        """log-log slope and its description."""
        assert fit_slope([1, 2, 4], [8, 4, 2]) == pytest.approx(-1.0)
        assert fit_slope([4, 4], [1, 2]) is None
        assert [describe_slope(s) for s in (None, -1.2, -0.9, -0.5, 0, 1)] == [
            "insufficient",
            "superlinear",
            "strong",
            "moderate",
            "weak",
            "regressing",
        ]
        assert format_time(1500) == "1.50s" and format_time(15) == "15.0ms"

    def test_rank_counts(self):
        """Fewer than two distinct rank counts is an error."""
        require_rank_counts([2, 4])
        with pytest.raises(ValueError, match="two different MPI rank"):
            require_rank_counts([16, 16])

    def test_resolve_log_files(self, tmp_path):
        """Explicit files are kept, patterns expanded, misses errors."""
        first, second = tmp_path / "log_a.txt", tmp_path / "log_b.txt"
        first.touch()
        second.touch()
        assert resolve_log_files([tmp_path / "log_*.txt"]) == [
            str(first),
            str(second),
        ]
        assert resolve_log_files([first]) == [str(first)]
        with pytest.raises(ValueError, match="no log file matches"):
            resolve_log_files([tmp_path / "c*.log"])

    def test_select_timers(self):
        """A timer is kept if it crosses the threshold in any log."""
        a = LogTimers(
            "a",
            "a",
            2,
            1,
            100.0,
            {0: {"x": 50.0, "y": 5.0}},
            {0: {"x": 1, "y": 1}},
        )
        b = LogTimers(
            "b",
            "b",
            4,
            1,
            100.0,
            {0: {"x": 4.0, "y": 12.0}},
            {0: {"x": 1, "y": 1}},
        )
        assert timers.select_timers([a, b], 10.0, "root") == ["x", "y"]
        assert timers.select_timers([a, b], 20.0, "root") == ["x"]


class TestRuntime:
    """``scaling runtime``."""

    def test_runtime_excludes_step_zero(self, tmp_path):
        """Runtime is the step time analyse log uses (no init row)."""
        log = load_log_runtime(
            str(_write_log(tmp_path / "a.log", 4, [100.0, 300.0]))
        )
        assert log.n_ranks == 4 and log.n_steps == 2
        assert log.runtime == pytest.approx(500.0)
        assert load_log_runtime(str(tmp_path / "a.log"), (2, 2)).runtime == (
            pytest.approx(375.0)
        )
        with pytest.raises(ValueError, match="within requested steps 9-9"):
            load_log_runtime(str(tmp_path / "a.log"), (9, 9))

    def test_old_table_layout(self, tmp_path):
        """The wall-clock column is located from the header."""
        log = tmp_path / "old.log"
        log.write_text(
            "\n".join(
                [
                    "[0000] [00000.0] main: MPI is up and running with 4 "
                    "node(s).",
                    "#   Step  Time Scale-factor Redshift Time-step Time-bins "
                    "Updates g-Updates s-Updates Wall-clock time [ms] Props "
                    "Dead time [ms]",
                    "  1 0.0 0.0 0.0 0.0 1 56 0 0 0 10.0 265 99.0",
                    "  2 0.0 0.0 0.0 0.0 1 56 0 0 0 20.0 265 99.0",
                ]
            )
        )
        assert load_log_runtime(str(log)).runtime == pytest.approx(30.0)

    def test_end_to_end(self, tmp_path, capsys):
        """Summary, plot and the rank-count check."""
        logs = _logs(tmp_path)
        out = tmp_path / "out"
        scaling.analyse_swift_runtime_scaling(
            logs, output_path=out, step_range=(1, 2)
        )
        text = capsys.readouterr().out
        assert "RUNTIME SCALING SUMMARY" in text and "4.00x" in text
        assert "slope: -1.00" in text and "Only steps 1-2" in text
        out_dir = out / "scaling_runtime_analysis"
        assert (out_dir / "runtime_scaling.png").is_file()
        with pytest.raises(ValueError, match="two different MPI rank"):
            scaling.analyse_swift_runtime_scaling(
                [logs[0], logs[0]], output_path=out
            )
        with pytest.raises(ValueError, match="at least two"):
            scaling.analyse_swift_runtime_scaling([logs[0]])
        with pytest.raises(ValueError, match="START END"):
            scaling.analyse_swift_runtime_scaling(logs, step_range=(2, 1))


class TestTimers:
    """``scaling timers`` end to end on synthetic logs."""

    def test_outputs_and_tables(
        self, synthetic_scaling_db, tmp_path, capsys, fast_plots
    ):
        """Tables, notes and the two main plots; stale plots removed."""
        logs = _logs(tmp_path)
        out_dir = tmp_path / "out" / "scaling_timers_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "timer_scaling_rank_0001.png").touch()
        (out_dir / "timer_rank_scaling_old.png").touch()
        (out_dir / "keep.png").touch()
        scaling.analyse_swift_scaling(logs, output_path=tmp_path / "out")
        text = capsys.readouterr().out
        assert "SCALING INPUTS" in text and "Unmatched lines" in text
        assert "Total (timed)" in text and "80.0%" in text
        assert "engine_launch (tasks)" in text and "strong" in text
        assert "No total in the per-call view" in text
        assert sorted(p.name for p in out_dir.iterdir()) == [
            "keep.png",
            "scaling_summary.txt",
            "timer_scaling_mean_root.png",
            "timer_scaling_sum_root.png",
        ]

    def test_rank_source_all_and_v2_outputs(
        self, synthetic_scaling_db, tmp_path, capsys, fast_plots
    ):
        """-v 2 logs: per-rank plots, timer-by-rank plots and imbalance."""
        extra = "[0001] [{i:07.1f}] engine_launch: (tasks) took {ms:.3f} ms."
        logs = _logs(tmp_path, extra)
        scaling.analyse_swift_scaling(
            logs, output_path=tmp_path, rank_source="all", prefix="p"
        )
        captured = capsys.readouterr()
        assert "PER-RANK IMBALANCE (c.log: 2 ranks with timers of 8" in (
            captured.out
        )
        assert "fewer ranks than they ran with" in captured.err
        names = {
            p.name for p in (tmp_path / "scaling_timers_analysis").iterdir()
        }
        assert {
            "p_timer_scaling_sum_all.png",
            "p_timer_scaling_rank_0000.png",
            "p_timer_scaling_rank_0001.png",
            "p_timer_rank_scaling_engine_launch_tasks.png",
            "p_rank_imbalance_summary.txt",
        } <= names

    def test_imbalance_keeps_zeros(self):
        """A rank that never ran a timer counts as zero."""
        log = LogTimers(
            "l",
            "l",
            4,
            3,
            1.0,
            {0: {"x": 10.0}, 1: {}, 2: {}},
            {0: {"x": 1}, 1: {}, 2: {}},
        )
        table = timers.imbalance_table([log], ["x"])
        assert table is not None
        assert "0.0ms" in table and "3.00" in table
        assert timers.imbalance_table([log], ["y"]) is None

    def test_step_count_warning(
        self, synthetic_scaling_db, tmp_path, capsys, fast_plots
    ):
        """Logs covering different numbers of steps warn."""
        logs = [
            str(_write_log(tmp_path / "a.log", 2, [800.0] * 4)),
            str(_write_log(tmp_path / "b.log", 4, [400.0] * 2)),
        ]
        scaling.analyse_swift_scaling(logs, output_path=tmp_path)
        assert "different numbers of steps (2 to 4)" in capsys.readouterr().err

    def test_validation(self, synthetic_scaling_db, tmp_path):
        """Bad inputs are clear errors."""
        logs = _logs(tmp_path)
        with pytest.raises(ValueError, match="at least two"):
            scaling.analyse_swift_scaling(logs[:1])
        with pytest.raises(ValueError, match="START END"):
            scaling.analyse_swift_scaling(logs, step_range=(2, 1))
        with pytest.raises(ValueError, match="must be positive"):
            scaling.analyse_swift_scaling(logs, min_percent_threshold=0)
        with pytest.raises(ValueError, match="rank_source"):
            # Deliberately invalid to check the runtime validation
            scaling.analyse_swift_scaling(
                logs,
                rank_source="x",  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="two different MPI rank"):
            scaling.analyse_swift_scaling([logs[0], logs[0]])
        with pytest.raises(ValueError, match="No timers satisfied"):
            scaling.analyse_swift_scaling(
                logs, output_path=tmp_path, min_percent_threshold=99
            )


class TestPlots:
    """The shared scaling plot."""

    def test_real_render(self, tmp_path):
        """Renders with a log2 rank axis; nothing plottable gives None."""
        path = plots.scaling_plot(
            [2, 4, 8],
            {"Total": [4000.0, 2000.0, 1000.0], "x": [1.0, 0.0, 0.2]},
            tmp_path / "s.png",
            "t",
            "Time (ms)",
            total="Total",
        )
        assert path is not None and path.is_file()
        assert (
            plots.scaling_plot(
                [2, 4], {"x": [1.0, 0.0]}, tmp_path / "n.png", "t", "Time (ms)"
            )
            is None
        )


class TestCLI:
    """Argument parsing."""

    def _parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        scaling.add_scaling_arguments(parser.add_subparsers(dest="analysis"))
        return parser

    def test_timers_arguments(self):
        """The timers subcommand takes the timer-database options too."""
        args = self._parser().parse_args(
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
                "--no-match-revision",
                "--regenerate-db",
            ]
        )
        assert args.log_files == ["a.log", "b.log"]
        assert args.steps == [1, 20] and args.rank_source == "all"
        assert args.match_revision is False and args.regenerate_db is True
        assert args.min_percent_threshold == 5.0
        with pytest.raises(SystemExit):
            self._parser().parse_args(
                ["scaling", "timers", "a", "b", "--min-percent-threshold", "0"]
            )

    def test_runtime_arguments(self):
        """The runtime subcommand takes the shared options."""
        args = self._parser().parse_args(
            ["scaling", "runtime", "a.log", "b.log", "--show"]
        )
        assert args.show is True and args.steps is None

    @pytest.mark.parametrize(
        ("kind", "target", "extra"),
        [
            ("timers", "analyse_swift_scaling", {"rank_source": "root"}),
            ("runtime", "analyse_swift_runtime_scaling", {}),
        ],
    )
    def test_entry_points(self, kind, target, extra, tmp_path):
        """Each subcommand calls its analysis with the parsed options."""
        logs = [str(tmp_path / "a.log"), str(tmp_path / "b.log")]
        for log in logs:
            Path(log).touch()
        args = self._parser().parse_args(["scaling", kind, *logs])
        with patch(f"swiftsim_cli.modes.analyse.scaling.{target}") as run:
            args.func(args)
        kwargs = run.call_args.kwargs
        assert kwargs["log_files"] == logs
        for key, value in extra.items():
            assert kwargs[key] == value
