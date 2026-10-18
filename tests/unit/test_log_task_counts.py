"""Tests for the task counts analysis module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.modes.analyse.log_task_counts import (
    add_task_counts_arguments,
    analyse_swift_task_counts,
    run_swift_task_counts,
    task_count_series,
)


@pytest.mark.usefixtures("in_inputs_dir")
class TestTaskCountsArguments:
    """Tests for task counts CLI argument setup."""

    def test_add_task_counts_arguments(self):
        """Test that task counts arguments are added correctly."""
        from argparse import ArgumentParser

        parent_parser = ArgumentParser()
        subparsers = parent_parser.add_subparsers()

        add_task_counts_arguments(subparsers)

        # Parse test arguments
        args = parent_parser.parse_args(
            ["task-counts", "swift.log", "--prefix", "test"]
        )

        assert args.log_files == [Path("swift.log")]
        assert args.prefix == "test"
        assert args.show is False
        assert args.tasks is None

    def test_add_task_counts_arguments_with_tasks(self):
        """Test task counts arguments with task filter."""
        from argparse import ArgumentParser

        parent_parser = ArgumentParser()
        subparsers = parent_parser.add_subparsers()

        add_task_counts_arguments(subparsers)

        # Parse test arguments with tasks
        args = parent_parser.parse_args(
            ["task-counts", "swift.log", "--tasks", "sort", "self", "pair"]
        )

        assert args.log_files == [Path("swift.log")]
        assert args.tasks == ["sort", "self", "pair"]

    def test_add_task_counts_arguments_multiple_logs(self):
        """Test task counts arguments with multiple log files."""
        from argparse import ArgumentParser

        parent_parser = ArgumentParser()
        subparsers = parent_parser.add_subparsers()

        add_task_counts_arguments(subparsers)

        # Parse test arguments with multiple logs
        args = parent_parser.parse_args(
            ["task-counts", "swift.log", "a.log", "b.log", "--steps", "2", "5"]
        )

        assert args.log_files == [
            Path("swift.log"),
            Path("a.log"),
            Path("b.log"),
        ]
        assert args.steps == [2, 5]


class TestRunSwiftTaskCounts:
    """Tests for the task counts CLI entry point."""

    @patch(
        "swiftsim_cli.modes.analyse.log_task_counts.analyse_swift_task_counts"
    )
    def test_run_swift_task_counts(self, mock_analyse):
        """Test the CLI entry point calls analyse with correct args."""
        args = Mock()
        args.log_files = [Path("/path/to/test.log")]
        args.labels = ["Test Run"]
        args.out = Path("/output")
        args.prefix = "test"
        args.show = True
        args.tasks = ["sort", "self"]
        args.steps = [1, 3]

        run_swift_task_counts(args)

        mock_analyse.assert_called_once_with(
            log_files=[str(args.log_files[0])],
            labels=["Test Run"],
            output_path=args.out,
            prefix="test",
            show_plot=True,
            task_filter=["sort", "self"],
            step_range=[1, 3],
        )

    @patch(
        "swiftsim_cli.modes.analyse.log_task_counts.analyse_swift_task_counts"
    )
    def test_run_swift_task_counts_no_filter(self, mock_analyse):
        """Test the CLI entry point without task filter."""
        args = Mock()
        args.log_files = [Path("/path/to/test.log")]
        args.labels = None
        args.out = None
        args.prefix = None
        args.show = False
        args.tasks = None
        args.steps = None

        run_swift_task_counts(args)

        mock_analyse.assert_called_once_with(
            log_files=[str(args.log_files[0])],
            labels=["test"],
            output_path=None,
            prefix=None,
            show_plot=False,
            task_filter=None,
            step_range=None,
        )

    @patch(
        "swiftsim_cli.modes.analyse.log_task_counts.analyse_swift_task_counts"
    )
    def test_run_swift_task_counts_multiple_logs(self, mock_analyse):
        """Test the CLI entry point with multiple log files."""
        args = Mock()
        args.log_files = [
            Path("/path/to/test1.log"),
            Path("/path/to/test2.log"),
        ]
        args.labels = ["Run 1", "Run 2"]
        args.out = Path("/output")
        args.prefix = "multi"
        args.show = False
        args.tasks = None
        args.steps = None

        run_swift_task_counts(args)

        mock_analyse.assert_called_once_with(
            log_files=[
                str(args.log_files[0]),
                str(args.log_files[1]),
            ],
            labels=["Run 1", "Run 2"],
            output_path=args.out,
            prefix="multi",
            show_plot=False,
            task_filter=None,
            step_range=None,
        )


class TestAnalyseSwiftTaskCounts:
    """Tests for the core task counts analysis function."""

    def test_analyse_swift_task_counts_label_validation(self):
        """Test that labels must match number of log files."""
        import pytest

        with pytest.raises(ValueError, match="one label per input"):
            analyse_swift_task_counts(
                log_files=["log1.txt", "log2.txt"],
                labels=["Label 1"],  # Only one label for two files
                task_filter=None,
            )

    def test_series_from_fixture(self, logs_dir):
        """Task counts are taken from the step each block was printed in."""
        counts = task_count_series(str(logs_dir / "sample.txt"))
        steps, times, totals = counts.steps, counts.sim_times, counts.totals
        # One block per rebuild: init (row 0) and steps 1-4.
        assert steps == [0, 1, 2, 3, 4]
        assert times[0] == pytest.approx(1.225990e-05)
        assert all(len(v) == len(steps) for v in counts.task_series.values())
        assert "skipped" not in counts.task_series
        assert len(counts.skipped) == len(steps)
        assert not counts.partial
        # The block printed before row N belongs to step N (init keeps the
        # last of its blocks).
        assert totals == [297700, 297896, 297876, 296337, 296207]

    def test_missing_tasks_are_zero_filled(self, tmp_path):
        """Per-task series stay aligned with the steps."""
        log = tmp_path / "log.txt"
        log.write_text(
            _block(0, "Total = 5 (per cell = 1.0)", "sort=2 self=3")
            + ROW.format(n=1)
            + _block(0, "Total = 9 (per cell = 1.0)", "sort=4 pair=5")
            + ROW.format(n=2)
        )
        counts = task_count_series(str(log))
        steps, totals, series = counts.steps, counts.totals, counts.task_series
        assert steps == [1, 2]
        assert totals == [5, 9]
        assert series["self"].tolist() == [3, 0]
        assert series["pair"].tolist() == [0, 5]

    def test_ranks_summed_with_last_block_per_rank(self, tmp_path):
        """Blocks are summed over ranks, using each rank's last block."""
        log = tmp_path / "log.txt"
        log.write_text(
            _block(0, "Total = 100 (per cell = 1.0)", "self=100")
            + _block(1, "Total = 7 (per cell = 1.0)", "self=7")
            + _block(0, "Total = 10 (per cell = 1.0)", "self=10")
            + ROW.format(n=1)
        )
        counts = task_count_series(str(log))
        assert counts.totals == [17]
        assert counts.task_series["self"].tolist() == [17]
        assert counts.ranks_seen == [0, 1]
        filtered = task_count_series(str(log), ["self", "pair"]).totals
        assert filtered == [17]

    def test_incomplete_step_skipped(self, tmp_path):
        """Blocks after the last row have no simulation time."""
        log = tmp_path / "log.txt"
        log.write_text(
            _block(0, "Total = 5 (per cell = 1.0)", "self=5")
            + ROW.format(n=1)
            + _block(0, "Total = 6 (per cell = 1.0)", "self=6")
        )
        assert task_count_series(str(log)).steps == [1]

    def test_end_to_end(self, logs_dir, tmp_path):
        """Writes the two plots (no cumulative plot) for several logs."""
        log = str(logs_dir / "sample.txt")
        analyse_swift_task_counts(
            log_files=[log, log],
            labels=["a", "b"],
            output_path=tmp_path,
            task_filter=["self", "pair"],
        )
        out = tmp_path / "task_counts_analysis"
        for name in ["task_counts_per_step.png", "task_counts_by_type.png"]:
            assert (out / name).is_file()
        assert not (out / "task_counts_cumulative.png").exists()

    def test_series_step_range(self, logs_dir):
        """--steps keeps only the blocks of the requested steps."""
        counts = task_count_series(
            str(logs_dir / "sample.txt"), step_range=(1, 2)
        )
        assert counts.steps == [1, 2]

    def test_no_data(self, tmp_path, capsys):
        """A log without task counts is an error and makes no plots."""
        log = tmp_path / "log.txt"
        log.write_text("nothing here\n")
        with pytest.raises(ValueError, match="run SWIFT with -v 1"):
            analyse_swift_task_counts(
                log_files=[str(log)], output_path=tmp_path
            )
        assert "warning: no task counts found" in capsys.readouterr().err
        assert not (tmp_path / "task_counts_analysis").exists() or not any(
            (tmp_path / "task_counts_analysis").iterdir()
        )

    def test_system_total_preferred(self, tmp_path):
        """MPI totals use SWIFT's System total, not the local Total (T5)."""
        p = "[0000] [00001.0] engine_print_task_counts"
        log = tmp_path / "log.txt"
        log.write_text(
            "[0000] [00000.0] main: MPI is up and running with 2 node(s).\n"
            f"{p}: System total: 1000, no. cells: 10\n"
            + _block(0, "Total = 400 (per cell = 1.0)", "self=300 skipped=100")
            + ROW.format(n=1)
        )
        counts = task_count_series(str(log))
        assert counts.totals == [1000]
        assert counts.system_total
        assert counts.partial and counts.n_ranks == 2
        assert counts.skipped.tolist() == [100]

    def test_rank0_only_log_warns(self, logs_dir, tmp_path, capsys):
        """A rank-0-only log of a 4-rank run says so (T5)."""
        counts = task_count_series(str(logs_dir / "mpi4_v1.txt"))
        assert counts.ranks_seen == [0] and counts.n_ranks == 4
        analyse_swift_task_counts(
            [str(logs_dir / "mpi4_v1.txt")], output_path=tmp_path
        )
        err = capsys.readouterr().err
        assert "rank(s) 0 only, of 4 ranks" in err

    def test_unknown_task_name(self, logs_dir, tmp_path):
        """Unknown --tasks names list the valid ones (T17)."""
        with pytest.raises(ValueError, match="unknown task name.*bogus"):
            analyse_swift_task_counts(
                [str(logs_dir / "sample.txt")],
                output_path=tmp_path,
                task_filter=["self", "bogus"],
            )

    def test_summary_labels(self, logs_dir, tmp_path, capsys):
        """The summary says what the totals count (T13, T17)."""
        log = str(logs_dir / "sample.txt")
        analyse_swift_task_counts([log], output_path=tmp_path)
        out = capsys.readouterr().out
        assert "ALL (incl. skipped)" in out
        assert "skipped (any type)" in out
        analyse_swift_task_counts(
            [log], output_path=tmp_path, task_filter=["self"]
        )
        assert "SELECTED (self)" in capsys.readouterr().out


ROW = (
    "{n:8d}   1.0e-05    0.0078125  127.0   0.0e+00    1   56            0"
    "     100            0            0            0             1.000"
    "    265           495.008\n"
)


def _block(rank: int, total: str, counts: str) -> str:
    p = f"[{rank:04d}] [00001.0] engine_print_task_counts"
    return f"{p}: {total}\n{p}: task counts are [ {counts} ]\n"
