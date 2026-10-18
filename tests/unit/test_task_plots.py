"""Tests for the combined task/threadpool plotting mode."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pytest

from swiftsim_cli.modes.analyse import task_plots
from swiftsim_cli.modes.analyse.task_debug_data import (
    TaskDebugData,
    TaskFilter,
)
from swiftsim_cli.modes.analyse.task_plots import (
    PlotInterval,
    _add_legend,
    add_task_plots_arguments,
    analyse_swift_task_plots,
)


def _write_non_mpi_task_file(path):
    rows = [
        [
            0,
            0,
            0,
            0,
            1000,
            1400,
            0,
            0,
            0,
            0,
            1000,
            1,
            2,
            0,
            1,
            0,
            1,
            0,
            0,
        ],
        [0, 3, 1, 0, 1100, 1300, 0, 0, 0, 0, 1000, 1, 2, 0, 1, 0, 1, 0, 0],
        [1, 2, 4, 0, 1450, 1700, 0, 0, 0, 0, 1000, 2, 2, 0, 0, 1, 1, 0, 0],
    ]
    path.write_text("\n".join(" ".join(str(v) for v in row) for row in rows))


def _write_mpi_task_file(path):
    rows = [
        [
            0,
            0,
            0,
            0,
            0,
            1000,
            1500,
            0,
            0,
            0,
            0,
            0,
            1000,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            2,
            3,
            0,
            1100,
            1300,
            0,
            0,
            0,
            0,
            0,
            1000,
            1,
            2,
            0,
            1,
            0,
            1,
            0,
            0,
        ],
        [
            1,
            0,
            0,
            0,
            0,
            1050,
            1600,
            0,
            0,
            0,
            0,
            0,
            1000,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            1,
            0,
            3,
            2,
            0,
            1150,
            1450,
            0,
            0,
            0,
            0,
            0,
            1000,
            2,
            3,
            1,
            2,
            1,
            2,
            0,
            0,
        ],
    ]
    path.write_text("\n".join(" ".join(str(v) for v in row) for row in rows))


def _write_threadpool_file(path):
    path.write_text(
        "\n".join(
            [
                "# threadpool dump",
                "# {'num_threads': 2, 'cpufreq': 1000000}",
                "drift_mapper 0 0 1000 1250",
                "hydro_mapper 1 0 1125 1500",
                "occupancy -1 0 1000 1500",
            ]
        )
    )


class TestTaskPlotsMode:
    """Tests for the task-plots analysis mode."""

    def test_add_arguments(self):
        """The task-plots subparser accepts the expected switches."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="analysis_type", required=True)
        add_task_plots_arguments(subparsers)

        args = parser.parse_args(
            [
                "task-plots",
                "tasks.dat",
                "threadpool.dat",
                "--rank",
                "1",
                "--activity-plot",
                "--sort-threads",
                "--c-type",
                "zoom",
                "--ci-type",
                "zoom",
                "--cj-type",
                "background",
                "--c-subtype",
                "void",
            ]
        )

        assert args.analysis_type == "task-plots"
        assert args.rank == 1
        assert args.activity_plot is True
        assert args.sort_threads is True
        assert args.c_type == 1
        assert args.ci_type == 1
        assert args.cj_type == 2
        assert args.c_subtype == 2

    def test_task_debug_data_filters_tasks(self, tmp_path):
        """Position-specific task filters select exact ci and cj values."""
        task_file = tmp_path / "thread_info-step1.dat"
        _write_non_mpi_task_file(task_file)

        data = TaskDebugData(task_file)
        records = data.get_rank_records(
            0,
            TaskFilter(task="pair/density", ci_type=1, cj_type=2),
        )

        assert len(records) == 1
        assert records[0].label == "pair/density"
        assert data.thread_counts[0] == 2

    def test_task_debug_data_filters_either_cell(self, tmp_path):
        """Generic cell filters match either ci or cj metadata."""
        task_file = tmp_path / "thread_info-step1.dat"
        _write_non_mpi_task_file(task_file)
        data = TaskDebugData(task_file)

        assert len(data.get_rank_records(0, TaskFilter(c_type=1))) == 1
        assert len(data.get_rank_records(0, TaskFilter(ci_type=1))) == 1
        assert len(data.get_rank_records(0, TaskFilter(cj_type=1))) == 0
        assert len(data.get_rank_records(0, TaskFilter(c_type=2))) == 2
        assert len(data.get_rank_records(0, TaskFilter(c_subtype=1))) == 1
        assert len(data.get_rank_records(0, TaskFilter(ci_subtype=1))) == 0
        assert len(data.get_rank_records(0, TaskFilter(cj_subtype=1))) == 1

    def test_analyse_swift_task_plots_emits_one_file_per_rank(self, tmp_path):
        """MPI task files emit one combined plot per rank by default."""
        task_file = tmp_path / "thread_info_MPI-step1.dat"
        threadpool_file = tmp_path / "threadpool_info-rank0-step1.dat"
        output_dir = tmp_path / "outputs"
        _write_mpi_task_file(task_file)
        _write_threadpool_file(threadpool_file)

        analyse_swift_task_plots(
            task_file=task_file,
            threadpool_file=threadpool_file,
            output_path=output_dir,
            prefix="demo",
            show_plot=False,
        )

        assert (
            output_dir / "task_plots_analysis" / "demo_task_plots_rank0.png"
        ).exists()
        assert (
            output_dir / "task_plots_analysis" / "demo_task_plots_rank1.png"
        ).exists()
        assert (
            output_dir
            / "task_plots_analysis"
            / "demo_task_timing_tables_rank0.txt"
        ).exists()

    def test_analyse_swift_task_plots_infers_step_prefix(self, tmp_path):
        """Input step numbers become output prefixes when none is supplied."""
        task_file = tmp_path / "thread_info-step3.dat"
        threadpool_file = tmp_path / "threadpool_info-step3.dat"
        output_dir = tmp_path / "outputs"
        _write_non_mpi_task_file(task_file)
        _write_threadpool_file(threadpool_file)

        analyse_swift_task_plots(
            task_file=task_file,
            threadpool_file=threadpool_file,
            output_path=output_dir,
            show_plot=False,
        )

        assert (
            output_dir / "task_plots_analysis" / "step3_task_plots_rank0.png"
        ).exists()

    def test_timing_report_includes_task_and_cell_breakdowns(self, tmp_path):
        """Timing report includes integrated and per-task cell combinations."""
        task_file = tmp_path / "thread_info-step1.dat"
        threadpool_file = tmp_path / "threadpool-step1.dat"
        output_dir = tmp_path / "outputs"
        _write_non_mpi_task_file(task_file)
        _write_threadpool_file(threadpool_file)

        analyse_swift_task_plots(
            task_file=task_file,
            threadpool_file=threadpool_file,
            output_path=output_dir,
            prefix="demo",
            show_plot=False,
        )

        report = (
            output_dir
            / "task_plots_analysis"
            / "demo_task_timing_tables_rank0.txt"
        ).read_text()
        assert "TASK TYPE TIMING" in report
        assert "CELL TYPE TIMING BY TASK TYPE" in report
        assert "CELL TYPE/SUBTYPE TIMING (ALL TASK TYPES)" in report
        assert "CELL TYPE/SUBTYPE TIMING BY TASK TYPE" in report
        assert "pair/density" in report
        assert "Background/Neighbour + Zoom/Regular" in report

    def test_legends_are_outside_top_and_bottom_panels(self):
        """Panel legends sit above the top and below the bottom axes."""
        _, (top_axis, bottom_axis) = plt.subplots(2, 1)
        intervals = [PlotInterval(0, 0.0, 1.0, "blue", "task")]

        _add_legend(top_axis, intervals, ncol=1, above=True)
        _add_legend(bottom_axis, intervals, ncol=1, above=False)

        assert top_axis.get_legend().get_bbox_to_anchor()._bbox.y0 > 1
        assert bottom_axis.get_legend().get_bbox_to_anchor()._bbox.y0 < 0
        plt.close(top_axis.figure)

    def test_analyse_swift_task_plots_can_emit_activity_plot(self, tmp_path):
        """Activity plot generation writes an additional per-rank figure."""
        task_file = tmp_path / "thread_info-step1.dat"
        threadpool_file = tmp_path / "threadpool-step1.dat"
        output_dir = tmp_path / "outputs"
        _write_non_mpi_task_file(task_file)
        _write_threadpool_file(threadpool_file)

        analyse_swift_task_plots(
            task_file=task_file,
            threadpool_file=threadpool_file,
            output_path=output_dir,
            prefix="demo",
            activity_plot=True,
            task_filter=TaskFilter(ci_type=1),
            show_plot=False,
        )

        assert (
            output_dir
            / "task_plots_analysis"
            / "demo_task_plots_rank0_ci1.png"
        ).exists()
        assert (
            output_dir
            / "task_plots_analysis"
            / "demo_task_activity_rank0_ci1.png"
        ).exists()


def _mpi_row(rank, task, subtask, tic, toc, cpufreq, thread=0):
    """Build one 21-column MPI task-debug row."""
    return (
        [rank, thread, task, subtask, 0, tic, toc]
        + [0] * 5
        + [
            cpufreq,
            1,
            2,
            0,
            1,
            0,
            1,
            0,
            0,
        ]
    )


def _write_rows(path, rows):
    path.write_text("\n".join(" ".join(str(v) for v in row) for row in rows))


class TestTaskDebugDataParsing:
    """Regression tests for task-debug parsing."""

    def test_each_mpi_rank_header_is_dropped(self, tmp_path):
        """No rank gets a fake task-0 record spanning its step."""
        task_file = tmp_path / "thread_info_MPI-step1.dat"
        _write_mpi_task_file(task_file)

        data = TaskDebugData(task_file)

        assert data.ranks == [0, 1]
        assert all(record.task_type != "none" for record in data.records)
        assert len(data.get_rank_records(1)) == 1
        assert data.rank_bounds_ms(1) == pytest.approx((1150.0, 1450.0))
        report = task_plots._task_timing_report(data.get_rank_records(1), 1)
        assert "Total task worker time: 300.000 ms" in report

    def test_cpu_frequency_is_read_per_rank(self, tmp_path):
        """Each rank converts ticks with its own header's CPU frequency."""
        task_file = tmp_path / "thread_info_MPI-step1.dat"
        _write_rows(
            task_file,
            [
                _mpi_row(0, 0, 0, 1000, 2000, 1000000),
                _mpi_row(0, 2, 3, 1000, 2000, 1000000),
                _mpi_row(1, 0, 0, 1000, 2000, 2000000),
                _mpi_row(1, 2, 3, 1000, 2000, 2000000),
            ],
        )

        data = TaskDebugData(task_file)

        assert data.rank_bounds_ms(0) == pytest.approx((1.0, 2.0))
        assert data.rank_bounds_ms(1) == pytest.approx((0.5, 1.0))
        assert data.mintic_to_ms(2000, rank=1) == pytest.approx(1.0)

    def test_unknown_task_indices_fall_back(self, tmp_path):
        """Task and subtype IDs newer than our tables don't crash."""
        task_file = tmp_path / "thread_info_MPI-step1.dat"
        _write_rows(
            task_file,
            [
                _mpi_row(0, 0, 0, 1000, 2000, 1000000),
                _mpi_row(0, 999, 0, 1000, 1500, 1000000),
                _mpi_row(0, 3, 888, 1500, 2000, 1000000),
            ],
        )

        labels = sorted(r.label for r in TaskDebugData(task_file).records)

        assert labels == ["pair/unknown_888", "unknown_999"]

    def test_unexpected_column_count_raises(self, tmp_path):
        """Layouts other than SWIFT's 19/21 columns are rejected."""
        task_file = tmp_path / "thread_info-step1.dat"
        _write_rows(task_file, [[1] * 13, [1] * 13])

        with pytest.raises(ValueError, match="13 columns"):
            TaskDebugData(task_file)


class TestThreadpoolRankMatching:
    """A threadpool dump is only drawn on the rank that wrote it."""

    def _run(self, tmp_path, threadpool_name, rank=None):
        task_file = tmp_path / "thread_info_MPI-step1.dat"
        threadpool_file = tmp_path / threadpool_name
        _write_mpi_task_file(task_file)
        _write_threadpool_file(threadpool_file)
        analyse_swift_task_plots(
            task_file=task_file,
            threadpool_file=threadpool_file,
            output_path=tmp_path / "out",
            prefix="demo",
            rank=rank,
            show_plot=False,
        )

    def test_overlay_only_on_rank_in_filename(self, tmp_path, monkeypatch):
        """Rank 1's dump appears on rank 1 only."""
        calls = []
        original = task_plots._plot_combined_timeline

        def spy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(task_plots, "_plot_combined_timeline", spy)
        self._run(tmp_path, "threadpool_info-rank1-step1.dat")

        assert [len(c["threadpool_records"]) for c in calls] == [0, 2]

    def test_multi_rank_without_rank_requires_flag(self, tmp_path):
        """A rank-less dump with a multi-rank task file needs --rank."""
        with pytest.raises(ValueError, match="--rank"):
            self._run(tmp_path, "threadpool-step1.dat")

        self._run(tmp_path, "threadpool-step1.dat", rank=1)
        assert (
            tmp_path
            / "out"
            / "task_plots_analysis"
            / "demo_task_plots_rank1.png"
        ).exists()

    def test_mismatched_rank_flag_raises(self, tmp_path):
        """--rank must agree with the rank in the threadpool filename."""
        with pytest.raises(ValueError, match="rank 0"):
            self._run(tmp_path, "threadpool_info-rank0-step1.dat", rank=1)


def test_importing_task_plots_does_not_force_backend():
    """Importing the module must leave the user's backend alone."""
    env = {**os.environ, "MPLBACKEND": "pdf"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import matplotlib; "
            "import swiftsim_cli.modes.analyse.task_plots; "
            "print(matplotlib.get_backend())",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout.strip() == "pdf"
