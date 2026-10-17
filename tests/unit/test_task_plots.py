"""Tests for the combined task/threadpool plotting mode."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

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
        threadpool_file = tmp_path / "threadpool-step1.dat"
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
            output_dir
            / "demo_task_plots_analysis"
            / "demo_task_plots_rank0.png"
        ).exists()
        assert (
            output_dir
            / "demo_task_plots_analysis"
            / "demo_task_plots_rank1.png"
        ).exists()
        assert (
            output_dir
            / "demo_task_plots_analysis"
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
            output_dir
            / "step3_task_plots_analysis"
            / "step3_task_plots_rank0.png"
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
            / "demo_task_plots_analysis"
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
            / "demo_task_plots_analysis"
            / "demo_task_plots_rank0_ci1.png"
        ).exists()
        assert (
            output_dir
            / "demo_task_plots_analysis"
            / "demo_task_activity_rank0_ci1.png"
        ).exists()
