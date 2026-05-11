"""Tests for the combined task/threadpool plotting mode."""

from __future__ import annotations

import argparse

from swiftsim_cli.modes.analyse.task_debug_data import (
    TaskDebugData,
    TaskFilter,
)
from swiftsim_cli.modes.analyse.task_plots import (
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
            ]
        )

        assert args.analysis_type == "task-plots"
        assert args.rank == 1
        assert args.activity_plot is True
        assert args.sort_threads is True

    def test_task_debug_data_filters_tasks(self, tmp_path):
        """Task filtering keeps the zoom-tool symmetric behavior."""
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
            output_dir / "task_plots_analysis" / "demo_task_plots_rank0.png"
        ).exists()
        assert (
            output_dir / "task_plots_analysis" / "demo_task_plots_rank1.png"
        ).exists()

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
