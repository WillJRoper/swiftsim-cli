"""Tests for task-plots: dump layouts, names, threadpool parsing, plots."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from swiftsim_cli.modes.analyse import task_names, task_plots
from swiftsim_cli.modes.analyse.task_debug_data import (
    TaskDebugData,
    TaskFilter,
    detect_layout,
)
from swiftsim_cli.modes.analyse.task_names import (
    builtin_names,
    parse_source_names,
    resolve_task_names,
)
from swiftsim_cli.modes.analyse.task_plot_metadata import (
    TASK_TYPES,
    UPSTREAM_SUBTYPES,
    ZOOM_SUBTYPES,
    colour_map,
    distinct_colours,
)
from swiftsim_cli.modes.analyse.task_plots import (
    _lanes,
    _raster_lanes,
    _task_filter_suffix,
    add_task_plots_arguments,
    analyse_swift_task_plots,
)
from swiftsim_cli.modes.analyse.threadpool_data import ThreadpoolData

DATA = Path(__file__).parent.parent / "data" / "tasks"
EAGLE = DATA / "thread_info-step1002.dat"  # upstream, non-MPI, 11 columns
EAGLE_POOL = DATA / "threadpool_info-step1002.dat"
FLARES = DATA / "thread_info_MPI-step802.dat"  # 13-col header, 17-col rows
ZOOM = DATA / "thread_info-step2.dat"  # zoom, non-MPI, 19 columns
SRC = DATA / "swift_src"

UPSTREAM = builtin_names(False)
ZOOM_NAMES = builtin_names(True)


def _write_rows(path, rows):
    path.write_text("\n".join(" ".join(str(v) for v in row) for row in rows))
    return path


def _mpi_row(rank, task, subtask, tic, toc, cpufreq, thread=0, cells=None):
    """One 21-column (zoom MPI) row; task 0 makes it a header."""
    cells = cells if cells is not None else [1, 2, 0, 1, 0, 1]
    return (
        [rank, thread, task, subtask, 0, tic, toc]
        + [0] * 5
        + [cpufreq if task == 0 else 0]
        + cells
        + [0.0, 0.0]
    )


def _write_mpi_file(path):
    """Two ranks, two tasks on rank 0 and one on rank 1."""
    return _write_rows(
        path,
        [
            _mpi_row(0, 0, 0, 1000, 1500, 1000000),
            _mpi_row(0, 2, 3, 1100, 1300, 1000000),
            _mpi_row(0, 3, 1, 1200, 1400, 1000000, thread=1),
            _mpi_row(1, 0, 0, 1000, 1800, 1000000),
            _mpi_row(1, 3, 2, 1150, 1450, 1000000, cells=[2, 3, 1, 2, 1, 2]),
        ],
    )


def _write_threadpool(path, tic=1000, toc=1500):
    path.write_text(
        "# map_function thread_id chunk_size tic toc\n"
        "# {'num_threads': 2, 'cpufreq': 1000000}\n"
        f"drift_mapper 0 1 {tic} {tic + 200}\n"
        f"(null) 3 1 {tic + 100} {toc}\n"
        f"drift_mapper -1 10 {tic} {toc}\n"
    )
    return path


# =============================================================================
# Dump layouts
# =============================================================================


class TestLayouts:
    """Every task_dump_all layout is detected and parsed."""

    def test_upstream_non_mpi_11_columns(self):
        """EAGLE dump: no rank column, no cell columns."""
        data = TaskDebugData(EAGLE, names=UPSTREAM)
        assert (data.layout.is_mpi, data.layout.ncols) == (False, 11)
        assert not data.layout.has_cells
        assert data.ranks == [0]
        assert len(data.tasks) == 39
        assert (data.tasks["ci_type"] == -1).all()
        assert data.cpu_clock_khz_by_rank[0] == pytest.approx(2595119.055)
        assert data.step_duration_ms(0) > 0
        assert (data.tasks["start_ms"] >= 0).all()

    def test_ragged_mpi_13_column_header_17_column_rows(self):
        """FLARES dump: the short header is read separately."""
        data = TaskDebugData(FLARES, names=UPSTREAM)
        assert data.layout.is_mpi
        assert (data.layout.header_cols, data.layout.ncols) == (13, 17)
        assert not data.layout.has_cells
        assert data.file_ranks == [0, 1]
        assert len(data.tasks) == 48

    def test_rank_filter_happens_before_building_rows(self):
        """--rank keeps only that rank's rows."""
        data = TaskDebugData(FLARES, ranks=[1], names=UPSTREAM)
        assert data.ranks == [1]
        assert set(data.tasks["rank"]) == {1}
        assert len(data.tasks) == 24
        with pytest.raises(ValueError, match="rank 7 not present"):
            TaskDebugData(FLARES, ranks=[7], names=UPSTREAM)

    def test_upstream_mpi_13_columns(self, tmp_path):
        """Upstream MPI rows (13 columns) parse without cell data."""
        rows = [line.split()[:13] for line in FLARES.read_text().splitlines()]
        data = TaskDebugData(
            _write_rows(tmp_path / "t.dat", rows), names=UPSTREAM
        )
        assert data.layout.ncols == 13 and data.layout.is_mpi
        assert len(data.tasks) == 48

    def test_zoom_non_mpi_19_columns_with_legacy_cell_type(self):
        """Cell type 3 selects the legacy zoom cell table (T7)."""
        data = TaskDebugData(ZOOM, names=builtin_names(True, 3))
        assert data.layout.has_cells and data.layout.ncols == 19
        assert data.names.cell_types[3] == "Background"
        mask = data.filter_mask(TaskFilter(c_type="bkg"))
        assert (
            mask.sum()
            == (
                (data.tasks["ci_type"] == 3) | (data.tasks["cj_type"] == 3)
            ).sum()
        )
        assert mask.sum() > 0
        assert data.filter_mask(TaskFilter(c_type=3)).sum() == mask.sum()

    def test_zoom_mpi_21_columns(self, tmp_path):
        """Per-rank headers are dropped and cell filters apply."""
        data = TaskDebugData(
            _write_mpi_file(tmp_path / "t.dat"), names=ZOOM_NAMES
        )
        assert data.layout.ncols == 21 and data.layout.has_cells
        assert data.ranks == [0, 1]
        assert (data.tasks["type"] > 0).all()
        assert data.thread_counts == {0: 2, 1: 1}
        rank1 = data.rank_tasks(1)
        assert rank1["start_ms"].tolist() == pytest.approx([0.15])
        assert data.step_duration_ms(1) == pytest.approx(0.8)
        assert data.filter_mask(TaskFilter(ci_type="zoom")).sum() == 2
        assert data.filter_mask(TaskFilter(cj_subtype="void")).sum() == 1

    def test_ticks_keep_int64_precision(self, tmp_path):
        """Ticks above 2**53 are not rounded through float64."""
        base = 2**60
        data = TaskDebugData(
            _write_rows(
                tmp_path / "t.dat",
                [
                    _mpi_row(0, 0, 0, base, base + 10_000, 1000),
                    _mpi_row(0, 2, 3, base + 1, base + 3, 1000),
                ],
            ),
            names=ZOOM_NAMES,
        )
        start, end = data.tasks[["start_ms", "end_ms"]].iloc[0]
        assert (start, end) == (1.0, 3.0)  # 1 kHz: one tick per ms

    def test_per_rank_cpu_frequency(self, tmp_path):
        """Each rank's ticks use its own header's frequency."""
        data = TaskDebugData(
            _write_rows(
                tmp_path / "t.dat",
                [
                    _mpi_row(0, 0, 0, 0, 2000, 1000000),
                    _mpi_row(0, 2, 3, 1000, 2000, 0),
                    _mpi_row(1, 0, 0, 0, 2000, 2000000),
                    _mpi_row(1, 2, 3, 1000, 2000, 0),
                ],
            ),
            names=ZOOM_NAMES,
        )
        assert data.rank_tasks(0)["end_ms"].iloc[0] == pytest.approx(2.0)
        assert data.rank_tasks(1)["end_ms"].iloc[0] == pytest.approx(1.0)

    def test_unknown_layout_raises(self, tmp_path):
        """Other column counts are rejected with the expected layouts."""
        path = _write_rows(tmp_path / "t.dat", [[1] * 15, [1] * 15])
        with pytest.raises(ValueError, match="11/19 non-MPI"):
            detect_layout(path)

    def test_out_of_range_indices_warn(self, tmp_path, capsys):
        """Indices beyond the tables warn and show as unknown_N."""
        data = TaskDebugData(
            _write_rows(
                tmp_path / "t.dat",
                [
                    _mpi_row(0, 0, 0, 1000, 2000, 1000000),
                    _mpi_row(0, 999, 0, 1000, 1500, 0),
                    _mpi_row(0, 3, 888, 1500, 2000, 0),
                ],
            ),
            names=ZOOM_NAMES,
        )
        assert sorted(map(str, data.tasks["label"])) == [
            "pair/unknown_888",
            "unknown_999",
        ]
        err = capsys.readouterr().err
        assert "task type indices 999" in err
        assert "task subtype indices 888" in err

    def test_cell_filter_without_cells_raises(self):
        """Cell filters on a dump without cell columns are an error."""
        data = TaskDebugData(EAGLE, names=UPSTREAM)
        with pytest.raises(ValueError, match="no cell metadata"):
            data.filter_mask(TaskFilter(c_type="zoom"))

    def test_unknown_names_list_valid_ones(self, tmp_path):
        """Bad task labels and cell names name the valid choices (T17)."""
        data = TaskDebugData(
            _write_mpi_file(tmp_path / "t.dat"), names=ZOOM_NAMES
        )
        with pytest.raises(ValueError, match="types: sort, self, pair"):
            data.filter_mask(TaskFilter(tasks=("pair/forse",)))
        with pytest.raises(ValueError, match="regular, zoom, background"):
            data.filter_mask(TaskFilter(c_type="buffer"))


# =============================================================================
# Names from source
# =============================================================================


class TestTaskNames:
    """Names come from the SWIFT source, with built-in fallbacks."""

    def test_builtin_tables_by_layout(self):
        """Upstream has no progeny subtype; zoom inserts it at 6."""
        assert UPSTREAM.subtypes == UPSTREAM_SUBTYPES
        assert ZOOM_NAMES.subtypes == ZOOM_SUBTYPES
        assert ZOOM_SUBTYPES[6] == "progeny"
        assert UPSTREAM_SUBTYPES[6] == "fof"
        assert "rt_sorts" in TASK_TYPES

    def test_parse_synthetic_source(self):
        """Drift test: arrays and enums are read from the C source."""
        names = parse_source_names(
            (SRC / "src" / "task.c").read_bytes(),
            (SRC / "src" / "cell.h").read_bytes(),
        )
        assert names["task_types"][-1] == "my_new_task"
        assert names["subtypes"][5:] == ["progeny", "my_new_subtype"]
        assert names["cell_types"] == [
            "Regular",
            "Zoom",
            "Buffer",
            "Background",
        ]
        assert names["cell_subtypes"] == ["Regular", "Neighbour", "Void"]

    def test_parse_without_cell_enums(self):
        """Upstream cell.h has no cell types: only task tables."""
        names = parse_source_names((SRC / "src" / "task.c").read_bytes(), None)
        assert set(names) == {"task_types", "subtypes"}

    def test_missing_arrays_raise(self):
        """A task.c without the arrays is an error."""
        with pytest.raises(ValueError, match="taskID_names"):
            parse_source_names(b"int x = 1;", None)

    def test_resolve_from_swift_src_and_cache(self, monkeypatch):
        """--swift-src is parsed once, then served from the cache."""
        names = resolve_task_names(SRC, has_cells=True, max_cell_type=3)
        assert names.task_types[-1] == "my_new_task"
        assert names.cell_types[3] == "Background"
        assert "names from" in names.source

        def boom(*_):
            raise AssertionError("cache not used")

        monkeypatch.setattr(task_names, "parse_source_names", boom)
        again = resolve_task_names(SRC, has_cells=True)
        assert again.task_types == names.task_types

    def test_bad_swift_src_raises(self, tmp_path):
        """An explicit --swift-src without src/task.c is an error."""
        with pytest.raises(ValueError, match="--swift-src"):
            resolve_task_names(tmp_path, has_cells=False)

    def test_fallback_warns(self, capsys):
        """Without a profile checkout the built-in tables are used."""
        names = resolve_task_names(None, has_cells=False)
        assert names == UPSTREAM
        assert "built-in upstream tables" in capsys.readouterr().err

    def test_short_source_cell_table_falls_back_to_legacy(
        self, tmp_path, capsys
    ):
        """A dump with cell type 3 and a 3-type source uses legacy names."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "task.c").write_bytes((SRC / "src" / "task.c").read_bytes())
        (src / "cell.h").write_text(
            "enum cell_types { cell_type_regular, cell_type_zoom, "
            "cell_type_bkg };"
        )
        names = resolve_task_names(tmp_path, has_cells=True, max_cell_type=3)
        assert names.cell_types[2:] == ("Buffer", "Background")
        assert "older zoom dump" in capsys.readouterr().err

    @pytest.mark.skipif(
        "SWIFT_SRC" not in os.environ, reason="set SWIFT_SRC to a checkout"
    )
    def test_builtin_tables_match_real_checkout(self):
        """Smoke test: the built-in tables match a real SWIFT tree."""
        root = Path(os.environ["SWIFT_SRC"])
        cell_h = root / "src" / "cell.h"
        names = parse_source_names(
            (root / "src" / "task.c").read_bytes(), cell_h.read_bytes()
        )
        assert tuple(names["task_types"]) == TASK_TYPES
        expected = (
            ZOOM_SUBTYPES if "cell_types" in names else UPSTREAM_SUBTYPES
        )
        assert tuple(names["subtypes"]) == expected
        master = subprocess.run(
            ["git", "-C", str(root), "show", "origin/master:src/task.c"],
            capture_output=True,
            check=False,
        )
        if master.returncode == 0:
            upstream = parse_source_names(master.stdout, None)
            assert tuple(upstream["task_types"]) == TASK_TYPES
            assert tuple(upstream["subtypes"]) == UPSTREAM_SUBTYPES


# =============================================================================
# Threadpool
# =============================================================================


class TestThreadpool:
    """Threadpool dumps: main thread (-1), (null) names, errors."""

    def test_parse(self, tmp_path):
        """Worker ids become contiguous; -1 stays the main thread."""
        pool = ThreadpoolData(_write_threadpool(tmp_path / "tp.dat"))
        assert pool.cpu_clock_khz == 1000.0
        assert pool.thread_count == 2
        assert pool.rows["thread"].tolist() == [0, 1, -1]
        assert pool.rows["function"].tolist() == ["drift", "(null)", "drift"]
        assert pool.tick_range() == (1000, 1500)
        assert pool.to_ms(1000)["end_ms"].tolist() == [0.2, 0.5, 0.5]

    def test_real_dump(self):
        """The EAGLE excerpt parses, including main-thread rows."""
        pool = ThreadpoolData(EAGLE_POOL)
        assert (pool.rows["thread"] == -1).sum() == 2
        assert pool.thread_count == 16

    def test_errors(self, tmp_path):
        """A missing header or no rows is a clear error naming the file."""
        bad = tmp_path / "bad.dat"
        bad.write_text("drift 0 1 2 3\n")
        with pytest.raises(ValueError, match="bad.dat: missing or malformed"):
            ThreadpoolData(bad)
        empty = tmp_path / "empty.dat"
        empty.write_text("# a\n# {'num_threads': 1, 'cpufreq': 1000}\n")
        with pytest.raises(ValueError, match="no data"):
            ThreadpoolData(empty)


# =============================================================================
# Plots and the analysis entry point
# =============================================================================


def _spy(calls, original):
    """Record each call's keyword arguments, then call ``original``."""

    def wrapper(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    return wrapper


def _run(tmp_path, task_file, names=ZOOM_NAMES, **kwargs):
    analyse_swift_task_plots(
        task_file=task_file,
        output_path=tmp_path / "out",
        prefix="demo",
        names=names,
        **kwargs,
    )
    return tmp_path / "out" / "task_plots_analysis"


class TestAnalysis:
    """End-to-end behaviour of analyse_swift_task_plots."""

    def test_real_dump_with_threadpool(self, tmp_path):
        """The EAGLE excerpt and its threadpool dump plot together."""
        out = tmp_path / "out" / "task_plots_analysis"
        analyse_swift_task_plots(
            task_file=EAGLE,
            threadpool_file=EAGLE_POOL,
            output_path=tmp_path / "out",
            names=UPSTREAM,
            activity_plot=True,
        )
        assert (out / "step1002_task_plots_rank0.png").exists()
        assert (out / "step1002_task_activity_rank0.png").exists()
        report = (out / "step1002_task_timing_tables_rank0.txt").read_text()
        assert "TASK TYPE TIMING" in report
        assert "CELL TYPE" not in report

    def test_threadpool_is_optional(self, tmp_path, monkeypatch):
        """Without --threadpool only the task panel is drawn (T3, T10)."""
        figures = []
        monkeypatch.setattr(
            task_plots,
            "save_figure",
            lambda fig, path, show=False: figures.append(fig),
        )
        _run(tmp_path, FLARES, names=UPSTREAM)
        assert len(figures) == 2
        assert all(len(fig.axes) == 1 for fig in figures)

    def test_shared_window_from_headers(self, tmp_path, monkeypatch):
        """All ranks share the longest header step as the x-range (T9)."""
        calls: list[dict] = []
        original = task_plots._plot_timeline
        monkeypatch.setattr(
            task_plots,
            "_plot_timeline",
            _spy(calls, original),
        )
        _run(tmp_path, _write_mpi_file(tmp_path / "thread_info_MPI-step1.dat"))
        assert [c["window_ms"] for c in calls] == pytest.approx([0.8, 0.8])
        assert [c["start_ms"] for c in calls] == [0.0, 0.0]

    def test_rank_without_matches_is_skipped(self, tmp_path, capsys):
        """One rank failing the filter warns; the others still plot (T8)."""
        task_file = _write_mpi_file(tmp_path / "thread_info_MPI-step1.dat")
        out = _run(tmp_path, task_file, task_filter=TaskFilter(cj_type=3))
        assert "rank 0: no tasks matched" in capsys.readouterr().err
        assert not list(out.glob("demo_task_plots_rank0*"))
        assert (out / "demo_task_plots_rank1_cjunknown_3.png").exists()
        with pytest.raises(ValueError, match="no task records matched"):
            _run(tmp_path, task_file, task_filter=TaskFilter(depth=9))

    def test_filter_suffix_uses_names(self):
        """Suffixes name cell types and subtypes consistently (T24)."""
        suffix = _task_filter_suffix(
            TaskFilter(
                tasks=("pair/force", "sort"),
                ci_type=1,
                cj_type="bkg",
                c_subtype=2,
                depth=3,
            ),
            ZOOM_NAMES,
        )
        assert suffix == "_pair-force+sort_cizoom_cjbackground_csubvoid_depth3"

    def test_threadpool_must_overlap_step(self, tmp_path):
        """A dump from another run/step is refused (T11)."""
        task_file = _write_mpi_file(tmp_path / "thread_info_MPI-step1.dat")
        pool = _write_threadpool(
            tmp_path / "threadpool_info-rank0-step1.dat",
            tic=10**9,
            toc=2 * 10**9,
        )
        with pytest.raises(ValueError, match="does not overlap"):
            _run(tmp_path, task_file, threadpool_file=pool)

    def test_threadpool_only_on_its_rank(self, tmp_path, monkeypatch):
        """Rank 1's dump appears on rank 1 only."""
        calls: list[dict] = []
        original = task_plots._plot_timeline
        monkeypatch.setattr(
            task_plots,
            "_plot_timeline",
            _spy(calls, original),
        )
        task_file = _write_mpi_file(tmp_path / "thread_info_MPI-step1.dat")
        pool = _write_threadpool(tmp_path / "threadpool_info-rank1-step1.dat")
        _run(tmp_path, task_file, threadpool_file=pool)
        assert [c["pool_rows"] is None for c in calls] == [True, False]

    def test_rankless_threadpool_needs_rank(self, tmp_path):
        """A rank-less dump with a multi-rank task file needs --rank."""
        task_file = _write_mpi_file(tmp_path / "thread_info_MPI-step1.dat")
        pool = _write_threadpool(tmp_path / "threadpool-step1.dat")
        with pytest.raises(ValueError, match="--rank"):
            _run(tmp_path, task_file, threadpool_file=pool)
        out = _run(tmp_path, task_file, threadpool_file=pool, rank=1)
        assert (out / "demo_task_plots_rank1.png").exists()
        with pytest.raises(ValueError, match="rank 0"):
            _run(
                tmp_path,
                task_file,
                threadpool_file=_write_threadpool(
                    tmp_path / "threadpool_info-rank0-step1.dat"
                ),
                rank=1,
            )


class TestHelpers:
    """Lanes, raster, colours and arguments."""

    def test_lanes_clip_and_expand(self):
        """Intervals are clipped and spread round-robin over lanes."""
        lane, s, e, index = _lanes(
            np.array([0, 0, 0, 1]),
            np.array([0.0, 2.0, 4.0, -1.0]),
            np.array([1.0, 3.0, 20.0, 0.5]),
            2,
            0.0,
            10.0,
        )
        assert lane.tolist() == [0, 1, 0, 2]
        assert e.tolist() == [1.0, 3.0, 10.0, 0.5]
        assert s.tolist() == [0.0, 2.0, 4.0, 0.0]
        assert index.tolist() == [0, 1, 2, 3]

    def test_raster_paints_every_interval(self):
        """Even sub-pixel intervals cover at least one column."""
        palette = np.array([[1.0, 0, 0, 1], [0, 0, 1.0, 1]])
        image = _raster_lanes(
            np.array([0, 1]),
            np.array([0.0, 5.0]),
            np.array([0.0001, 10.0]),
            np.array([0, 1]),
            palette,
            2,
            10.0,
            100,
            rows_per_lane=2,
        )
        assert image.shape == (4, 100, 4)
        assert (image[0] == 0).all()  # lane separator row
        assert image[1, 0].tolist() == [255, 0, 0, 255]
        assert (image[3, 50:] == [0, 0, 255, 255]).all()
        assert (image[3, :50] == 0).all()

    def test_colours_distinct_and_stable(self):
        """Up to 60 labels get distinct colours in the order given."""
        assert len(set(distinct_colours(60))) == 60
        assert len(set(distinct_colours(90))) == 90
        labels = [f"l{i}" for i in range(40)]
        assert colour_map(labels) == colour_map(labels)
        assert len(set(colour_map(labels).values())) == 40

    def test_arguments(self, in_inputs_dir):
        """--threadpool is optional; cell values take names or indices."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="analysis_type", required=True)
        add_task_plots_arguments(sub)
        args = parser.parse_args(
            [
                "task-plots",
                "tasks.dat",
                "--threadpool",
                "threadpool.dat",
                "--rank",
                "1",
                "--c-type",
                "Zoom",
                "--cj-type",
                "3",
                "--tasks",
                "pair/force",
            ]
        )
        assert args.threadpool == Path("threadpool.dat")
        assert (args.c_type, args.cj_type) == ("zoom", 3)
        assert (
            parser.parse_args(["task-plots", "tasks.dat"]).threadpool is None
        )
        with pytest.raises(SystemExit):
            parser.parse_args(["task-plots", "tasks.dat", "threadpool.dat"])
        with pytest.raises(SystemExit):
            parser.parse_args(["task-plots", "tasks.dat", "--c-type", "-1"])


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
