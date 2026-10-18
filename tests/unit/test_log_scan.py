"""Tests for the single-pass log scanner."""

import pytest

from swiftsim_cli.timing.log_scan import (
    parse_revision,
    read_log_revision,
    scan_log,
)

ROW = (
    "{n:8d}   1.0e-05    0.0078125  127.0   0.0e+00    1   56            0"
    "     100            0            0            0             {wall}"
    "    265           495.008\n"
)


def _step(scan, number):
    return next(s for s in scan.steps if s.number == number)


def _keys(step, rank=0):
    return [r.key for r in step.records[rank]]


@pytest.fixture(scope="module")
def sample_scan(synthetic_db, logs_dir):
    """The sample log scanned with the synthetic database."""
    return scan_log(logs_dir / "sample.txt", synthetic_db)


class TestStepAssignment:
    """Records before row N belong to step N."""

    def test_rows_and_kinds(self, sample_scan):
        """Row 0 closes initialisation; a trailing step is incomplete."""
        kinds = [(s.number, s.kind) for s in sample_scan.steps]
        assert kinds == [
            (0, "init"),
            (1, "step"),
            (2, "step"),
            (3, "step"),
            (4, "step"),
            (5, "incomplete"),
        ]
        assert _step(sample_scan, 5).wallclock_ms is None
        assert not sample_scan.restart

    def test_init_contains_engine_init_particles(self, sample_scan):
        """Row 0's wall-clock is engine_init_particles, in the init block."""
        init = sample_scan.init_step
        assert init.wallclock_ms == pytest.approx(15118.246)
        (rec,) = [
            r for r in init.records[0] if r.key == "engine_init_particles"
        ]
        assert rec.ms == pytest.approx(15118.246)

    def test_off_by_one(self, sample_scan):
        """The timers printed after row 0 belong to step 1 (row 3196.752)."""
        step1 = _step(sample_scan, 1)
        assert step1.wallclock_ms == pytest.approx(3196.752)
        keys = _keys(step1)
        assert keys[0] == "engine_step: Writing step info to files"
        assert keys[-2:] == [
            "engine_collect_end_of_step",
            "zoom_void_timestep_collect",
        ]
        by_key = {r.key: r.ms for r in step1.records[0]}
        top = (
            by_key["engine_prepare"]
            + by_key["engine_launch (tasks)"]
            + by_key["engine_collect_end_of_step"]
        )
        assert by_key["engine_prepare"] == pytest.approx(1514.4, abs=0.05)
        assert by_key["engine_launch (tasks)"] == pytest.approx(
            1670.1, abs=0.05
        )
        assert by_key["engine_collect_end_of_step"] == pytest.approx(
            12.1, abs=0.05
        )
        assert top == pytest.approx(3196.752, rel=1e-3)

    def test_restart(self, synthetic_db, logs_dir):
        """A first row with 0.000 wall-clock marks a restart."""
        scan = scan_log(logs_dir / "fos_restart.txt", synthetic_db)
        init = scan.init_step
        assert init is not None
        assert scan.restart
        assert init.number == 47241 and init.wallclock_ms == 0.0
        assert "restart_read" in _keys(init)
        assert scan.steps[-1].kind == "incomplete"
        assert [s.number for s in scan.complete_steps] == [
            47242,
            47243,
            47244,
            47245,
        ]

    def test_log_starting_mid_run(self, synthetic_db, logs_dir):
        """A log cut from the middle of a run has no init block."""
        scan = scan_log(logs_dir / "fos_fof.txt", synthetic_db)
        assert scan.init_step is None
        assert [s.number for s in scan.complete_steps] == [47572, 47573]

    def test_init_and_last_elapsed(self, sample_scan):
        """Elapsed times split the run into init and the rest."""
        assert sample_scan.init_end_s == pytest.approx(23.5)
        assert sample_scan.last_elapsed_s >= sample_scan.init_end_s


class TestMatching:
    """Timer lines are matched to timers or kept as unknown."""

    def test_all_sample_lines_matched(self, sample_scan):
        """Every timer line of the fixture is known to the synthetic DB."""
        assert sample_scan.matched == 302
        assert not sample_scan.unmatched

    def test_unknown_records_kept(self, synthetic_db, tmp_path):
        """Unknown timers become unknown:<func>:<label> records."""
        log = tmp_path / "log.txt"
        log.write_text(
            "[00001.0] mystery: Doing 12 things took 1.5 ms.\n"
            "[00001.1] mystery: took 2.0 ms.\n"
            "[00001.1] mystery: nothing took place here\n"
        )
        scan = scan_log(log, synthetic_db)
        records = scan.steps[0].records[0]
        assert [(r.key, r.kind, r.ms) for r in records] == [
            ("unknown:mystery: Doing # things", "operation", 1.5),
            ("unknown:mystery", "function", 2.0),
        ]
        assert scan.unmatched == {"mystery: Doing # things": 1, "mystery": 1}
        assert scan.steps[0].kind == "incomplete"

    def test_runtime_label_and_unit(self, synthetic_db, tmp_path):
        """`(%s) took` keys include the label; seconds become ms."""
        log = tmp_path / "log.txt"
        log.write_text(
            "[00001.0] engine_launch: (tasks) took 2.000 s.\n"
            "[00001.0] engine_launch: (fof) took 3.000 ms.\n"
        )
        records = scan_log(log, synthetic_db).steps[0].records[0]
        assert [(r.key, r.ms) for r in records] == [
            ("engine_launch (tasks)", 2000.0),
            ("engine_launch (fof)", 3.0),
        ]

    def test_without_db_timers_are_skipped(self, logs_dir):
        """Scanning without a DB still finds steps and task counts."""
        scan = scan_log(logs_dir / "sample.txt")
        assert scan.matched == 0 and not scan.timer_ranks
        assert len(scan.complete_steps) == 4

    def test_zero_times(self, synthetic_db, logs_dir):
        """0.000 ms timers and wall-clock times scan fine."""
        scan = scan_log(logs_dir / "zero_time.txt", synthetic_db)
        assert scan.matched == 4
        assert all(r.ms == 0.0 for s in scan.steps for r in s.records[0])


class TestMPI:
    """Rank prefixes and per-rank blocks."""

    def test_v1_only_rank0_timers(self, synthetic_db, logs_dir):
        """At -v 1 only rank 0 prints timers; rows are unprefixed."""
        scan = scan_log(logs_dir / "mpi4_v1.txt", synthetic_db)
        assert scan.n_ranks == 4
        assert scan.timer_ranks == {0}
        assert scan.revision == "6cdef369"
        assert [s.number for s in scan.complete_steps] == [1, 2, 3]

    def test_v2_two_ranks(self, synthetic_db, logs_dir):
        """At -v 2 each rank's records are kept apart."""
        scan = scan_log(logs_dir / "mpi_v2_two_rank.txt", synthetic_db)
        step = _step(scan, 1)
        assert scan.timer_ranks == {0, 1}
        assert _keys(step, 0) == _keys(step, 1)
        r0 = sum(r.ms for r in step.records[0])
        r1 = sum(r.ms for r in step.records[1])
        assert r1 == pytest.approx(1.25 * r0, rel=1e-3)

    def test_interleaved_task_count_blocks(self, tmp_path):
        """Interleaved blocks from two ranks are not mixed up."""
        log = tmp_path / "log.txt"
        p = "engine_print_task_counts"
        log.write_text(
            f"[0000] [00001.0] {p}: System total: 10, no. cells: 2\n"
            f"[0001] [00001.0] {p}: System total: 20, no. cells: 4\n"
            f"[0001] [00001.0] {p}: Total = 20 (per cell = 5.0)\n"
            f"[0000] [00001.0] {p}: Total = 10 (per cell = 5.0)\n"
            f"[0001] [00001.0] {p}: task counts are [ self=7 pair=13 ]\n"
            f"[0000] [00001.0] {p}: task counts are [ self=3 pair=7 ]\n"
            + ROW.format(n=1, wall="1.000")
        )
        (step,) = scan_log(log).steps
        blocks = {b.rank: b for b in step.task_counts}
        assert blocks[0].counts == {"self": 3, "pair": 7}
        assert blocks[0].system_total == 10 and blocks[0].total_tasks == 10
        assert blocks[1].counts == {"self": 7, "pair": 13}
        assert blocks[1].num_cells == 4 and blocks[1].per_cell_avg == 5.0
        assert step.sim_time == pytest.approx(1.0e-05)

    def test_task_categories(self, sample_scan):
        """CPU time per task category is summed per step and rank."""
        cats = _step(sample_scan, 1).task_categories[0]
        assert cats["gravity"] == pytest.approx(237866.12)
        assert "total" not in cats


class TestRevision:
    """The Revision: header line."""

    @pytest.mark.parametrize(
        "line, expected",
        [
            (
                " Revision: v2025.04-235-g9ee518c2, Branch: x",
                ("9ee518c2", False),
            ),
            (" Revision: d5386777-dirty, Branch: y", ("d5386777", True)),
            (
                " Revision: v0.9.0-3290-g6cdef369, Branch: z",
                ("6cdef369", False),
            ),
            (" Revision: unknown, Branch: z", (None, False)),
            ("no revision here", (None, False)),
        ],
    )
    def test_parse_revision(self, line, expected):
        """Hashes are taken from git-describe style revisions."""
        assert parse_revision(line) == expected

    def test_read_log_revision(self, logs_dir, tmp_path):
        """The revision is read from the log header."""
        assert read_log_revision(logs_dir / "sample.txt") == "9ee518c2"
        empty = tmp_path / "empty.log"
        empty.write_text("nothing\n")
        assert read_log_revision(empty) is None
