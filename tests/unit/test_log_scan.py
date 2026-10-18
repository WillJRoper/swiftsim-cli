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
            "[00001.2] mystery: mystery() took (FOF SCALING): 4.0 ms.\n"
        )
        scan = scan_log(log, synthetic_db)
        records = scan.steps[0].records[0]
        assert [(r.key, r.kind, r.ms) for r in records] == [
            ("unknown:mystery: Doing # things", "operation", 1.5),
            ("unknown:mystery", "function", 2.0),
            ("unknown:mystery", "function", 4.0),
        ]
        assert scan.unmatched == {"mystery: Doing # things": 1, "mystery": 2}
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
        assert read_log_revision(logs_dir / "sample.txt") == (
            "9ee518c2",
            False,
        )
        empty = tmp_path / "empty.log"
        empty.write_text("nothing\n")
        assert read_log_revision(empty) == (None, False)


WARNING = "hydro_part_has_no_neighbours: WARNING: Gas particle treated badly."


class TestDamagedLines:
    """Lines with other output appended (from real interleaved logs)."""

    def _scan(self, synthetic_db, tmp_path, text):
        log = tmp_path / "log.txt"
        log.write_text(ROW.format(n=0, wall="1.000") + text)
        return scan_log(log, synthetic_db)

    def test_time_read_up_to_the_cut(self, synthetic_db, tmp_path):
        """A cut after a complete time keeps the timer (ms assumed)."""
        scan = self._scan(
            synthetic_db,
            tmp_path,
            f"[00002.0] engine_launch: (tasks) took 280.495 m[00002.1] "
            f"{WARNING}\n"
            f"[00002.0] engine_prepare: took 10.089 ms (including unskip, "
            f"rebuild and rewei[00002.1] {WARNING}\n"
            f"[00002.0] engine_collect_end_of_step: took 7.867[00002.1] "
            f"{WARNING}\n" + ROW.format(n=1, wall="300.000"),
        )
        step = _step(scan, 1)
        assert {r.key: r.ms for r in step.records[0]} == {
            "engine_launch (tasks)": pytest.approx(280.495),
            "engine_prepare": pytest.approx(10.089),
            "engine_collect_end_of_step": pytest.approx(7.867),
        }
        assert scan.cut_lines == 3 and scan.recovered == 3
        assert not scan.unmatched and scan.lost == 0

    def test_truncated_time_joined_with_its_rest(self, synthetic_db, tmp_path):
        """An incomplete time is joined with the rest on a later line."""
        scan = self._scan(
            synthetic_db,
            tmp_path,
            f"[00002.0] engine_launch: (tasks) to[00002.1] {WARNING}\n"
            f"[00002.1] {WARNING}\n"
            "ok 3439.710 ms.\n"
            f"[00002.0] engine_unskip: took 3[00002.1] {WARNING}\n"
            "1.250 ms.\n" + ROW.format(n=1, wall="3500.000"),
        )
        keys = {r.key: r.ms for r in _step(scan, 1).records[0]}
        assert keys["engine_launch (tasks)"] == pytest.approx(3439.71)
        assert keys["engine_unskip"] == pytest.approx(31.25)
        assert scan.recovered == 2 and scan.lost == 0

    def test_prefix_cut_mid_name(self, synthetic_db, tmp_path):
        """'[el] engine_lau' + later 'nch: ...' is one line again."""
        scan = self._scan(
            synthetic_db,
            tmp_path,
            f"[00002.0] engine_lau[00002.1] {WARNING}\n"
            "nch: (tasks) took 1931.072 ms.\n"
            f"[00003.0[00003.1] {WARNING}\n"
            "] engine_launch: (tasks) took 100.000 ms.\n"
            + ROW.format(n=1, wall="2100.000"),
        )
        recs = _step(scan, 1).records[0]
        assert [r.ms for r in recs] == [1931.072, 100.0]
        assert recs[0].elapsed == 2.0 and recs[1].elapsed == 3.0

    def test_cut_step_row_rejoined(self, synthetic_db, tmp_path):
        """A step row cut by a warning is rebuilt from its displaced rest."""
        row = ROW.format(n=1, wall="146.776").rstrip("\n")
        head, tail = row[: row.index("265")], row[row.index("265") :]
        scan = self._scan(
            synthetic_db,
            tmp_path,
            "[00002.0] engine_launch: (tasks) took 140.000 ms.\n"
            f"{head}[00002.1] {WARNING}\n{tail}\n",
        )
        assert _step(scan, 1).wallclock_ms == pytest.approx(146.776)
        assert scan.recovered_rows == 1 and not scan.step_gaps

    def test_unreadable_fragment_dropped(self, synthetic_db, tmp_path):
        """What cannot be rebuilt is counted, not made an unknown timer."""
        scan = self._scan(
            synthetic_db,
            tmp_path,
            f"[00002.0] engine_unskip: took 3[00002.1] {WARNING}\n"
            f"[00002.0] engine_prepare: took 1[00002.1] {WARNING}\n"
            + ROW.format(n=1, wall="1.000"),
        )
        assert scan.lost == 1  # the first head was replaced
        assert not scan.unmatched and not _step(scan, 1).records[0]

    def test_missing_row_marks_a_gap(self, synthetic_db, tmp_path):
        """A step closed after a lost row is a gap, not a clean step."""
        log = tmp_path / "log.txt"
        log.write_text(
            ROW.format(n=0, wall="1.000")
            + ROW.format(n=1, wall="2.000")
            + ROW.format(n=3, wall="3.000")
            + ROW.format(n=4, wall="4.000")
        )
        scan = scan_log(log, synthetic_db)
        assert [(s.number, s.kind) for s in scan.steps] == [
            (0, "init"),
            (1, "step"),
            (3, "gap"),
            (4, "step"),
        ]
        assert scan.step_gaps == [(1, 3)]
        assert [s.number for s in scan.complete_steps] == [1, 3, 4]


class TestLateRankLines:
    """-v 2: other ranks' lines reaching the log after rank 0's row."""

    def test_assigned_by_elapsed_time(self, synthetic_db, tmp_path):
        """A late rank-1 line goes to the step its elapsed time is in."""
        log = tmp_path / "log.txt"
        log.write_text(
            ROW.format(n=0, wall="1.000")
            + "[0000] [00010.0] engine_launch: (tasks) took 900.000 ms.\n"
            + ROW.format(n=1, wall="1000.000")
            # rank 1's step-1 timers, printed after rank 0's row 1
            + "[0001] [00009.9] engine_launch: (tasks) took 800.000 ms.\n"
            + "[0001] [00010.0] engine_collect_end_of_step: took 5.0 ms.\n"
            # step 2 (rank 0 by position, rank 1 by time)
            + "[0000] [00011.0] engine_launch: (tasks) took 950.000 ms.\n"
            + "[0001] [00010.9] engine_launch: (tasks) took 850.000 ms.\n"
            + ROW.format(n=2, wall="1000.000")
        )
        scan = scan_log(log, synthetic_db)
        s1, s2 = _step(scan, 1), _step(scan, 2)
        assert [r.ms for r in s1.records[1]] == [800.0, 5.0]
        assert [r.ms for r in s2.records[1]] == [850.0]
        assert [r.ms for r in s1.records[0]] == [900.0]
        assert [r.ms for r in s2.records[0]] == [950.0]
