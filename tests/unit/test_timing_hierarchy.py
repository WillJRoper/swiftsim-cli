"""Tests for the call-path aggregation and its tables."""

import copy

import pytest

from swiftsim_cli.timing.log_scan import scan_log
from swiftsim_cli.timing.timer_db import TimerDB, build_timer_db
from swiftsim_cli.timing.timing_hierarchy import (
    UNTIMED,
    aggregate,
    between_steps_table,
    build_step_trees,
    coverage_outliers,
    coverage_table,
    flat_table,
    function_table,
    hierarchy_for,
    rank_imbalance_table,
    samples_by_key,
    select_steps,
    top_level_table,
    tree_table,
    write_coverage_csv,
)

ROW = (
    "{n:8d}   1.0e-05    0.0078125  127.0   0.0e+00    1   56            0"
    "     100            0            0            0             {wall}"
    "    265           495.008\n"
)


@pytest.fixture(scope="module")
def sample(synthetic_db, logs_dir):
    """Scan, step trees and hierarchy of the sample log."""
    scan = scan_log(logs_dir / "sample.txt", synthetic_db)
    steps = select_steps(scan.steps, None)
    trees = build_step_trees(steps, 0, synthetic_db)
    total = sum(s.wallclock_ms or 0.0 for s in steps)
    return scan, trees, aggregate(trees, 0, total)


class TestCoverage:
    """The step trees account for the step wall-clock time."""

    def test_per_step_coverage(self, sample):
        """Every step is >= 99% timed and not double counted."""
        _, trees, _ = sample
        assert [t.step.number for t in trees] == [1, 2, 3, 4]
        for tree in trees:
            assert 99.0 <= tree.coverage <= 100.1

    def test_step1_ground_truth(self, sample):
        """Step 1 = engine_prepare + engine_launch + collect (+ tiny)."""
        _, trees, _ = sample
        roots = {r.record.key: r.inclusive for r in trees[0].roots}
        assert roots["engine_prepare"] == pytest.approx(1514.4, abs=0.05)
        assert roots["engine_launch (tasks)"] == pytest.approx(
            1670.1, abs=0.05
        )
        assert trees[0].timed == pytest.approx(3196.752, rel=1e-4)

    def test_top_level_sums_to_100(self, sample):
        """Top-level timers plus untimed is exactly the step time."""
        _, _, h = sample
        top = sum(h.paths[p].inclusive for p in h.children(()))
        assert top + h.untimed == pytest.approx(h.total)
        assert h.untimed >= 0
        table = top_level_table(h, 50, "T")
        assert UNTIMED in table and "100.0%" in table

    def test_exclusive_sums_to_timed(self, sample):
        """Exclusive times add up to the timed time, never more."""
        _, _, h = sample
        exclusive = sum(s.exclusive for s in h.paths.values())
        assert exclusive == pytest.approx(h.timed)
        assert exclusive <= h.total
        assert sum(h.percent(s.exclusive) for _, s, _ in h.flat()) <= 100.0

    def test_no_negative_exclusive(self, sample):
        """No timer is smaller than its children on this fixture."""
        assert sample[2].negative == 0


class TestAggregation:
    """Call-path statistics."""

    def test_paths(self, sample):
        """A function under several paths gets one entry per path."""
        _, _, h = sample
        rebuild = ("engine_prepare", "engine_rebuild")
        assert h.paths[rebuild].calls == 4
        assert h.paths[rebuild + ("engine_unskip",)].steps == {1, 2, 3, 4}
        assert h.paths[("engine_launch (tasks)",)].exclusive == pytest.approx(
            h.paths[("engine_launch (tasks)",)].inclusive
        )

    def test_flat_inclusive_counted_once(self, sample):
        """The flat table merges paths; exclusive first."""
        _, _, h = sample
        rows = {key: (s, outer) for key, s, outer in h.flat()}
        stats, outer = rows["space_rebuild"]
        assert outer == pytest.approx(stats.inclusive)
        assert stats.exclusive < stats.inclusive
        excl = [s.exclusive for _, s, _ in h.flat()]
        assert excl == sorted(excl, reverse=True)

    def test_merged_subtree_and_callers(self, synthetic_db, sample):
        """--hierarchy-functions merges subtrees over all callers."""
        scan, trees, _ = sample
        init_trees = build_step_trees([scan.init_step], 0, synthetic_db)
        h = aggregate(init_trees + trees, 0, 1.0)
        subtree, callers = h.merged_subtree("engine_rebuild")
        assert {c: s.calls for c, s in callers.items()} == {
            "engine_init_particles": 2,
            "engine_prepare": 4,
        }
        assert subtree[("engine_rebuild",)].calls == 6
        assert subtree[("engine_rebuild", "space_rebuild")].calls == 6
        text = function_table(h, "engine_rebuild", 5)
        assert "CALLERS OF engine_rebuild" in text
        assert "engine_init_particles" in text

    def test_names_and_keys(self, sample):
        """Timers can be named by function or by key."""
        _, _, h = sample
        assert h.known("engine_launch")
        assert h.known("engine_launch (tasks)")
        assert not h.known("no_such_function")
        # An operation is only matched by its key
        assert not h.known("compute_potential_global")

    def test_top_n_limits_rows(self, sample):
        """--top-n limits tables and children per node."""
        _, _, h = sample
        # title + header + 3 rows
        assert flat_table(h, 3, "F").count("\n| ") == 5
        assert "more)" in tree_table(h, 2, "T")
        assert "other timers" in top_level_table(h, 2, "T")

    def test_samples(self, sample):
        """Per-call samples cover every node once."""
        _, trees, h = sample
        samples = samples_by_key(trees)
        assert len(samples["engine_rebuild"]) == 4
        assert sum(len(v) for v in samples.values()) == sum(
            s.calls for s in h.paths.values()
        )


class TestEdgeCases:
    """Zero times, step ranges and ranks."""

    def test_zero_times(self, synthetic_db, logs_dir):
        """All-zero logs give empty statistics, not ZeroDivisionError."""
        scan = scan_log(logs_dir / "zero_time.txt", synthetic_db)
        h = hierarchy_for(scan.steps, 0, synthetic_db)
        assert h.total == 0 and h.n_steps == 0
        assert h.percent(5.0) == 0.0
        top_level_table(h, 5, "T")
        tree_table(h, 5, "T")
        flat_table(h, 5, "F")
        coverage_table([], 5)

    def test_zero_time_timers_in_a_step(self, synthetic_db, tmp_path):
        """0.000 ms timers in a timed step are harmless."""
        log = tmp_path / "log.txt"
        log.write_text(
            "[00001.0] engine_launch: (tasks) took 0.000 ms.\n"
            "       1   1.0e-05    0.0078125  127.0   0.0e+00    1   56"
            "            0     100            0            0            0"
            "             1.000    265           495.008\n"
        )
        scan = scan_log(log, synthetic_db)
        h = hierarchy_for(scan.steps, 0, synthetic_db)
        assert h.untimed == pytest.approx(1.0)
        assert "0.0%" in flat_table(h, 5, "F")

    def test_step_range(self, sample):
        """--steps selects complete steps in the range."""
        scan, _, _ = sample
        assert [s.number for s in select_steps(scan.steps, (2, 3))] == [2, 3]
        assert select_steps(scan.steps, (10, 20)) == []

    def test_restart_row_excluded(self, synthetic_db, logs_dir):
        """The 0.000 restart row is not part of the step time."""
        scan = scan_log(logs_dir / "fos_restart.txt", synthetic_db)
        steps = select_steps(scan.steps, None)
        assert 47241 not in [s.number for s in steps]

    def test_ranks_are_separate(self, synthetic_db, logs_dir):
        """Each rank is aggregated alone; imbalance compares them."""
        scan = scan_log(logs_dir / "mpi_v2_two_rank.txt", synthetic_db)
        h0 = hierarchy_for(scan.steps, 0, synthetic_db)
        h1 = hierarchy_for(scan.steps, 1, synthetic_db)
        assert h0.total == h1.total  # step rows come from rank 0
        assert h1.timed == pytest.approx(1.25 * h0.timed, rel=1e-3)
        table = rank_imbalance_table({0: h0, 1: h1}, 5)
        assert "1.11" in table  # 1.25 / mean(1, 1.25)
        assert "RANK IMBALANCE (2 ranks)" in table

    def test_coverage_table(self, sample, tmp_path):
        """Only outliers are tabled; the CSV lists every step."""
        _, trees, _ = sample
        table = coverage_table(trees, 5)
        assert "PER-STEP COVERAGE OUTLIERS (rank 0: 0 of 4 steps" in table
        csv = write_coverage_csv(trees, tmp_path / "c.csv")
        lines = csv.read_text().splitlines()
        assert lines[0].startswith("Step,Wall-clock (ms)")
        assert len(lines) == 5 and "3196.752" in lines[1]

    def test_coverage_outliers(self, sample):
        """Steps outside 90-105% are outliers, worst first."""
        _, trees, _ = sample
        trees = copy.deepcopy(trees)
        trees[0].step.wallclock_ms = 10 * trees[0].timed
        trees[1].step.wallclock_ms = 0.5 * trees[1].timed
        assert set(map(id, coverage_outliers(trees))) == {
            id(trees[0]),
            id(trees[1]),
        }
        table = coverage_table(trees, 1)
        assert "2 of 4 steps" in table
        # Only the worst (top_n=1): 200% is further from 100% than 10%.
        assert " 200.0 " in table and " 10.0 " not in table


class TestBetweenSteps:
    """Timers main() prints between steps are not part of any step."""

    def _scan(self, synthetic_db, tmp_path, dump_from_repartition=False):
        inner = (
            "[00001.4] task_dump_stats: took 50.000 ms.\n"
            "[00001.4] engine_repartition: took 60.000 ms.\n"
            if dump_from_repartition
            else ""
        )
        log = tmp_path / "log.txt"
        log.write_text(
            ROW.format(n=0, wall="100.000")
            + inner
            + "[00001.5] engine_launch: (tasks) took 900.000 ms.\n"
            + "[00001.6] task_dump_all: took 300.000 ms.\n"
            + "[00001.7] task_dump_stats: took 200.000 ms.\n"
            + ROW.format(n=1, wall="1000.000")
        )
        return scan_log(log, synthetic_db)

    def test_task_dumps_between_steps(self, synthetic_db, tmp_path):
        """task_dump_* after a step leave the step tree and its coverage."""
        scan = self._scan(synthetic_db, tmp_path)
        (tree,) = build_step_trees(
            select_steps(scan.steps, None), 0, synthetic_db
        )
        assert [r.record.key for r in tree.roots] == ["engine_launch (tasks)"]
        assert [r.key for r in tree.between] == [
            "task_dump_all",
            "task_dump_stats",
        ]
        assert tree.coverage == pytest.approx(90.0)
        h = aggregate([tree], 0, 1000.0)
        assert h.between["task_dump_all"].inclusive == 300.0
        assert "task_dump_all" not in {p[-1] for p in h.paths}
        table = between_steps_table(h, 5)
        assert "BETWEEN STEPS" in table and "300.0" in table

    def test_dump_inside_repartition_stays(self, synthetic_db, tmp_path):
        """task_dump_stats under engine_repartition belongs to the step."""
        scan = self._scan(synthetic_db, tmp_path, dump_from_repartition=True)
        (tree,) = build_step_trees(
            select_steps(scan.steps, None), 0, synthetic_db
        )
        rep = next(
            r for r in tree.roots if r.record.key == "engine_repartition"
        )
        assert [c.record.key for c in rep.children] == ["task_dump_stats"]
        assert [r.key for r in tree.between] == [
            "task_dump_all",
            "task_dump_stats",
        ]

    def test_db_rules(self, synthetic_db):
        """in_step / outside_step_root follow the static call graph."""
        db = synthetic_db
        assert db.in_step("engine_prepare")
        assert db.in_step("task_dump_stats")  # via engine_repartition
        assert not db.in_step("task_dump_all")
        assert db.in_step("not_in_source")
        assert db.outside_step_root("task_dump_stats")
        assert not db.outside_step_root("engine_launch")
        assert not db.outside_step_root("engine_step")

    def test_no_engine_step(self, tmp_path):
        """Without engine_step in the source nothing is between steps."""
        (tmp_path / "a.c").write_text(
            "void main(void) { f(); }\n"
            'void f(void) { message("took %f ms"); }\n'
        )
        db = TimerDB(build_timer_db(tmp_path))
        assert db.in_step("f") and not db.outside_step_root("f")
