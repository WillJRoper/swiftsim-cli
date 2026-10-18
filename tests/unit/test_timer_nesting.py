"""Tests for the runtime timer-tree reconstruction."""

from collections import defaultdict

import pytest

from swiftsim_cli.timing.log_scan import Record, scan_log
from swiftsim_cli.timing.timer_db import TimerDB
from swiftsim_cli.timing.timer_nesting import build_tree, can_contain


def _parents(scan, db, rank=0):
    """(key, parent key) -> [calls, ms] over every block of the log."""
    out = defaultdict(lambda: [0, 0.0])

    def walk(node, parent):
        entry = out[(node.record.key, parent)]
        entry[0] += 1
        entry[1] += node.record.ms
        for child in node.children:
            walk(child, node.record.key)

    for step in scan.steps:
        for root in build_tree(step.records.get(rank, []), db):
            walk(root, f"<{step.kind}>")
    return out


def _callers(parents, key):
    return {p: tuple(v) for (k, p), v in parents.items() if k == key}


@pytest.fixture(scope="module")
def sample_parents(synthetic_db, logs_dir):
    """Caller attribution over the sample log."""
    return _parents(
        scan_log(logs_dir / "sample.txt", synthetic_db), synthetic_db
    )


@pytest.fixture(scope="module")
def fos_parents(synthetic_db, logs_dir):
    """Caller attribution over the restarted FOS log."""
    scan = scan_log(logs_dir / "fos_restart.txt", synthetic_db)
    return _parents(scan, synthetic_db)


class TestMultipleCallers:
    """Functions with several callers are split by actual caller."""

    def test_engine_rebuild_init_vs_prepare(self, sample_parents):
        """engine_rebuild runs in engine_init_particles and engine_prepare."""
        callers = _callers(sample_parents, "engine_rebuild")
        assert set(callers) == {"engine_init_particles", "engine_prepare"}
        assert callers["engine_init_particles"][0] == 2
        assert callers["engine_init_particles"][1] == pytest.approx(
            3595.9, abs=0.1
        )
        # 4 steps plus the incomplete trailing step
        assert callers["engine_prepare"][0] == 5

    def test_engine_unskip_rebuild_vs_prepare(self, fos_parents):
        """engine_unskip is under engine_rebuild only on rebuild steps."""
        callers = _callers(fos_parents, "engine_unskip")
        assert set(callers) == {"engine_rebuild", "engine_prepare"}
        # The rebuild step 47242 (log line 183) ...
        assert callers["engine_rebuild"] == (1, pytest.approx(0.579))
        # ... and the plain steps 47243-47245 plus the trailing step.
        assert callers["engine_prepare"] == (
            4,
            pytest.approx(0.879 + 0.528 + 1.276 + 0.662),
        )

    def test_scheduler_reweight_in_maketasks(self, sample_parents):
        """scheduler_reweight is attributed to engine_maketasks only."""
        assert set(_callers(sample_parents, "scheduler_reweight")) == {
            "engine_maketasks"
        }

    def test_no_double_counting(self, sample_parents):
        """Each record appears once: space_rebuild only under rebuild."""
        assert set(_callers(sample_parents, "space_rebuild")) == {
            "engine_rebuild"
        }

    def test_untimed_intermediate(self, sample_parents):
        """Timers below an untimed caller sit at the step top level."""
        callers = _callers(
            sample_parents, "compute_potential_global: Gpart assignment"
        )
        assert callers["<step>"][0] == 2

    def test_operation_containing_a_call(self, sample_parents):
        """An operation claims the timed function called inside it."""
        callers = _callers(
            sample_parents,
            "zoom_engine_make_self_gravity_tasks: "
            "Making bkg->bkg gravity tasks",
        )
        assert set(callers) == {"engine_maketasks: Making gravity tasks"}


def _db():
    """F calls g; f has one operation 'op' spanning lines 2-5."""
    return TimerDB(
        {
            "timers": [
                dict(
                    function="f",
                    kind="function",
                    pattern="^took",
                    label="",
                    sites=["a.c:9"],
                    ranges=[["a.c", 1, 9]],
                ),
                dict(
                    function="f",
                    kind="operation",
                    pattern="^op",
                    label="op",
                    sites=["a.c:5"],
                    ranges=[["a.c", 2, 5]],
                ),
                dict(
                    function="g",
                    kind="function",
                    pattern="^took",
                    label="",
                    sites=["a.c:20"],
                    ranges=[["a.c", 15, 20]],
                ),
                dict(
                    function="h",
                    kind="function",
                    pattern="^took",
                    label="",
                    sites=["a.c:30"],
                    ranges=[["a.c", 25, 30]],
                ),
            ],
            "calls": {"f": ["g"]},
            "call_sites": {"f": [["a.c", 3, "g"], ["a.c", 7, "h"]]},
        }
    )


def _rec(db, func, kind, ms, elapsed, line=0):
    timer = next(t for t in db.by_function[func] if t.kind == kind)
    key = func if kind == "function" else f"{func}: {timer.label}"
    return Record(0, elapsed, ms, func, kind, key, timer, line)


class TestClaimingRules:
    """The three claiming conditions."""

    def test_static_rules(self):
        """Functions contain their ops and reachable functions."""
        db = _db()
        f = _rec(db, "f", "function", 10, 1.0)
        op = _rec(db, "f", "operation", 5, 1.0)
        g = _rec(db, "g", "function", 1, 1.0)
        h = _rec(db, "h", "function", 1, 1.0)
        assert can_contain(f, op, db)
        assert can_contain(f, g, db)
        assert not can_contain(f, h, db)
        assert not can_contain(g, f, db)
        # The op contains g (called on line 3) but not h (line 7).
        assert can_contain(op, g, db)
        assert not can_contain(op, h, db)

    def test_condition_2_timing(self):
        """A record that ended before R started is not claimed."""
        db = _db()
        g_early = _rec(db, "g", "function", 1, 1.0)
        f = _rec(db, "f", "function", 10, 5.0)  # started at 4.99 s
        roots = build_tree([g_early, f], db)
        assert [r.record.function for r in roots] == ["g", "f"]

    def test_condition_3_sum(self):
        """Claiming stops when the children would exceed the parent."""
        db = _db()
        g1 = _rec(db, "g", "function", 6, 1.0, line=1)
        g2 = _rec(db, "g", "function", 6, 1.0, line=2)
        f = _rec(db, "f", "function", 10, 1.0, line=3)
        roots = build_tree([g1, g2, f], db)
        assert [r.record.line for r in roots] == [1, 3]
        assert [c.record.line for c in roots[1].children] == [2]
        assert roots[1].exclusive == pytest.approx(4)

    def test_nested_ops_and_order(self):
        """Children keep log order; unclaimed records stay top level."""
        db = _db()
        g = _rec(db, "g", "function", 2, 1.0, line=1)
        op = _rec(db, "f", "operation", 3, 1.0, line=2)
        h = _rec(db, "h", "function", 1, 1.0, line=3)
        f = _rec(db, "f", "function", 10, 1.0, line=4)
        # h is not reachable from f: it stays at the top level
        h_root, f_root = build_tree([g, op, h, f], db)
        assert h_root.record.line == 3
        assert [c.record.line for c in f_root.children] == [2]
        assert [c.record.line for c in f_root.children[0].children] == [1]

    def test_unknown_records(self):
        """Unknown timers nest by function name; unknown ops claim nothing."""
        db = _db()
        unknown_g = Record(0, 1.0, 1, "g", "function", "unknown:g", None, 1)
        unknown_op = Record(
            0, 1.0, 2, "f", "operation", "unknown:f: x", None, 2
        )
        f = _rec(db, "f", "function", 10, 1.0, line=3)
        assert not can_contain(unknown_op, unknown_g, db)
        (root,) = build_tree([unknown_g, unknown_op, f], db)
        assert [c.record.line for c in root.children] == [1, 2]
