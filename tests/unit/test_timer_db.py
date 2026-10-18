"""Tests for the tree-sitter timer database."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from swiftsim_cli.timing import timer_db
from swiftsim_cli.timing.timer_db import (
    TimerDB,
    build_timer_db,
    load_source_db,
    parse_timer_format,
    resolve_timer_db,
    source_key,
)


def _timers(db: TimerDB, function: str):
    return {t.label: t for t in db.by_function[function]}


class TestParseTimerFormat:
    """Format string -> regex, label and kind."""

    @pytest.mark.parametrize(
        "fmt, body, ms",
        [
            ("took %.3f %s.", "took 1.036 ms.", 1.036),
            ("took %f %s", "took 30.110611 ms", 30.110611),
            (
                "Setting initial group index took: %.3f %s.",
                "Setting initial group index took: 1.206 ms.",
                1.206,
            ),
            (
                "fof_link() took (FOF SCALING): %.3f %s.",
                "fof_link() took (FOF SCALING): 548.263 ms.",
                548.263,
            ),
            ("(%s) took %.3f %s.", "(tasks) took 2190.281 ms.", 2190.281),
            (
                "took %.3f %s (including reweight).",
                "took 26.726 ms (including reweight).",
                26.726,
            ),
            (
                "Writing   step info\nto files took %.3f %s  ",
                "Writing step info to files took 0.010 ms",
                0.010,
            ),
            ("Finding IDstook: %.3f %s.", "Finding IDstook: 2.0 ms.", 2.0),
        ],
    )
    def test_regex_matches_log_body(self, fmt, body, ms):
        """Each format's regex matches the line it prints."""
        import re

        parsed = parse_timer_format(fmt)
        assert parsed is not None
        m = re.match(parsed.pattern, body)
        assert m is not None
        assert float(m["ms"]) == pytest.approx(ms)

    def test_unit_group(self):
        """The %s after the time is captured as the unit."""
        import re

        parsed = parse_timer_format("took %.3f %s.")
        assert parsed is not None
        m = re.match(parsed.pattern, "took 2 s.")
        assert m is not None
        assert m["unit"] == "s"

    @pytest.mark.parametrize(
        "fmt, kind",
        [
            ("took %.3f %s.", "function"),
            ("(%s) took %.3f %s.", "function"),
            (
                "took %.3f %s (including unskip, rebuild and reweight).",
                "function",
            ),
            ("took: %.3f %s.", "function"),
            ("Making gravity tasks took %.3f %s.", "operation"),
            ("took %.3f %s to find active cells.", "operation"),
            ("fof_search() took (FOF SCALING): %.3f %s.", "operation"),
        ],
    )
    def test_one_static_rule(self, fmt, kind):
        """Function timer iff no literal text around the time."""
        parsed = parse_timer_format(fmt)
        assert parsed is not None
        assert parsed.kind == kind

    def test_runtime_label_groups(self):
        """%s before 'took' becomes a label group used in the key."""
        import re

        parsed = parse_timer_format("Copying for '%s' took %.3f %s.")
        assert parsed is not None
        m = re.match(parsed.pattern, "Copying for 'gas' took 1.0 ms.")
        assert m is not None
        assert m["l0"] == "gas"

    def test_not_a_timer(self):
        """No number after 'took' -> not a timer."""
        assert parse_timer_format("took a while") is None
        assert parse_timer_format("nothing here %d") is None


class TestSyntheticTree:
    """Parsing the synthetic C tree in tests/data/swift_src."""

    def test_pointer_returning_function(self, synthetic_db):
        """`struct cell *f(...)` is named f."""
        assert "special_pointer_function" in synthetic_db.by_function

    def test_comment_and_string_braces(self, synthetic_db):
        """Braces in comments/strings do not move the timers."""
        timers = _timers(synthetic_db, "special_comment_brace")
        assert set(timers) == {"", "Doing the thing"}
        assert timers[""].kind == "function"
        assert timers["Doing the thing"].kind == "operation"
        # The next function is still found.
        assert "special_mapped_timed" in synthetic_db.by_function

    def test_ifdef_unbalanced_braces_use_fallback(self, synthetic_db):
        """Sites tree-sitter cannot place are attributed by brace counting."""
        timers = _timers(synthetic_db, "special_ifdef")
        assert set(timers) == {"MPI branch", "Serial branch"}
        assert synthetic_db.stats["fallback_sites"] >= 1

    def test_suffix_rule(self, synthetic_db):
        """Unparenthesised text after the time makes an operation."""
        timers = _timers(synthetic_db, "special_suffixes")
        assert timers[""].kind == "function"
        assert timers["took ... to find active cells"].kind == "operation"

    def test_duplicate_sites_grouped(self, synthetic_db):
        """Identical (function, regex) sites form one logical timer."""
        (timer,) = synthetic_db.by_function["special_duplicate"]
        assert len(timer.sites) == 2
        assert len(timer.ranges) == 2

    def test_commented_out_timer_ignored(self, synthetic_db):
        """A message in a comment is not a timer."""
        assert "special_commented" not in synthetic_db.by_function

    def test_took_colon_and_runtime_label_formats(self, synthetic_db):
        """`took:` and `(%s) took` formats are parsed."""
        fof = _timers(synthetic_db, "fof_allocate")
        assert fof["Setting initial group index"].kind == "operation"
        (launch,) = synthetic_db.by_function["engine_launch"]
        assert launch.kind == "function"
        assert launch.label == "(%s)"

    def test_threadpool_mapper_is_a_call(self, synthetic_db):
        """Functions passed as arguments are reachable."""
        reach = synthetic_db.reach("special_threadpool_caller")
        assert {"special_mapper", "special_mapped_timed"} <= reach

    def test_reach_through_untimed_intermediate(self, synthetic_db):
        """engine_step -> pm_mesh_compute_potential -> compute_potential."""
        assert "compute_potential_global" in synthetic_db.reach("engine_step")

    def test_multi_caller_structure(self, synthetic_db):
        """engine_unskip is reached from rebuild and prepare."""
        assert "engine_rebuild" in synthetic_db.reach("engine_prepare")
        assert "engine_unskip" in synthetic_db.calls["engine_rebuild"]
        assert "engine_unskip" in synthetic_db.calls["engine_prepare"]
        assert "engine_rebuild" in synthetic_db.reach("engine_init_particles")
        assert "scheduler_reweight" in synthetic_db.calls["engine_maketasks"]

    def test_op_reach_uses_line_range(self, synthetic_db):
        """Only calls inside an operation's range are reachable from it."""
        ops = _timers(synthetic_db, "engine_maketasks")
        gravity = synthetic_db.op_reach(ops["Making gravity tasks"])
        hydro = synthetic_db.op_reach(ops["Making hydro tasks"])
        assert "zoom_engine_make_self_gravity_tasks" in gravity
        assert "zoom_engine_make_self_gravity_tasks" not in hydro

    def test_tic_choice(self, synthetic_db):
        """An operation starts at its own getticks, not the function's."""
        ops = _timers(synthetic_db, "engine_maketasks")
        (_, start, end) = ops["Making gravity tasks"].ranges[0]
        (_, fstart, fend) = ops[""].ranges[0]
        assert fstart < start < end < fend

    def test_lexically_inside(self):
        """Nested operation ranges are detected."""
        payload = {
            "timers": [
                dict(
                    function="f",
                    kind="operation",
                    pattern="^a",
                    label="a",
                    sites=["x.c:9"],
                    ranges=[["x.c", 2, 9]],
                ),
                dict(
                    function="f",
                    kind="operation",
                    pattern="^b",
                    label="b",
                    sites=["x.c:5"],
                    ranges=[["x.c", 3, 5]],
                ),
            ],
            "calls": {},
            "call_sites": {},
        }
        db = TimerDB(payload)
        outer, inner = db.timers
        assert db.lexically_inside(inner, outer)
        assert not db.lexically_inside(outer, inner)

    def test_excluded_directories(self, tmp_path):
        """tests/, examples/ and argparse/ are not parsed."""
        timer = (
            'void f(void){ticks t=getticks(); message("took %.3f %s.",1,"");}'
        )
        for sub in ("src", "src/tests", "src/examples", "src/argparse"):
            (tmp_path / sub).mkdir(parents=True, exist_ok=True)
            name = sub.replace("/", "_")
            (tmp_path / sub / f"{name}.c").write_text(
                timer.replace("f(", f"{name}(")
            )
        db = TimerDB(build_timer_db(tmp_path))
        assert set(db.by_function) == {"src"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def tiny_repo(tmp_path, monkeypatch):
    """A git repo with two commits whose timers differ."""
    repo = tmp_path / "swift"
    (repo / "src").mkdir(parents=True)
    src = repo / "src" / "engine.c"
    src.write_text(
        'void old_fn(void){ticks t=getticks();message("took %.3f %s.",1,"");}'
    )
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "one")
    first = _git(repo, "rev-parse", "HEAD")
    src.write_text(
        'void new_fn(void){ticks t=getticks();message("took %.3f %s.",1,"");}'
    )
    _git(repo, "commit", "-qam", "two")
    monkeypatch.setattr(
        timer_db,
        "load_swift_profile",
        lambda: SimpleNamespace(swiftsim_dir=repo),
    )
    return repo, first


class TestCacheAndRevisions:
    """Database caching and revision matching."""

    def test_revision_auto_match(self, tiny_repo):
        """A log from an older commit gets that commit's timers."""
        repo, first = tiny_repo
        db, status = resolve_timer_db(first[:8])
        assert set(db.by_function) == {"old_fn"}
        assert "matched log revision" in status
        assert _git(repo, "status", "--porcelain") == ""

    def test_head_revision_uses_working_tree(self, tiny_repo):
        """A log from HEAD uses the working tree."""
        repo, _ = tiny_repo
        head = _git(repo, "rev-parse", "HEAD")
        db, status = resolve_timer_db(head[:10])
        assert set(db.by_function) == {"new_fn"}
        assert "matches the working tree" in status

    def test_no_match_revision(self, tiny_repo):
        """--no-match-revision uses the working tree."""
        _, first = tiny_repo
        db, status = resolve_timer_db(first, match_revision=False)
        assert set(db.by_function) == {"new_fn"}
        assert "disabled" in status

    def test_unknown_revision_falls_back(self, tiny_repo):
        """A revision missing from the repo falls back with a warning."""
        db, status = resolve_timer_db("deadbeef")
        assert set(db.by_function) == {"new_fn"}
        assert "not found" in status

    def test_swift_src_override(self, tiny_repo, swift_src_dir):
        """--swift-src wins over the profile."""
        db, status = resolve_timer_db("deadbeef", swift_src=swift_src_dir)
        assert "engine_rebuild" in db.by_function
        assert "--swift-src" in status

    def test_no_source_raises(self, monkeypatch):
        """Without a checkout there is a clear error."""
        monkeypatch.setattr(
            timer_db,
            "load_swift_profile",
            lambda: SimpleNamespace(swiftsim_dir=None),
        )
        with pytest.raises(ValueError, match="--swift-src"):
            resolve_timer_db(None)

    def test_cache_staleness(self, tiny_repo, monkeypatch):
        """Rebuilt only when HEAD, uncommitted changes or the flag say so."""
        repo, _ = tiny_repo
        calls = []
        real_build = timer_db.build_timer_db

        def counting_build(root):
            calls.append(root)
            return real_build(root)

        monkeypatch.setattr(timer_db, "build_timer_db", counting_build)
        key = source_key(repo)
        load_source_db(repo)
        load_source_db(repo)
        assert len(calls) == 1

        # An uncommitted edit changes the key and triggers a rebuild...
        (repo / "src" / "engine.c").write_text(
            'void edited(void){message("took %.3f %s.",1,"");}'
        )
        assert source_key(repo) != key
        db = load_source_db(repo)
        assert len(calls) == 2 and "edited" in db.by_function

        # ...and so does a further edit of the same modified file.
        key2 = source_key(repo)
        (repo / "src" / "engine.c").write_text(
            'void edited_again(void){message("took %.3f %s.",1,"");}   '
        )
        assert source_key(repo) != key2

        load_source_db(repo, regenerate=True)
        load_source_db(repo, regenerate=True)
        assert len(calls) == 4

    def test_parser_version_in_key(self, tiny_repo, monkeypatch):
        """A parser upgrade invalidates the cache."""
        repo, _ = tiny_repo
        key = source_key(repo)
        monkeypatch.setattr(timer_db, "PARSER_VERSION", 999)
        assert source_key(repo) != key

    def test_non_git_tree_key(self, tmp_path):
        """Plain directories are keyed on file sizes and times."""
        (tmp_path / "a.c").write_text("void a(void){}")
        key = source_key(tmp_path)
        (tmp_path / "a.c").write_text("void a(void){ }")
        assert source_key(tmp_path) != key

    def test_cache_file_is_json(self, tiny_repo):
        """The cache lives in the profile cache dir as JSON."""
        repo, _ = tiny_repo
        load_source_db(repo)
        files = list(timer_db._db_cache_dir().glob("tree-*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["key"] == source_key(repo)
