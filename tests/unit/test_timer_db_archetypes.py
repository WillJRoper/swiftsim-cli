"""Data-driven tests of the timer parser on archetypal C syntax.

Every ``tests/data/c_archetypes/<category>/<case>.yaml`` is one case (see
the README there for the convention). Each check below is parametrised
over the cases that declare it, so adding a case means adding files.
"""

import os
import re
import shutil
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML

from swiftsim_cli.timing.log_scan import scan_log
from swiftsim_cli.timing.timer_db import (
    _PRINTF_SPEC,
    _TOOK,
    UNIT_TO_MS,
    LogicalTimer,
    TimerDB,
    _source_files,
    build_timer_db,
    parse_source,
    timer_key,
)

ARCHETYPES = Path(__file__).parents[1] / "data" / "c_archetypes"
CASES = sorted(ARCHETYPES.rglob("*.yaml"))
CHECKS = (
    "functions",
    "sites",
    "timers",
    "calls",
    "reach",
    "op_reach",
    "lines",
    "log",
)


def _case_id(path: Path) -> str:
    return path.relative_to(ARCHETYPES).with_suffix("").as_posix()


@cache
def _spec(case: str) -> dict[str, Any]:
    return YAML(typ="safe").load(ARCHETYPES / f"{case}.yaml") or {}


def _sources(case: str) -> Path:
    """The case's source: ``<case>.c`` or the directory ``<case>/``."""
    single = ARCHETYPES / f"{case}.c"
    return single if single.is_file() else ARCHETYPES / case


def _root(case: str, tmp: Path) -> Path:
    """A directory holding only the case's sources."""
    root = tmp / case
    src = _sources(case)
    if src.is_file():
        root.mkdir(parents=True)
        shutil.copy(src, root / src.name)
    else:
        shutil.copytree(src, root)
    return root


class Case:
    """A parsed archetype case."""

    def __init__(self, case: str, tmp: Path) -> None:
        """Parse the case's sources both per file and as a database."""
        self.name = case
        self.spec = _spec(case)
        self.root = _root(case, tmp)
        self.db = TimerDB(build_timer_db(self.root))
        self.files = {
            p.relative_to(self.root).as_posix(): p.read_bytes()
            for p in sorted(self.root.rglob("*.[ch]"))
        }
        self.parsed = {
            rel: parse_source(src, rel) for rel, src in self.files.items()
        }
        self.sites = {
            f"{s['file']}:{s['line']}": s
            for p in self.parsed.values()
            for s in p.sites
        }
        self.timer_of = {
            site: timer for timer in self.db.timers for site in timer.sites
        }


@pytest.fixture(scope="module")
def load(tmp_path_factory):
    """Load (and cache) a case by name."""
    tmp = tmp_path_factory.mktemp("archetypes")
    loaded: dict[str, Case] = {}

    def _load(case: str) -> Case:
        if case not in loaded:
            loaded[case] = Case(case, tmp)
        return loaded[case]

    return _load


def _params(check: str) -> list[Any]:
    """One param per case declaring ``check`` (xfail where flagged)."""
    params = []
    for path in CASES:
        case = _case_id(path)
        spec = _spec(case)
        if check != "render" and check not in spec:
            continue
        reason = (spec.get("xfail") or {}).get(check)
        marks = (
            [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        )
        params.append(pytest.param(case, id=case, marks=marks))
    return params


def test_corpus_is_well_formed():
    """Every case has sources, a description and only known keys."""
    assert CASES, "no archetype cases found"
    known = {*CHECKS, "description", "xfail"}
    for path in CASES:
        case = _case_id(path)
        spec = _spec(case)
        assert _sources(case).exists(), f"{case}: no sources"
        assert spec.get("description"), f"{case}: no description"
        assert set(spec) <= known, f"{case}: unknown keys {set(spec) - known}"
        assert set(spec.get("xfail") or {}) <= set(CHECKS) | {"render"}


# =============================================================================
# Source structure
# =============================================================================


@pytest.mark.parametrize("case", _params("functions"))
def test_functions(case, load):
    """Function definitions found, with their line spans (exact)."""
    c = load(case)
    single = len(c.files) == 1
    found: dict[str, list[list[int]]] = {}
    for rel, parsed in c.parsed.items():
        for f in parsed.functions:
            key = f.name if single else f"{rel}:{f.name}"
            found.setdefault(key, []).append(list(f.lines))
    actual = {k: v[0] if len(v) == 1 else v for k, v in found.items()}
    assert actual == c.spec["functions"]


@pytest.mark.parametrize("case", _params("sites"))
def test_sites(case, load):
    """Timer sites (exact set) and their function, kind, label and tic."""
    c = load(case)
    expected = c.spec["sites"] or {}
    assert set(c.timer_of) == set(expected)
    for site, want in expected.items():
        timer = c.timer_of[site]
        file, line = site.rsplit(":", 1)
        rng = timer.ranges[timer.sites.index(site)]
        got = dict(
            function=timer.function,
            kind=timer.kind,
            label=timer.label,
            tic=c.sites[site]["tic"],
            start=rng[1],
            format=c.sites[site]["format"],
        )
        assert rng[0] == file and rng[2] == int(line)
        assert {k: got[k] for k in want} == want, site


@pytest.mark.parametrize("case", _params("timers"))
def test_logical_timers(case, load):
    """Grouping of sites into logical timers (exact partition)."""
    c = load(case)
    actual = sorted(sorted(t.sites) for t in c.db.timers)
    assert actual == sorted(sorted(g) for g in c.spec["timers"])


@pytest.mark.parametrize("case", _params("calls"))
def test_calls(case, load):
    """Call graph edges of the listed callers (exact per caller).

    A value is either a list of callees or a mapping callee -> call-site
    lines (the latter is only recorded for functions with operations).
    """
    c = load(case)
    for caller, want in c.spec["calls"].items():
        assert c.db.calls.get(caller, []) == sorted(want), caller
        if isinstance(want, dict):
            lines: dict[str, list[int]] = {}
            for _, line, callee in c.db.call_sites.get(caller, []):
                lines.setdefault(callee, []).append(line)
            assert lines == want, caller


@pytest.mark.parametrize("case", _params("reach"))
def test_reach(case, load):
    """Functions reachable through calls (exact)."""
    c = load(case)
    for function, want in c.spec["reach"].items():
        assert c.db.reach(function) == set(want), function


@pytest.mark.parametrize("case", _params("op_reach"))
def test_op_reach(case, load):
    """Functions reachable from calls inside an operation's range."""
    c = load(case)
    for site, want in c.spec["op_reach"].items():
        assert c.db.op_reach(c.timer_of[site]) == set(want), site


# =============================================================================
# Regex round trips
# =============================================================================


def _match(timer: LogicalTimer, body: str) -> tuple[float, str] | None:
    """(ms, key) the timer parses from a log-line body, or None."""
    m = timer.regex.match(body)
    if m is None:
        return None
    unit = m.groupdict().get("unit") or "ms"
    return float(m["ms"]) * UNIT_TO_MS[unit], timer.key(m)


@pytest.mark.parametrize("case", _params("lines"))
def test_lines(case, load):
    """Hand-written bodies: expected ms and key, near misses rejected."""
    c = load(case)
    for site, want in c.spec["lines"].items():
        timer = c.timer_of[site]
        for item in want.get("match", []):
            got = _match(timer, item["body"])
            assert got is not None, (site, item["body"])
            assert got[0] == pytest.approx(item["ms"]), item["body"]
            if "key" in item:
                assert got[1] == item["key"], item["body"]
        for body in want.get("reject", []):
            assert _match(timer, body) is None, (site, body)


def _render(fmt: str, value: float, unit: str) -> tuple[str, float, list[str]]:
    """Render a timer format like printf would.

    Returns the log-line body, the milliseconds it reports and the values
    printed for each ``%s`` before "took". The time is the first float
    conversion after "took" (else the first integer one) and a ``%s``
    right after it is the unit.
    """
    fmt = " ".join(fmt.split())
    took = _TOOK.search(fmt)
    assert took is not None
    specs = list(_PRINTF_SPEC.finditer(fmt))
    after = [m for m in specs if m.start() >= took.end()]
    floats = [m for m in after if m.group()[-1] in "fFeEgG"]
    ints = [m for m in after if m.group()[-1] in "diu"]
    time_spec = (floats or ints)[0]
    out: list[str] = []
    labels: list[str] = []
    pos, ms_value = 0, 0.0
    printed_unit: str | None = None
    for m in specs:
        out.append(fmt[pos : m.start()])
        spec = re.sub(r"(hh|h|ll|l|j|z|t|L)(?=[a-zA-Z]$)", "", m.group())
        conv = spec[-1]
        if m is time_spec:
            text = spec % (int(value) if conv in "diu" else value)
            ms_value = float(text)
            out.append(text)
        elif conv == "%":
            out.append("%")
        elif conv == "s" and m.start() < took.start():
            labels.append(f"lab{len(labels)}")
            out.append(labels[-1])
        elif (
            conv == "s"
            and m.start() > time_spec.start()
            and printed_unit is None
            and not fmt[time_spec.end() : m.start()].strip()
        ):
            printed_unit = unit
            out.append(unit)
        elif conv == "s":
            out.append("xyz")
        else:
            out.append(spec % (3 if conv in "dicuoxX" else 2.5))
        pos = m.end()
    out.append(fmt[pos:])
    body = "".join(out).rstrip()
    literal = re.match(
        r"\s*(milliseconds|seconds|ms|us|ns|s)\b", fmt[time_spec.end() :]
    )
    used = printed_unit or (literal.group(1) if literal else "ms")
    ms = ms_value * UNIT_TO_MS[used]
    return body, ms, labels


@pytest.mark.parametrize("case", _params("render"))
def test_rendered_formats_round_trip(case, load):
    """Every site's format, rendered printf-style, parses back exactly."""
    c = load(case)
    for site, timer in c.timer_of.items():
        fmt = c.sites[site]["format"]
        for value, unit in ((12.345, "ms"), (1.5, "s"), (250.0, "us")):
            body, ms, labels = _render(fmt, value, unit)
            got = _match(timer, body)
            assert got is not None, (site, body, timer.pattern)
            assert got[0] == pytest.approx(ms), (site, body)
            label = timer.label
            for value_label in labels:
                label = re.sub(r"%s", value_label, label, count=1)
            assert got[1] == timer_key(timer.function, timer.kind, label)
            # Near misses: no number, or the text before "took" missing.
            no_number = re.sub(r"took(.*?)\d", r"took\1x", body, count=1)
            assert _match(timer, no_number) is None, (site, no_number)
            took = _TOOK.search(fmt)
            assert took is not None
            prefix = fmt[: took.start()]
            if re.search(r"[A-Za-z0-9]", _PRINTF_SPEC.sub("", prefix)):
                bare = body[body.index("took") :]
                assert _match(timer, bare) is None, (site, bare)


# =============================================================================
# Full log dispatch
# =============================================================================


@pytest.mark.parametrize("case", _params("log"))
def test_log_dispatch(case, load, tmp_path):
    """Real-looking log lines go through scan_log to the right timer.

    ``log.records`` lists ``[rank, function, key, ms]`` in log order;
    ``log.unmatched`` maps unknown "function: label" keys to counts.
    """
    c = load(case)
    want = c.spec["log"]
    path = tmp_path / "log.txt"
    path.write_text(want["text"])
    scan = scan_log(path, c.db)
    records = sorted(
        (r for s in scan.steps for rs in s.records.values() for r in rs),
        key=lambda r: r.line,
    )
    actual = [[r.rank, r.function, r.key, r.ms] for r in records]
    expected = want.get("records", [])
    assert [a[:3] for a in actual] == [e[:3] for e in expected]
    assert [a[3] for a in actual] == pytest.approx([e[3] for e in expected])
    assert dict(scan.unmatched) == want.get("unmatched", {})


# =============================================================================
# Real SWIFT checkout (opt-in)
# =============================================================================

# Timer messages the parser may not place in a function (none today).
REAL_SWIFT_UNPLACED: set[str] = set()


@pytest.mark.skipif(
    not os.environ.get("SWIFT_SRC"), reason="SWIFT_SRC not set"
)
def test_real_swift_checkout():
    """Sane invariants on a real SWIFT tree (set SWIFT_SRC to run)."""
    root = Path(os.environ["SWIFT_SRC"])
    payload = build_timer_db(root)
    db = TimerDB(payload)
    stats = payload["stats"]
    print(f"\nSWIFT_SRC stats: {stats}")
    assert stats["sites"] >= 200

    formats, unplaced = {}, []
    for path in _source_files(root):
        rel = path.relative_to(root).as_posix()
        parsed = parse_source(path.read_bytes(), rel)
        unplaced += parsed.unplaced
        formats.update({f"{s['file']}:{s['line']}": s for s in parsed.sites})
    assert set(unplaced) <= REAL_SWIFT_UNPLACED

    sites = [s for t in db.timers for s in t.sites]
    assert len(sites) == len(set(sites))
    assert not [s for s in sites if re.search(r"(^|/)(tests|examples)/", s)]
    for timer in db.timers:
        for site in timer.sites:
            body, ms, _ = _render(formats[site]["format"], 12.345, "ms")
            got = _match(timer, body)
            assert got is not None, (site, body, timer.pattern)
            assert got[0] == pytest.approx(ms), (site, body)
