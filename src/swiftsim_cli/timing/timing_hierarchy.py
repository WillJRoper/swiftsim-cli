"""Aggregate per-step timer trees into call-path statistics and tables.

Every number here comes from the per-step trees of one MPI rank (ranks are
never summed). Times are aggregated by *call path* (the chain of timers
from the step down to the timer), so a function called from several places
gets one entry per caller. Each entry has:

* inclusive: the time the timer reported;
* exclusive: inclusive minus the time of the timers it contains.

Exclusive times of all entries add up to the timed part of the steps, so
they can be shown as shares of the step time; inclusive times of nested
entries overlap and cannot be summed.

The step time is the sum of the step-table wall-clock times of the steps
analysed. The top-level breakdown (the timers not contained in any other,
plus the untimed remainder) sums to 100% of it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from swiftsim_cli.timing.log_scan import Step
from swiftsim_cli.timing.timer_db import TimerDB
from swiftsim_cli.timing.timer_nesting import Node, build_tree
from swiftsim_cli.utilities.tables import create_ascii_table

UNTIMED = "untimed in step"

Path = tuple[str, ...]


@dataclass
class PathStats:
    """Statistics of one call path."""

    calls: int = 0
    inclusive: float = 0.0
    exclusive: float = 0.0
    max: float = 0.0
    steps: set[int] = field(default_factory=set)

    @property
    def mean(self) -> float:
        """Mean inclusive time per call (ms)."""
        return self.inclusive / self.calls if self.calls else 0.0

    def add(self, other: PathStats) -> None:
        """Accumulate another entry into this one."""
        self.calls += other.calls
        self.inclusive += other.inclusive
        self.exclusive += other.exclusive
        self.max = max(self.max, other.max)
        self.steps |= other.steps


@dataclass
class StepTree:
    """The timer tree of one rank in one step."""

    step: Step
    rank: int
    roots: list[Node]

    @property
    def timed(self) -> float:
        """Time covered by the top-level timers (ms)."""
        return sum(r.inclusive for r in self.roots)

    @property
    def coverage(self) -> float | None:
        """Timed fraction of the step's wall-clock time (percent)."""
        wall = self.step.wallclock_ms
        return 100.0 * self.timed / wall if wall else None


def build_step_trees(
    steps: list[Step], rank: int, db: TimerDB
) -> list[StepTree]:
    """Nest the records of one rank in each step."""
    return [
        StepTree(step, rank, build_tree(step.records.get(rank, []), db))
        for step in steps
    ]


def select_steps(
    steps: list[Step], step_range: tuple[int, int] | None
) -> list[Step]:
    """Complete steps with a wall-clock time, optionally within a range."""
    return [
        s
        for s in steps
        if s.kind == "step"
        and s.wallclock_ms
        and (step_range is None or step_range[0] <= s.number <= step_range[1])
    ]


@dataclass
class Hierarchy:
    """Call-path statistics of one rank over a set of steps.

    Attributes:
        rank: The MPI rank.
        paths: Statistics per call path.
        kinds: Timer key -> (function, kind).
        total: The denominator: summed wall-clock time of the steps (ms).
        n_steps: Number of steps aggregated.
        negative: Nodes whose children add up to more than 1% over them.
    """

    rank: int
    paths: dict[Path, PathStats]
    kinds: dict[str, tuple[str, str]]
    total: float
    n_steps: int
    negative: int = 0
    _children: dict[Path, list[Path]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Index the children of each path."""
        children: dict[Path, list[Path]] = defaultdict(list)
        for path in self.paths:
            children[path[:-1]].append(path)
        for kids in children.values():
            kids.sort(key=lambda p: -self.paths[p].inclusive)
        self._children = dict(children)

    def children(self, path: Path) -> list[Path]:
        """Child paths, largest inclusive time first."""
        return self._children.get(path, [])

    @property
    def timed(self) -> float:
        """Time covered by top-level timers (ms)."""
        return sum(self.paths[p].inclusive for p in self.children(()))

    @property
    def untimed(self) -> float:
        """Step time not covered by any timer (ms)."""
        return self.total - self.timed

    def percent(self, value: float) -> float:
        """Value as a percentage of the step time (0 if there is none)."""
        return 100.0 * value / self.total if self.total > 0 else 0.0

    def flat(self) -> list[tuple[str, PathStats, float]]:
        """Totals per timer, merged over call paths.

        Returns:
            (key, stats, outermost inclusive) per timer, largest exclusive
            time first. ``stats.inclusive`` sums every occurrence; the
            outermost inclusive counts a timer nested in itself once.
        """
        merged: dict[str, PathStats] = defaultdict(PathStats)
        outer: dict[str, float] = defaultdict(float)
        for path, stats in self.paths.items():
            merged[path[-1]].add(stats)
            if path[-1] not in path[:-1]:
                outer[path[-1]] += stats.inclusive
        return sorted(
            ((k, s, outer[k]) for k, s in merged.items()),
            key=lambda row: -row[1].exclusive,
        )

    def matches(self, key: str, name: str) -> bool:
        """Whether a timer key is the timer called ``name``.

        ``name`` may be a key or the name of a function with a function
        timer.
        """
        function, kind = self.kinds.get(key, ("", ""))
        return key == name or (function == name and kind == "function")

    def known(self, name: str) -> bool:
        """Whether any timer in the hierarchy matches ``name``."""
        return any(self.matches(key, name) for key in self.kinds)

    def merged_subtree(
        self, name: str
    ) -> tuple[dict[Path, PathStats], dict[str, PathStats]]:
        """Merge every subtree rooted at a timer, whatever its callers.

        Returns:
            (subtree, callers): statistics per path relative to the timer
            (the timer itself is ``(name,)``), and per caller ("<step>" for
            top level).
        """
        subtree: dict[Path, PathStats] = defaultdict(PathStats)
        callers: dict[str, PathStats] = defaultdict(PathStats)
        for path, stats in self.paths.items():
            index = next(
                (i for i, k in enumerate(path) if self.matches(k, name)), None
            )
            if index is None:
                continue
            rel = (name, *path[index + 1 :])
            subtree[rel].add(stats)
            if len(rel) == 1:
                caller = path[index - 1] if index else "<step>"
                callers[caller].add(stats)
        return dict(subtree), dict(callers)


def aggregate(trees: list[StepTree], rank: int, total: float) -> Hierarchy:
    """Aggregate step trees into call-path statistics.

    Args:
        trees: The trees to aggregate (one rank).
        rank: The rank they belong to.
        total: The denominator for percentages (ms).
    """
    paths: dict[Path, PathStats] = defaultdict(PathStats)
    kinds: dict[str, tuple[str, str]] = {}
    negative = 0

    def walk(node: Node, parent: Path, step: int) -> None:
        nonlocal negative
        record = node.record
        path = (*parent, record.key)
        kinds.setdefault(record.key, (record.function, record.kind))
        stats = paths[path]
        stats.calls += 1
        stats.inclusive += record.ms
        exclusive = node.exclusive
        stats.exclusive += exclusive
        stats.max = max(stats.max, record.ms)
        stats.steps.add(step)
        if exclusive < -0.01 * record.ms:
            negative += 1
        for child in node.children:
            walk(child, path, step)

    for tree in trees:
        for root in tree.roots:
            walk(root, (), tree.step.number)
    return Hierarchy(rank, dict(paths), kinds, total, len(trees), negative)


def hierarchy_for(
    steps: list[Step],
    rank: int,
    db: TimerDB,
    step_range: tuple[int, int] | None = None,
) -> Hierarchy:
    """Aggregated statistics of one rank over the selected steps."""
    selected = select_steps(steps, step_range)
    total = sum(s.wallclock_ms or 0.0 for s in selected)
    return aggregate(build_step_trees(selected, rank, db), rank, total)


def samples_by_key(trees: list[StepTree]) -> dict[str, list[float]]:
    """Per-call inclusive times of every timer in the trees."""
    samples: dict[str, list[float]] = defaultdict(list)
    stack = [node for tree in trees for node in tree.roots]
    while stack:
        node = stack.pop()
        samples[node.record.key].append(node.record.ms)
        stack.extend(node.children)
    return dict(samples)


# =============================================================================
# Tables
# =============================================================================


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def top_level_table(h: Hierarchy, top_n: int, title: str) -> str:
    """Top-level timers plus the untimed remainder (sums to 100%)."""
    headers = ["Timer", "Time (ms)", "% of step time", "Calls", "Steps"]
    rows = []
    kids = h.children(())
    for path in kids[:top_n]:
        s = h.paths[path]
        rows.append(
            [
                path[-1],
                _fmt(s.inclusive),
                _pct(h.percent(s.inclusive)),
                s.calls,
                len(s.steps),
            ]
        )
    rest = kids[top_n:]
    if rest:
        rest_time = sum(h.paths[p].inclusive for p in rest)
        rows.append(
            [
                f"({len(rest)} other timers)",
                _fmt(rest_time),
                _pct(h.percent(rest_time)),
                sum(h.paths[p].calls for p in rest),
                "",
            ]
        )
    rows.append([UNTIMED, _fmt(h.untimed), _pct(h.percent(h.untimed)), "", ""])
    rows.append(
        ["TOTAL", _fmt(h.total), _pct(100.0 if h.total else 0), "", ""]
    )
    return create_ascii_table(headers, rows, title)


def _tree_rows(
    h: Hierarchy,
    path: Path,
    parent_time: float,
    depth: int,
    top_n: int,
    stats_of: dict[Path, PathStats],
    children_of: dict[Path, list[Path]],
    rows: list[list[object]],
) -> None:
    """Depth-first rows of a call tree."""
    kids = children_of.get(path, [])
    for child in kids[:top_n]:
        s = stats_of[child]
        rows.append(
            [
                "  " * depth + child[-1],
                _fmt(s.inclusive),
                _fmt(s.exclusive),
                _pct(100 * s.inclusive / parent_time if parent_time else 0),
                _pct(h.percent(s.inclusive)),
                s.calls,
                f"{s.mean:.2f}",
                _fmt(s.max),
            ]
        )
        _tree_rows(
            h,
            child,
            s.inclusive,
            depth + 1,
            top_n,
            stats_of,
            children_of,
            rows,
        )
    if len(kids) > top_n:
        rest = sum(stats_of[p].inclusive for p in kids[top_n:])
        rows.append(
            [
                "  " * depth + f"({len(kids) - top_n} more)",
                _fmt(rest),
                "",
                _pct(100 * rest / parent_time if parent_time else 0),
                _pct(h.percent(rest)),
                "",
                "",
                "",
            ]
        )


_TREE_HEADERS = [
    "Timer",
    "Incl. (ms)",
    "Excl. (ms)",
    "% of parent",
    "% of step time",
    "Calls",
    "Mean (ms)",
    "Max (ms)",
]


def tree_table(h: Hierarchy, top_n: int, title: str) -> str:
    """The aggregated call tree (``top_n`` children per node)."""
    rows: list[list[object]] = []
    _tree_rows(h, (), h.total, 0, top_n, h.paths, h._children, rows)
    return create_ascii_table(_TREE_HEADERS, rows, title)


def flat_table(h: Hierarchy, top_n: int, title: str) -> str:
    """Timers ranked by exclusive time, with inclusive alongside."""
    headers = [
        "Timer",
        "Excl. (ms)",
        "Excl. % of step",
        "Incl. (ms)*",
        "Incl. % of step*",
        "Calls",
        "Mean incl. (ms)",
    ]
    rows = [
        [
            key,
            _fmt(s.exclusive),
            _pct(h.percent(s.exclusive)),
            _fmt(outer),
            _pct(h.percent(outer)),
            s.calls,
            f"{s.mean:.2f}",
        ]
        for key, s, outer in h.flat()[:top_n]
    ]
    note = (
        "\n* Inclusive times contain nested timers and are not additive; "
        "exclusive times add up to the timed share of the step time."
    )
    return create_ascii_table(headers, rows, title) + note


def function_table(h: Hierarchy, name: str, top_n: int) -> str:
    """A timer's merged subtree over all its callers, with the callers."""
    subtree, callers = h.merged_subtree(name)
    root = subtree.get((name,), PathStats())
    lines = [
        f"{name}: {root.inclusive:.1f} ms in {root.calls} calls "
        f"({h.percent(root.inclusive):.1f}% of step time)"
    ]
    caller_rows = [
        [
            caller,
            s.calls,
            _fmt(s.inclusive),
            _pct(100 * s.inclusive / root.inclusive if root.inclusive else 0),
        ]
        for caller, s in sorted(callers.items(), key=lambda c: -c[1].inclusive)
    ]
    lines.append(
        create_ascii_table(
            ["Called from", "Calls", "Time (ms)", f"% of {name}"],
            caller_rows,
            f"CALLERS OF {name}",
        )
    )
    children: dict[Path, list[Path]] = defaultdict(list)
    for path in subtree:
        if len(path) > 1:
            children[path[:-1]].append(path)
    for kids in children.values():
        kids.sort(key=lambda p: -subtree[p].inclusive)
    rows: list[list[object]] = []
    _tree_rows(h, (name,), root.inclusive, 0, top_n, subtree, children, rows)
    if rows:
        lines.append(
            create_ascii_table(
                _TREE_HEADERS, rows, f"INSIDE {name} (all callers merged)"
            )
        )
    return "\n".join(lines)


def rank_imbalance_table(hierarchies: dict[int, Hierarchy], top_n: int) -> str:
    """Exclusive time per timer across ranks: min, mean, max, imbalance."""
    per_key: dict[str, dict[int, float]] = defaultdict(dict)
    for rank, h in hierarchies.items():
        for key, s, _ in h.flat():
            per_key[key][rank] = s.exclusive
    ranks = sorted(hierarchies)
    rows = []
    for key, values in per_key.items():
        series = [values.get(r, 0.0) for r in ranks]
        mean = sum(series) / len(series)
        rows.append(
            (
                mean,
                [
                    key,
                    _fmt(min(series)),
                    _fmt(mean),
                    _fmt(max(series)),
                    f"{max(series) / mean:.2f}" if mean > 0 else "-",
                    f"{ranks[series.index(max(series))]}",
                ],
            )
        )
    rows.sort(key=lambda r: -r[0])
    return create_ascii_table(
        [
            "Timer (exclusive)",
            "Min (ms)",
            "Mean (ms)",
            "Max (ms)",
            "Max/mean",
            "Slowest rank",
        ],
        [r for _, r in rows[:top_n]],
        f"RANK IMBALANCE ({len(ranks)} ranks)",
    )


def coverage_table(trees: list[StepTree]) -> str:
    """Per-step wall-clock time, timed time and coverage."""
    rows = []
    for tree in trees:
        wall = tree.step.wallclock_ms or 0.0
        top = max(tree.roots, key=lambda n: n.inclusive, default=None)
        rows.append(
            [
                tree.step.number,
                _fmt(wall),
                _fmt(tree.timed),
                _pct(tree.coverage or 0.0),
                _fmt(wall - tree.timed),
                top.record.key if top else "",
            ]
        )
    return create_ascii_table(
        [
            "Step",
            "Wall-clock (ms)",
            "Timed (ms)",
            "Coverage",
            "Untimed (ms)",
            "Largest timer",
        ],
        rows,
        f"PER-STEP COVERAGE (rank {trees[0].rank})" if trees else "",
    )
