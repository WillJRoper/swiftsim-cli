"""Per-log data of the scaling analyses.

Each log is scanned once. Timer statistics are exclusive times from the
per-step timer trees of each MPI rank (see ``timing.timing_hierarchy``),
so nested timers are never double counted; runtimes are summed step-table
wall-clock times of the steps analysed (initialisation, the 0.000 row of
a restarted run and any incomplete step excluded).
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from swiftsim_cli.modes.analyse.log_timing import (
    check_scan,
    unmatched_percent,
)
from swiftsim_cli.timing.log_scan import (
    LogScan,
    Step,
    read_log_revision,
    scan_log,
)
from swiftsim_cli.timing.timer_db import resolve_timer_db
from swiftsim_cli.timing.timing_hierarchy import hierarchy_for, select_steps
from swiftsim_cli.utilities.messages import reading

Statistic = Literal["sum", "mean"]
RankSource = Literal["root", "all"]


@dataclass
class LogTimers:
    """Timer statistics of one log.

    Attributes:
        path: The log file.
        label: Its file name.
        n_ranks: MPI rank count of the run (1 for non-MPI runs).
        n_steps: Steps analysed.
        step_time: Summed wall-clock time of those steps (ms).
        exclusive: Exclusive time (ms) per timer, per rank with timers.
        calls: Calls per timer, per rank with timers.
        unmatched: Share of the log's timer lines matching no timer (%).
    """

    path: str
    label: str
    n_ranks: int
    n_steps: int
    step_time: float
    exclusive: dict[int, dict[str, float]] = field(default_factory=dict)
    calls: dict[int, dict[str, int]] = field(default_factory=dict)
    unmatched: float = 0.0

    def _ranks(self, source: RankSource) -> list[int]:
        return [0] if source == "root" else sorted(self.exclusive)

    def series(
        self, statistic: Statistic, source: RankSource
    ) -> dict[str, float]:
        """Value per timer for the chosen view.

        ``sum``: exclusive time over the run, of rank 0 or averaged over
        the ranks with timers (so perfect strong scaling is 1/N either
        way). ``mean``: exclusive time per call.
        """
        ranks = self._ranks(source)
        times: dict[str, float] = {}
        calls: dict[str, int] = {}
        for rank in ranks:
            for key, ms in self.exclusive.get(rank, {}).items():
                times[key] = times.get(key, 0.0) + ms
                calls[key] = calls.get(key, 0) + self.calls[rank][key]
        if statistic == "mean":
            return {k: v / calls[k] for k, v in times.items() if calls[k]}
        return {k: v / max(1, len(ranks)) for k, v in times.items()}

    def total(self, statistic: Statistic, source: RankSource) -> float:
        """The summed (``sum``) or per-call (``mean``) time of all timers."""
        if statistic == "sum":
            return sum(self.series("sum", source).values())
        ranks = self._ranks(source)
        time = sum(sum(self.exclusive.get(r, {}).values()) for r in ranks)
        n = sum(sum(self.calls.get(r, {}).values()) for r in ranks)
        return time / n if n else 0.0

    def percent(self, key: str, source: RankSource) -> float:
        """A timer's exclusive time as a share of the step time (%)."""
        value = self.series("sum", source).get(key, 0.0)
        return 100 * value / self.step_time if self.step_time > 0 else 0.0


@dataclass
class LogRuntime:
    """Runtime of one log: summed wall-clock time of its steps."""

    path: str
    label: str
    n_ranks: int
    n_steps: int
    runtime: float


def resolve_log_files(log_files: list[Path]) -> list[str]:
    """Resolve explicit files and shell-style glob patterns."""
    resolved: list[str] = []
    for log_file in log_files:
        raw = str(log_file)
        if Path(raw).exists():
            resolved.append(raw)
            continue
        matches = sorted(glob.glob(raw))
        if not matches:
            raise ValueError(f"no log file matches '{raw}'")
        resolved.extend(matches)
    return resolved


def _steps(
    scan: LogScan, path: str, step_range: tuple[int, int] | None
) -> list[Step]:
    steps = select_steps(scan.steps, step_range)
    if not steps:
        within = (
            f" within steps {step_range[0]}-{step_range[1]}"
            if step_range
            else ""
        )
        raise ValueError(f"no complete steps{within} in '{path}'")
    return steps


def load_log_timers(
    path: str,
    step_range: tuple[int, int] | None = None,
    swift_src: Path | None = None,
    match_revision: bool = True,
    regenerate_db: bool = False,
) -> LogTimers:
    """Scan one log and collect its exclusive timer statistics."""
    revision, dirty = read_log_revision(path)
    db, status = resolve_timer_db(
        revision, swift_src, match_revision, regenerate_db, dirty
    )
    print(f"  Timer database: {status}")
    with reading(path):
        scan = scan_log(path, db)
    check_scan(scan, Path(path).name)
    steps = _steps(scan, path, step_range)
    data = LogTimers(
        path=path,
        label=Path(path).name,
        n_ranks=scan.n_ranks,
        n_steps=len(steps),
        step_time=sum(s.wallclock_ms or 0.0 for s in steps),
        unmatched=unmatched_percent(scan),
    )
    print(
        f"  {data.n_steps} steps, {len(scan.timer_ranks)}/{scan.n_ranks} "
        f"ranks with timers, {scan.matched} timer lines matched, "
        f"{data.unmatched:.1f}% unmatched"
    )
    for rank in sorted(scan.timer_ranks):
        rows = hierarchy_for(scan.steps, rank, db, step_range).flat()
        data.exclusive[rank] = {key: s.exclusive for key, s, _ in rows}
        data.calls[rank] = {key: s.calls for key, s, _ in rows}
    return data


def load_log_runtime(
    path: str, step_range: tuple[int, int] | None = None
) -> LogRuntime:
    """Scan one log for its rank count and the wall-clock time of its steps.

    Steps closed after missing (damaged) rows still count: their
    wall-clock time is that of one step.
    """
    with reading(path):
        scan = scan_log(path)
    steps = select_steps(scan.steps, step_range, kinds=("step", "gap"))
    if not steps:
        within = (
            f" within requested steps {step_range[0]}-{step_range[1]}"
            if step_range
            else ""
        )
        raise ValueError(
            f"No step-table wallclock rows found in '{path}'{within}"
        )
    return LogRuntime(
        path=path,
        label=Path(path).name,
        n_ranks=scan.n_ranks,
        n_steps=len(steps),
        runtime=sum(s.wallclock_ms or 0.0 for s in steps),
    )


def require_rank_counts(rank_counts: list[int]) -> None:
    """Raise unless the logs cover at least two MPI rank counts."""
    if len(set(rank_counts)) < 2:
        raise ValueError(
            "scaling needs logs of at least two different MPI rank counts "
            f"(got {sorted(rank_counts)}); is each log from a different "
            "run size?"
        )


def fit_slope(rank_counts: list[int], values: list[float]) -> float | None:
    """Slope of log(time) against log(ranks) (-1 is perfect scaling)."""
    points = [(n, v) for n, v in zip(rank_counts, values) if n > 0 and v > 0]
    if len({n for n, _ in points}) < 2:
        return None
    slope, _ = np.polyfit(
        np.log10([n for n, _ in points]), np.log10([v for _, v in points]), 1
    )
    return float(slope)


def describe_slope(slope: float | None) -> str:
    """Short qualitative description of a scaling slope."""
    if slope is None:
        return "insufficient"
    for limit, text in (
        (-1.15, "superlinear"),
        (-0.75, "strong"),
        (-0.25, "moderate"),
        (0.10, "weak"),
    ):
        if slope <= limit:
            return text
    return "regressing"


def format_slope(slope: float | None) -> str:
    """A slope to two decimals, or n/a."""
    return "n/a" if slope is None else f"{slope:.2f}"


def format_time(ms: float) -> str:
    """A time in ms as ms or s."""
    return f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.1f}ms"


def clear_outputs(directory: Path, prefix: str | None, *patterns: str) -> None:
    """Remove outputs of an earlier run matching the patterns."""
    lead = f"{prefix}_" if prefix else ""
    if directory.is_dir():
        for pattern in patterns:
            for path in directory.glob(lead + pattern):
                path.unlink()
