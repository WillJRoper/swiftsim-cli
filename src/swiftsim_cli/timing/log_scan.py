"""Single streaming pass over a SWIFT log.

Collects, per step and MPI rank: timer records, the task-category CPU
times printed by ``scheduler_report_task_times`` and the
``engine_print_task_counts`` blocks, plus the step table rows and header
information.

Step assignment: SWIFT prints step N's table row at the *start* of step
N+1, so the records between rows N-1 and N belong to step N. Records
before the first row are initialisation (row 0's wall-clock time is
``engine_init_particles``; a restarted run's first row has a wall-clock
time of 0.000). Records after the last row belong to an incomplete step
with no wall-clock time.

MPI ranks (``-v 2``): step rows are printed by rank 0 only, and other
ranks' lines may reach the log after rank 0's row of the step they belong
to. Rank 0 records are assigned by log position; a record of another rank
is assigned to the earliest step whose closing row was printed at or after
its elapsed time (rank 0's last elapsed time before the row), so a late
line still lands in its own step. Elapsed times are printed to 0.1 s, so
records within 0.1 s of a row go to the step that row closes.

Damaged lines: output of several threads can interleave, appending one
line to another (``... took 280.495 m[25656.3] other: ...``). Such lines
are split at the embedded ``[elapsed]`` stamp. A cut timer line is still
recognised when its text up to a complete time matches a known timer (the
unit is then assumed to be ms). Otherwise, as the displaced rest of a cut
line usually follows on a line of its own (``nch: (tasks) took 3.2 ms.``),
the start is kept and joined with the next unprefixed line that is not a
step row; what still cannot be read is dropped and counted. Step rows are
recovered the same way. A step closed after missing rows is marked
``gap`` (its records span several steps but its wall-clock time is one
step's) and left out of the per-step analysis.
"""

from __future__ import annotations

import re
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from swiftsim_cli.timing.timer_db import (
    UNIT_TO_MS,
    LogicalTimer,
    TimerDB,
    timer_key,
)
from swiftsim_cli.utilities.step_table import (
    StepTableReader,
    step_wallclock_ms,
)

# "[rank] [elapsed] function: body" (the rank only appears in MPI runs).
_PREFIX = re.compile(r"^(?:\[(\d+)\]\s+)?\[([\d.]+)\]\s+(\w+):\s+(.*)$")
# An "[rank] [elapsed]" stamp inside a line: the start of another line.
_STAMP = re.compile(r"(?:\[\d+\]\s+)?\[\d+\.\d\]")
# A time printed in full (SWIFT prints three decimals).
_FULL_TIME = re.compile(r"\.\d{3}$")
# Any timing line, used for lines no known timer matches.
_GENERIC_TOOK = re.compile(
    r"^(?P<label>.*?)\s*took\b\s*:?\s*(?:\([^)]*\)\s*:?\s*)?"
    r"(?P<ms>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>[A-Za-z]+)?"
)
_TASK_CATEGORY = re.compile(r"^\*\*\*\s+([^:]+):\s+([\d.]+)\s+ms")
_REVISION = re.compile(r"Revision:\s*([^,\s]+)")
_REVISION_SHA = re.compile(r"(?:^|-g)([0-9a-f]{7,40})(-dirty)?$")
_MPI_RANKS = re.compile(r"MPI is up and running with\s+(\d+)\s+node\(s\)")

_TASK_HEADER = re.compile(
    r"^System total:\s*(\d+)\s*,\s*no\. cells:\s*(\d+)\s*$"
)
_TASK_TOTAL = re.compile(
    r"^Total\s*=\s*(\d+)\s*\((maximum )?per cell\s*=\s*([0-9.eE+\-]+)\s*\)"
)
_TASK_COUNTS = re.compile(r"^task counts are\s*\[(.*)\]\s*$")


@dataclass(slots=True)
class Record:
    """One timer line of the log.

    Attributes:
        rank: MPI rank that printed it (0 for non-MPI runs).
        elapsed: Elapsed run time printed on the line (seconds).
        ms: Time reported by the timer (milliseconds).
        function: Function that printed it.
        kind: "function" or "operation".
        key: Display key (see :func:`timer_db.timer_key`).
        timer: The matching logical timer, None for unknown timers.
        line: Line number in the log (1-based).
    """

    rank: int
    elapsed: float
    ms: float
    function: str
    kind: str
    key: str
    timer: LogicalTimer | None
    line: int


@dataclass
class TaskCountSnapshot:
    """One ``engine_print_task_counts`` block printed by one rank."""

    rank: int
    line_index: int
    system_total: int | None = None
    num_cells: int | None = None
    total_tasks: int | None = None
    per_cell_avg: float | None = None
    per_cell_max: float | None = None
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class Step:
    """Everything printed during one step (or initialisation).

    Attributes:
        number: Step number (the number of the row that closes it).
        kind: "init", "step", "gap" (closed after missing step rows, so
            its records span several steps) or "incomplete" (no closing
            row).
        wallclock_ms: The row's wall-clock time.
        sim_time: The row's simulation time.
        records: Timer records per rank, in log order.
        task_categories: Task-category CPU time (ms) per rank.
        task_counts: Task-count blocks, in log order.
    """

    number: int
    kind: str = "incomplete"
    wallclock_ms: float | None = None
    sim_time: float | None = None
    records: dict[int, list[Record]] = field(
        default_factory=lambda: defaultdict(list)
    )
    task_categories: dict[int, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    task_counts: list[TaskCountSnapshot] = field(default_factory=list)


@dataclass
class LogScan:
    """The result of scanning one log.

    Attributes:
        steps: Initialisation, steps and any incomplete step, in order.
        revision: Commit hash from the "Revision:" line.
        revision_dirty: Whether the revision was marked dirty.
        n_ranks: MPI rank count (1 for non-MPI runs).
        restart: Whether the run restarted (first row has 0 wall-clock).
        matched: Timer lines matched to a known timer.
        unmatched: Unmatched timer lines by "function: label".
        timer_ranks: Ranks that printed timer lines.
        init_end_s: Rank-0 elapsed time when the first row was printed.
        last_elapsed_s: Last rank-0 elapsed time in the log.
        step_gaps: (previous, next) step numbers of rows out of sequence.
        cut_lines: Lines another line was appended to.
        recovered: Damaged timer lines still matched to a timer.
        recovered_rows: Damaged step rows still read.
        lost: Damaged lines that could not be read (dropped).
    """

    steps: list[Step]
    revision: str | None = None
    revision_dirty: bool = False
    n_ranks: int = 1
    restart: bool = False
    matched: int = 0
    unmatched: Counter[str] = field(default_factory=Counter)
    timer_ranks: set[int] = field(default_factory=set)
    init_end_s: float | None = None
    last_elapsed_s: float = 0.0
    step_gaps: list[tuple[int, int]] = field(default_factory=list)
    cut_lines: int = 0
    recovered: int = 0
    recovered_rows: int = 0
    lost: int = 0

    @property
    def complete_steps(self) -> list[Step]:
        """Steps closed by a row (excluding initialisation)."""
        return [s for s in self.steps if s.kind in ("step", "gap")]

    @property
    def init_step(self) -> Step | None:
        """The initialisation block, if the log has a first row."""
        return next((s for s in self.steps if s.kind == "init"), None)


def parse_revision(text: str) -> tuple[str | None, bool]:
    """Commit hash and dirty flag from a "Revision: ..." line."""
    m = _REVISION.search(text)
    if m is None:
        return None, False
    sha = _REVISION_SHA.search(m.group(1))
    if sha is None:
        return None, False
    return sha.group(1), sha.group(2) is not None


def read_log_revision(
    path: str | Path, max_lines: int = 500
) -> tuple[str | None, bool]:
    """Commit hash a log was written by and whether it was ``-dirty``."""
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for _, line in zip(range(max_lines), handle):
            if "Revision:" in line:
                return parse_revision(line)
    return None, False


def _unknown_record(
    func: str, body: str, rank: int, elapsed: float, line: int
) -> Record | None:
    """Record for a timing line no known timer matches."""
    m = _GENERIC_TOOK.match(body)
    if m is None:
        return None
    label = re.sub(r"\d+(?:\.\d+)?", "#", " ".join(m.group("label").split()))
    func = sys.intern(func)
    unparenthesised = re.sub(r"\([^)]*\)", "", label)
    if unparenthesised.strip() == func:
        label = unparenthesised = ""  # "func() took ...": the whole function
    kind = (
        "operation" if re.search(r"[A-Za-z]", unparenthesised) else "function"
    )
    ms = float(m.group("ms")) * UNIT_TO_MS.get(m.group("unit") or "ms", 1.0)
    key = sys.intern("unknown:" + timer_key(func, kind, label))
    return Record(rank, elapsed, ms, func, kind, key, None, line)


def _add_task_count_line(
    body: str,
    rank: int,
    line: int,
    open_blocks: dict[int, TaskCountSnapshot],
    step: Step,
) -> None:
    """Feed one ``engine_print_task_counts`` line (blocks are per rank)."""
    if m := _TASK_HEADER.match(body):
        open_blocks[rank] = TaskCountSnapshot(
            rank, line, system_total=int(m[1]), num_cells=int(m[2])
        )
        return
    if m := _TASK_TOTAL.match(body):
        block = open_blocks.setdefault(rank, TaskCountSnapshot(rank, line))
        block.total_tasks = int(m[1])
        if m[2]:
            block.per_cell_max = float(m[3])
        else:
            block.per_cell_avg = float(m[3])
        return
    if m := _TASK_COUNTS.match(body):
        block = (
            open_blocks.pop(rank)
            if rank in open_blocks
            else TaskCountSnapshot(rank, line)
        )
        for token in m[1].split():
            name, _, value = token.partition("=")
            if name and value.rstrip(",").isdigit():
                block.counts[name] = int(value.rstrip(","))
        step.task_counts.append(block)


def _match_timer(
    timers: Sequence[LogicalTimer], body: str, cut: bool
) -> tuple[LogicalTimer, re.Match[str]] | None:
    """The timer matching a line body (or, for a cut line, its start)."""
    for timer in timers:
        tm = timer.regex.match(body)
        if tm is not None:
            return timer, tm
    if cut:
        for timer in timers:
            tm = timer.prefix_regex.match(body)
            if tm is not None and (
                tm.end() < len(body) or _FULL_TIME.search(tm["ms"])
            ):
                return timer, tm
    return None


def scan_log(path: str | Path, db: TimerDB | None = None) -> LogScan:
    """Scan a SWIFT log in one pass.

    Args:
        path: The log file.
        db: Timer database used to recognise timer lines; without one,
            timer lines are skipped.

    Returns:
        The scanned log.
    """
    scan = LogScan(steps=[])
    current = Step(number=0)
    table = StepTableReader()
    open_blocks: dict[int, TaskCountSnapshot] = {}
    by_function = db.by_function if db is not None else {}
    last_number: int | None = None
    # Rank-0 elapsed time at each closed row (parallel to scan.steps).
    boundaries: list[float] = []
    # The start of a cut line that could not be read on its own: its rest
    # may follow on a later line of its own.
    head = ""

    def close(row: dict[str, float]) -> None:
        nonlocal current, last_number
        number = int(row["Step"])
        wall = step_wallclock_ms(row)
        current.number = number
        current.wallclock_ms = wall
        current.sim_time = row.get("Time")
        if last_number is None:
            scan.init_end_s = scan.last_elapsed_s
            scan.restart = wall == 0.0
            current.kind = "init" if number == 0 or wall == 0.0 else "step"
        elif number != last_number + 1:
            current.kind = "gap"
            scan.step_gaps.append((last_number, number))
        else:
            current.kind = "step"
        scan.steps.append(current)
        boundaries.append(scan.last_elapsed_s)
        last_number = number
        current = Step(number=number + 1)

    def step_of(rank: int, elapsed: float) -> Step:
        if rank == 0 or not boundaries or elapsed > boundaries[-1]:
            return current
        return scan.steps[bisect_left(boundaries, elapsed)]

    def set_head(line: str) -> None:
        nonlocal head
        scan.lost += bool(head)
        head = line

    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line_no, raw in enumerate(handle, 1):
            rest = raw.rstrip()
            joined = False
            while rest:
                line, rest, was_joined = rest, "", joined
                joined = cut = False
                m = _PREFIX.match(line)
                start = m.start(4) if m else 1
                if line.find("[", start) >= 0 and (
                    stamp := _STAMP.search(line, start)
                ):
                    line, rest, cut = (
                        line[: stamp.start()],
                        line[stamp.start() :],
                        True,
                    )
                    scan.cut_lines += 1
                    if m is not None:
                        m = _PREFIX.match(line)
                if m is None:
                    if scan.revision is None and "Revision:" in line:
                        scan.revision, scan.revision_dirty = parse_revision(
                            line
                        )
                    row = table.feed(line)
                    if row is not None:
                        scan.recovered_rows += cut or was_joined
                        close(row)
                    elif head and not cut and line.strip():
                        # The rest of a cut line, printed on a later line.
                        rest, head, joined = head + line, "", True
                    elif cut and not was_joined:
                        set_head(line)
                    elif was_joined:
                        scan.lost += 1
                    continue

                rank = int(m[1]) if m[1] else 0
                elapsed = float(m[2])
                func, body = m[3], m[4]
                if rank == 0:
                    scan.last_elapsed_s = elapsed
                timers = by_function.get(func, ())
                if cut and not was_joined and (not body or timers):
                    found = _match_timer(timers, body, True) if body else None
                    if found is None:
                        set_head(line)
                        continue
                damaged = cut or was_joined

                if db is not None and "took" in body:
                    record = None
                    found = _match_timer(timers, body, damaged)
                    if found is not None:
                        timer, tm = found
                        unit = tm.groupdict().get("unit") or "ms"
                        record = Record(
                            rank,
                            elapsed,
                            float(tm["ms"]) * UNIT_TO_MS.get(unit, 1.0),
                            timer.function,
                            timer.kind,
                            timer.key(tm),
                            timer,
                            line_no,
                        )
                        scan.matched += 1
                        scan.recovered += damaged
                    elif damaged:
                        scan.lost += 1
                    else:
                        record = _unknown_record(
                            func, body, rank, elapsed, line_no
                        )
                        if record is not None:
                            scan.unmatched[record.key[len("unknown:") :]] += 1
                    if record is not None:
                        step_of(rank, elapsed).records[rank].append(record)
                        scan.timer_ranks.add(rank)
                elif func == "scheduler_report_task_times":
                    if cm := _TASK_CATEGORY.match(body):
                        category = cm[1].strip()
                        if category.lower() != "total":
                            current.task_categories[rank][category] += float(
                                cm[2]
                            )
                elif func == "engine_print_task_counts":
                    _add_task_count_line(
                        body, rank, line_no, open_blocks, current
                    )
                elif func == "main" and (mm := _MPI_RANKS.search(body)):
                    scan.n_ranks = int(mm[1])

    if current.records or current.task_counts or current.task_categories:
        scan.steps.append(current)
    return scan
