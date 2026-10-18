"""Runtime call tree of the timers printed during one step.

A timer line is printed when its timer *ends*, so a timer's children are
printed before it. Records are walked in log order and each record R
claims earlier, still unclaimed records C (most recent first) when:

1. statically R can contain C: R is the function timer of F and C is an
   operation of F or C's function is reachable from F; or R is an
   operation of F and C's function is reachable from a call inside R's
   line range, or C is an operation of F lexically inside R;
2. C ended no earlier than R started (allowing for the 0.1 s resolution
   of the elapsed times printed in the log);
3. claiming C would not make R's children add up to more than R.

Whatever is left unclaimed at the end of the step is a top-level timer
of the step. A function called from several places (e.g. engine_unskip
from engine_rebuild and engine_prepare) is attributed to the caller that
actually contained each call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from swiftsim_cli.timing.log_scan import Record
from swiftsim_cli.timing.timer_db import TimerDB

# Slack (s) on condition 2: elapsed times are printed to 0.1 s.
ELAPSED_TOLERANCE_S = 0.15
# Relative and absolute (ms) slack on condition 3 for rounding.
SUM_TOLERANCE = 0.02
SUM_TOLERANCE_MS = 0.5


@dataclass(eq=False, slots=True)
class Node:
    """A timer record and the records it contains."""

    record: Record
    children: list[Node] = field(default_factory=list)

    @property
    def inclusive(self) -> float:
        """Time reported by the timer (ms)."""
        return self.record.ms

    @property
    def exclusive(self) -> float:
        """Time not covered by the children (ms)."""
        return self.record.ms - sum(c.record.ms for c in self.children)


def can_contain(outer: Record, inner: Record, db: TimerDB) -> bool:
    """Whether the source allows ``inner`` to run inside ``outer``."""
    if outer.kind == "function":
        if inner.function == outer.function:
            return inner.kind == "operation"
        return inner.function in db.reach(outer.function)
    if outer.timer is None:
        return False
    if inner.function == outer.function:
        return (
            inner.kind == "operation"
            and inner.timer is not None
            and db.lexically_inside(inner.timer, outer.timer)
        )
    return inner.function in db.op_reach(outer.timer)


def build_tree(records: list[Record], db: TimerDB) -> list[Node]:
    """Nest one rank's records of one step.

    Args:
        records: The records in log order.
        db: Timer database providing the static rules.

    Returns:
        The top-level nodes, in log order.
    """
    pending: list[Node] = []
    for record in records:
        node = Node(record)
        earliest = record.elapsed - record.ms / 1000.0 - ELAPSED_TOLERANCE_S
        limit = record.ms * (1 + SUM_TOLERANCE) + SUM_TOLERANCE_MS
        claimed: list[int] = []
        total = 0.0
        for i in range(len(pending) - 1, -1, -1):
            candidate = pending[i].record
            if candidate.elapsed < earliest:
                break
            if not can_contain(record, candidate, db):
                continue
            if total + candidate.ms > limit:
                break
            total += candidate.ms
            claimed.append(i)
        if claimed:
            node.children = [pending[i] for i in reversed(claimed)]
            taken = set(claimed)
            pending = [p for i, p in enumerate(pending) if i not in taken]
        pending.append(node)
    return pending
