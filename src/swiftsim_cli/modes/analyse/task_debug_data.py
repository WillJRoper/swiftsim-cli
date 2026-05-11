"""Parser for SWIFT task-debug ``thread_info`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .task_plot_metadata import SUBTYPES, TASKTYPES, task_colour, task_label


@dataclass(frozen=True)
class TaskRecord:
    """A single task interval from a task-debug file."""

    rank: int
    thread: int
    task_type: str
    subtype: str
    label: str
    colour: str
    start_ms: float
    end_ms: float
    ci_type: int
    cj_type: int
    ci_subtype: int
    cj_subtype: int
    ci_depth: int
    cj_depth: int


@dataclass(frozen=True)
class TaskFilter:
    """Filter options for selecting task-debug intervals."""

    task: str | None = None
    ci_type: int | None = None
    cj_type: int | None = None
    ci_subtype: int | None = None
    cj_subtype: int | None = None
    depth: int | None = None


class TaskDebugData:
    """Parsed task-debug data for one ``thread_info`` file."""

    def __init__(self, file_path: str | Path):
        """Read and parse a SWIFT task-debug file."""
        self.file_path = Path(file_path)
        self.data = np.loadtxt(self.file_path)
        if self.data.ndim != 2:
            raise ValueError(
                "Task-debug file "
                f"{self.file_path} does not contain tabular data"
            )

        self.is_mpi = bool(self.data.shape[1] == 21)
        self._columns = self._define_columns()
        self.cpu_clock_khz = (
            float(self.data[0, self._columns["cpu_clock"]]) / 1000.0
        )
        self.records = self._parse_records()
        self.ranks = sorted({record.rank for record in self.records})
        self.thread_counts = self._count_threads_by_rank()

    def _define_columns(self) -> dict[str, int]:
        """Return the relevant column offsets for this file format."""
        if self.is_mpi:
            return {
                "rank": 0,
                "thread": 1,
                "task": 2,
                "subtask": 3,
                "tic": 5,
                "toc": 6,
                "ci_type": 13,
                "cj_type": 14,
                "ci_subtype": 15,
                "cj_subtype": 16,
                "ci_depth": 17,
                "cj_depth": 18,
                "cpu_clock": 12,
            }
        return {
            "rank": -1,
            "thread": 0,
            "task": 1,
            "subtask": 2,
            "tic": 4,
            "toc": 5,
            "ci_type": 11,
            "cj_type": 12,
            "ci_subtype": 13,
            "cj_subtype": 14,
            "ci_depth": 15,
            "cj_depth": 16,
            "cpu_clock": 10,
        }

    def _parse_records(self) -> list[TaskRecord]:
        """Convert rows into task records with contiguous thread IDs."""
        rows = self.data[1:, :]
        tic_index = self._columns["tic"]
        toc_index = self._columns["toc"]
        rows = rows[(rows[:, tic_index] != 0) & (rows[:, toc_index] != 0)]

        rank_index = self._columns["rank"]
        if self.is_mpi:
            raw_ranks = [int(value) for value in rows[:, rank_index]]
        else:
            raw_ranks = [0 for _ in range(len(rows))]

        thread_index = self._columns["thread"]
        thread_map_by_rank: dict[int, dict[int, int]] = {}
        records: list[TaskRecord] = []

        for rank in sorted(set(raw_ranks)):
            rank_mask = np.asarray(raw_ranks) == rank
            rank_rows = rows[rank_mask]
            raw_threads = [int(value) for value in rank_rows[:, thread_index]]
            unique_threads = sorted(set(raw_threads))
            thread_map_by_rank[rank] = {
                thread_id: mapped_id
                for mapped_id, thread_id in enumerate(unique_threads)
            }

            for row, raw_thread in zip(rank_rows, raw_threads):
                task_type_index = int(row[self._columns["task"]])
                subtype_index = int(row[self._columns["subtask"]])
                task_type = TASKTYPES[task_type_index]
                subtype = SUBTYPES[subtype_index]

                records.append(
                    TaskRecord(
                        rank=rank,
                        thread=thread_map_by_rank[rank][int(raw_thread)],
                        task_type=task_type,
                        subtype=subtype,
                        label=task_label(task_type_index, subtype_index),
                        colour=task_colour(task_type, subtype),
                        start_ms=float(row[tic_index]) / self.cpu_clock_khz,
                        end_ms=float(row[toc_index]) / self.cpu_clock_khz,
                        ci_type=int(row[self._columns["ci_type"]]),
                        cj_type=int(row[self._columns["cj_type"]]),
                        ci_subtype=int(row[self._columns["ci_subtype"]]),
                        cj_subtype=int(row[self._columns["cj_subtype"]]),
                        ci_depth=int(row[self._columns["ci_depth"]]),
                        cj_depth=int(row[self._columns["cj_depth"]]),
                    )
                )

        return records

    def _count_threads_by_rank(self) -> dict[int, int]:
        """Count the number of worker threads present for each rank."""
        counts: dict[int, int] = {}
        for rank in self.ranks:
            threads = {
                record.thread for record in self.records if record.rank == rank
            }
            counts[rank] = len(threads)
        return counts

    def mintic_to_ms(self, mintic: int) -> float:
        """Convert an absolute tick to milliseconds."""
        return float(mintic) / self.cpu_clock_khz

    def rank_bounds_ms(self, rank: int) -> tuple[float, float]:
        """Return the min/max absolute times for a rank."""
        rank_records = [
            record for record in self.records if record.rank == rank
        ]
        if not rank_records:
            raise ValueError(f"Rank {rank} not present in {self.file_path}")
        return (
            min(record.start_ms for record in rank_records),
            max(record.end_ms for record in rank_records),
        )

    def get_rank_records(
        self,
        rank: int,
        task_filter: TaskFilter | None = None,
    ) -> list[TaskRecord]:
        """Return task records for one rank after applying filters."""
        if rank not in self.ranks:
            raise ValueError(f"Rank {rank} not present in {self.file_path}")

        if task_filter is None:
            task_filter = TaskFilter()

        return [
            record
            for record in self.records
            if record.rank == rank and _matches_filter(record, task_filter)
        ]


def _matches_filter(record: TaskRecord, task_filter: TaskFilter) -> bool:
    """Return whether a task record passes the provided task filter."""
    if task_filter.task is not None and record.label != task_filter.task:
        return False

    if not _matches_symmetric_pair(
        record.ci_type,
        record.cj_type,
        task_filter.ci_type,
        task_filter.cj_type,
    ):
        return False

    if not _matches_symmetric_pair(
        record.ci_subtype,
        record.cj_subtype,
        task_filter.ci_subtype,
        task_filter.cj_subtype,
    ):
        return False

    if task_filter.depth is not None and task_filter.depth not in (
        record.ci_depth,
        record.cj_depth,
    ):
        return False

    return True


def _matches_symmetric_pair(
    left_value: int,
    right_value: int,
    left_filter: int | None,
    right_filter: int | None,
) -> bool:
    """Apply the same symmetric cell-pair logic as the zoom task tools."""
    if left_filter is None and right_filter is None:
        return True

    if left_filter is None or right_filter is None:
        target = left_filter if left_filter is not None else right_filter
        return target in (left_value, right_value)

    return (left_value == left_filter and right_value == right_filter) or (
        left_value == right_filter and right_value == left_filter
    )


def discover_task_ranks(file_path: str | Path) -> NDArray[np.int32]:
    """Return the rank IDs present in a task-debug file."""
    task_data = TaskDebugData(file_path)
    return np.asarray(task_data.ranks, dtype=np.int32)
