"""Parser for SWIFT task-debug ``thread_info`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .task_plot_metadata import (
    subtype_name,
    task_colour,
    task_label,
    task_type_name,
)


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
    c_type: int | None = None
    ci_type: int | None = None
    cj_type: int | None = None
    c_subtype: int | None = None
    ci_subtype: int | None = None
    cj_subtype: int | None = None
    depth: int | None = None


# Column counts written by SWIFT's task_dump_all (SWIFT_DEBUG_TASKS).
NON_MPI_COLUMNS = 19
MPI_COLUMNS = 21


class TaskDebugData:
    """Parsed task-debug data for one ``thread_info`` file."""

    def __init__(self, file_path: str | Path):
        """Read and parse a SWIFT task-debug file."""
        self.file_path = Path(file_path)
        print(f"Loading task-debug data from {self.file_path}...", flush=True)
        self.data = np.loadtxt(self.file_path)
        if self.data.ndim != 2:
            raise ValueError(
                "Task-debug file "
                f"{self.file_path} does not contain tabular data"
            )

        ncols = self.data.shape[1]
        if ncols not in (NON_MPI_COLUMNS, MPI_COLUMNS):
            raise ValueError(
                f"Task-debug file {self.file_path} has {ncols} columns; "
                f"expected {NON_MPI_COLUMNS} (non-MPI thread_info) or "
                f"{MPI_COLUMNS} (thread_info_MPI)."
            )
        self.is_mpi = ncols == MPI_COLUMNS
        self._columns = self._define_columns()
        self.cpu_clock_khz = (
            float(self.data[0, self._columns["cpu_clock"]]) / 1000.0
        )
        self.cpu_clock_khz_by_rank: dict[int, float] = {}
        self._records_by_rank = self._parse_records()
        self.ranks = sorted(self._records_by_rank)
        self.records = [
            record
            for rank in self.ranks
            for record in self._rank_records(rank)
        ]
        self.thread_counts = {
            rank: len({record.thread for record in self._rank_records(rank)})
            for rank in self.ranks
        }

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

    def _parse_records(self) -> dict[int, list[TaskRecord]]:
        """Convert rows into per-rank task records with contiguous threads.

        SWIFT writes one header row per rank (task type ``none`` in MPI
        dumps, ``-1`` in non-MPI dumps) holding the step tic/toc and the
        CPU frequency. Header rows are dropped and each rank's ticks are
        converted with that rank's own CPU frequency.
        """
        rows = self.data
        task_index = self._columns["task"]
        tic_index = self._columns["tic"]
        toc_index = self._columns["toc"]
        clock_index = self._columns["cpu_clock"]

        if self.is_mpi:
            raw_ranks = rows[:, self._columns["rank"]].astype(int)
        else:
            raw_ranks = np.zeros(len(rows), dtype=int)

        header_mask = rows[:, task_index] <= 0
        for row, rank in zip(rows[header_mask], raw_ranks[header_mask]):
            self.cpu_clock_khz_by_rank.setdefault(
                int(rank), float(row[clock_index]) / 1000.0
            )

        keep = (
            ~header_mask
            & (rows[:, tic_index] != 0)
            & (rows[:, toc_index] != 0)
        )
        rows = rows[keep]
        raw_ranks = raw_ranks[keep]

        thread_index = self._columns["thread"]
        records_by_rank: dict[int, list[TaskRecord]] = {}

        for rank in (int(value) for value in np.unique(raw_ranks)):
            rank_rows = rows[raw_ranks == rank]
            clock_khz = self.cpu_clock_khz_by_rank.setdefault(
                rank, self.cpu_clock_khz
            )
            raw_threads = [int(value) for value in rank_rows[:, thread_index]]
            thread_map = {
                thread_id: mapped_id
                for mapped_id, thread_id in enumerate(sorted(set(raw_threads)))
            }

            rank_records: list[TaskRecord] = []
            for row, raw_thread in tqdm(
                zip(rank_rows, raw_threads),
                total=len(rank_rows),
                desc=f"Parsing task records (rank {rank})",
                unit="task",
            ):
                task_type_index = int(row[task_index])
                subtype_index = int(row[self._columns["subtask"]])
                task_type = task_type_name(task_type_index)
                subtype = subtype_name(subtype_index)

                rank_records.append(
                    TaskRecord(
                        rank=rank,
                        thread=thread_map[raw_thread],
                        task_type=task_type,
                        subtype=subtype,
                        label=task_label(task_type_index, subtype_index),
                        colour=task_colour(task_type, subtype),
                        start_ms=float(row[tic_index]) / clock_khz,
                        end_ms=float(row[toc_index]) / clock_khz,
                        ci_type=int(row[self._columns["ci_type"]]),
                        cj_type=int(row[self._columns["cj_type"]]),
                        ci_subtype=int(row[self._columns["ci_subtype"]]),
                        cj_subtype=int(row[self._columns["cj_subtype"]]),
                        ci_depth=int(row[self._columns["ci_depth"]]),
                        cj_depth=int(row[self._columns["cj_depth"]]),
                    )
                )
            records_by_rank[rank] = rank_records

        return records_by_rank

    def _rank_records(self, rank: int) -> list[TaskRecord]:
        """Return all (unfiltered) task records for one rank."""
        if rank not in self._records_by_rank:
            raise ValueError(f"Rank {rank} not present in {self.file_path}")
        return self._records_by_rank[rank]

    def mintic_to_ms(self, mintic: int, rank: int | None = None) -> float:
        """Convert an absolute tick to milliseconds.

        Args:
            mintic: Absolute tick value.
            rank: Rank whose CPU frequency to use (default: first header).
        """
        if rank is None:
            return float(mintic) / self.cpu_clock_khz
        return float(mintic) / self.cpu_clock_khz_by_rank[rank]

    def rank_bounds_ms(self, rank: int) -> tuple[float, float]:
        """Return the min/max absolute times for a rank."""
        rank_records = self._rank_records(rank)
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
        rank_records = self._rank_records(rank)
        if task_filter is None:
            return list(rank_records)

        return [
            record
            for record in rank_records
            if _matches_filter(record, task_filter)
        ]


def _matches_filter(record: TaskRecord, task_filter: TaskFilter) -> bool:
    """Return whether a task record passes the provided task filter."""
    if task_filter.task is not None and record.label != task_filter.task:
        return False

    if task_filter.ci_type is not None and (
        record.ci_type != task_filter.ci_type
    ):
        return False
    if task_filter.cj_type is not None and (
        record.cj_type != task_filter.cj_type
    ):
        return False
    if task_filter.c_type is not None and task_filter.c_type not in (
        record.ci_type,
        record.cj_type,
    ):
        return False
    if task_filter.ci_subtype is not None and (
        record.ci_subtype != task_filter.ci_subtype
    ):
        return False
    if task_filter.cj_subtype is not None and (
        record.cj_subtype != task_filter.cj_subtype
    ):
        return False
    if task_filter.c_subtype is not None and task_filter.c_subtype not in (
        record.ci_subtype,
        record.cj_subtype,
    ):
        return False

    if task_filter.depth is not None and task_filter.depth not in (
        record.ci_depth,
        record.cj_depth,
    ):
        return False

    return True
