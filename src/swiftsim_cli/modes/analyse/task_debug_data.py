"""Parser for SWIFT task-debug ``thread_info`` dumps.

``task_dump_all`` (SWIFT_DEBUG_TASKS) writes one header row per rank
followed by that rank's tasks. The layouts seen in the wild:

* 11 columns: upstream, non-MPI (``thread type subtype self tic toc``
  plus particle counts and ``sid``)
* 13 columns: upstream, MPI (a leading rank and a ``flags`` column)
* 17 columns: older zoom-branch MPI rows (13 + 4 floats, no cell
  metadata) under a 13-column header
* 19 / 21 columns: zoom branch, non-MPI / MPI (base + cell type, cell
  subtype and depth for ci and cj, plus two distances)

The header row is read separately (it may be shorter than the task rows)
and holds the step's tic/toc and the CPU frequency. Columns are read
vectorised with pandas; ticks stay int64 (they exceed 2^53).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

from swiftsim_cli.utilities.messages import reading, warn

from .task_names import TaskNames, resolve_task_names

if TYPE_CHECKING:
    import pandas as pd

CELL_FIELDS = (
    "ci_type",
    "cj_type",
    "ci_subtype",
    "cj_subtype",
    "ci_depth",
    "cj_depth",
)


@dataclass(frozen=True)
class TaskFilter:
    """Filter options for selecting task-debug intervals.

    Cell (sub)types are names or integers; names are resolved against the
    dump's tables when the filter is applied.
    """

    tasks: tuple[str, ...] | None = None
    c_type: int | str | None = None
    ci_type: int | str | None = None
    cj_type: int | str | None = None
    c_subtype: int | str | None = None
    ci_subtype: int | str | None = None
    cj_subtype: int | str | None = None
    depth: int | None = None

    @property
    def uses_cells(self) -> bool:
        """Whether any cell metadata filter is set."""
        return any(
            value is not None
            for value in (
                self.c_type,
                self.ci_type,
                self.cj_type,
                self.c_subtype,
                self.ci_subtype,
                self.cj_subtype,
                self.depth,
            )
        )


@dataclass(frozen=True)
class DumpLayout:
    """Column layout of a task dump."""

    is_mpi: bool
    ncols: int
    header_cols: int

    @property
    def base(self) -> int:
        """Columns of the upstream layout (11 non-MPI, 13 MPI)."""
        return 13 if self.is_mpi else 11

    @property
    def has_cells(self) -> bool:
        """Whether rows carry cell type/subtype/depth columns."""
        return self.ncols == self.base + 8

    def column(self, name: str) -> int:
        """Index of a named column."""
        offset = 1 if self.is_mpi else 0
        fixed = {"thread": 0, "type": 1, "subtype": 2, "tic": 4, "toc": 5}
        if name == "rank":
            return 0
        if name == "cpufreq":
            return self.base - 1
        if name in CELL_FIELDS:
            return self.base + CELL_FIELDS.index(name)
        return fixed[name] + offset


def detect_layout(path: Path) -> DumpLayout:
    """Work out a dump's layout from its header and first task row.

    Raises:
        ValueError: For an empty file or an unknown layout.
    """
    lines: list[list[str]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            if tokens and not tokens[0].startswith("#"):
                lines.append(tokens)
                if len(lines) == 2:
                    break
    if not lines:
        raise ValueError("task-debug file is empty")
    header = lines[0]
    ncols = len(lines[1]) if len(lines) > 1 else len(header)
    if len(header) in (11, 19):
        is_mpi = False
    elif len(header) in (13, 21):
        is_mpi = True
    else:
        # Non-MPI headers start "-2 -1 -1 1"; MPI ones with the rank.
        is_mpi = header[0] != "-2"
    layout = DumpLayout(is_mpi, ncols, len(header))
    if ncols not in (layout.base, layout.base + 4, layout.base + 8):
        raise ValueError(
            f"unrecognised task-debug layout: {len(header)}-column header, "
            f"{ncols}-column rows (expected SWIFT's 11/19 non-MPI or "
            "13/17/21 MPI columns)"
        )
    return layout


class TaskDebugData:
    """Columnar task-debug data for one ``thread_info`` dump.

    Attributes:
        layout: The dump's column layout.
        names: Name tables used to decode indices.
        file_ranks: Every rank with a header in the file.
        ranks: The ranks that were loaded (``ranks`` argument).
        tasks: One row per task of the loaded ranks with columns ``rank,
            thread`` (contiguous per rank), ``type, subtype, label``,
            ``start_ms, end_ms`` (relative to the rank's step start) and
            the cell columns (-1 when the dump has none).
        thread_counts: Threads per loaded rank.
    """

    def __init__(
        self,
        file_path: str | Path,
        ranks: Iterable[int] | None = None,
        names: TaskNames | None = None,
        swift_src: Path | None = None,
    ):
        """Read a dump, keeping only ``ranks`` (default: all)."""
        import pandas as pd

        self.file_path = Path(file_path)
        print(f"Loading task-debug data from {self.file_path}...", flush=True)
        with reading(self.file_path):
            self.layout = detect_layout(self.file_path)
            frame = self._read()

        layout = self.layout
        nrows = len(frame)

        def take(name: str) -> np.ndarray:
            # pop frees each column as it is converted
            return frame.pop(layout.column(name)).to_numpy()

        rank_col = (
            take("rank").astype(np.int32)
            if layout.is_mpi
            else np.zeros(nrows, dtype=np.int32)
        )
        types = take("type").astype(np.int32)
        tic = take("tic").astype(np.int64)
        toc = take("toc").astype(np.int64)
        cpufreq = take("cpufreq")

        # Header rows: task type -1 (non-MPI) or 0 (MPI).
        header = types <= 0
        self.step_ticks: dict[int, tuple[int, int]] = {}
        self.cpu_clock_khz_by_rank: dict[int, float] = {}
        for rank, t0, t1, freq in zip(
            rank_col[header], tic[header], toc[header], cpufreq[header]
        ):
            self.step_ticks.setdefault(int(rank), (int(t0), int(t1)))
            self.cpu_clock_khz_by_rank.setdefault(
                int(rank), float(freq) / 1000.0
            )
        if not self.step_ticks:
            raise ValueError(f"{self.file_path}: no header row found")
        self.file_ranks = sorted(self.step_ticks)

        selected = self.file_ranks if ranks is None else sorted(set(ranks))
        absent = [r for r in selected if r not in self.step_ticks]
        if absent:
            raise ValueError(
                f"rank {absent[0]} not present in {self.file_path} "
                f"(ranks: {_rank_list(self.file_ranks)})"
            )
        self.ranks = selected

        keep = ~header & (tic != 0) & (toc != 0)
        if ranks is not None:
            keep &= np.isin(rank_col, selected)
        rank_col, tic, toc = rank_col[keep], tic[keep], toc[keep]
        task_type = types[keep].astype(np.int16)
        subtype = take("subtype")[keep].astype(np.int16)
        thread = _contiguous_threads(rank_col, take("thread")[keep])
        cells = {
            field: (
                take(field)[keep].astype(np.int16)
                if layout.has_cells
                else np.full(rank_col.size, -1, dtype=np.int16)
            )
            for field in CELL_FIELDS
        }
        del types, header, keep
        if names is None:
            max_cell = int(
                max(
                    cells["ci_type"].max(initial=0),
                    cells["cj_type"].max(initial=0),
                )
            )
            names = resolve_task_names(swift_src, layout.has_cells, max_cell)
        self.names = names

        # Subtract in int64 first: ticks exceed float precision.
        start = np.empty(rank_col.size)
        end = np.empty(rank_col.size)
        for rank in selected:
            mask = rank_col == rank
            origin = self.step_ticks[rank][0]
            khz = self.cpu_clock_khz_by_rank[rank]
            start[mask] = (tic[mask] - origin) / khz
            end[mask] = (toc[mask] - origin) / khz
        del tic, toc

        self.tasks = pd.DataFrame(
            {
                "rank": rank_col,
                "thread": thread,
                "type": task_type,
                "subtype": subtype,
                "label": self._labels(task_type, subtype),
                "start_ms": start,
                "end_ms": end,
                **cells,
            },
            copy=False,
        )
        self._warn_out_of_range(task_type, subtype, cells)
        counts = self.tasks.groupby("rank")["thread"].max() + 1
        self.thread_counts = {
            rank: int(counts.get(rank, 0)) for rank in selected
        }

    def _read(self) -> pd.DataFrame:
        """Read the needed columns of every row (header rows included)."""
        import pandas as pd

        layout = self.layout
        fields = ["thread", "type", "subtype", "tic", "toc", "cpufreq"]
        if layout.is_mpi:
            fields.append("rank")
        if layout.has_cells:
            fields.extend(CELL_FIELDS)
        usecols = sorted({layout.column(f) for f in fields})
        # Columns missing from a short header row read as NaN, so only
        # columns the header has are read as int64.
        wide = {layout.column(f) for f in ("tic", "toc", "cpufreq")}
        dtype = {
            col: (
                (np.int64 if col in wide else np.int32)
                if col < layout.header_cols
                else np.float64
            )
            for col in usecols
        }
        try:
            return pd.read_csv(
                self.file_path,
                sep=r"\s+",
                header=None,
                names=list(range(layout.ncols)),
                usecols=usecols,
                dtype=dtype,
                comment="#",
                engine="c",
            )
        except (ValueError, pd.errors.ParserError) as error:
            raise ValueError(f"malformed task-debug file: {error}") from error

    def _labels(
        self, task_type: np.ndarray, subtype: np.ndarray
    ) -> pd.Categorical:
        """Vectorised ``type/subtype`` labels as a categorical."""
        import pandas as pd

        pairs = task_type.astype(np.int64) * 1_000_000 + subtype
        unique, inverse = np.unique(pairs, return_inverse=True)
        labels = [
            self.names.label(int(p // 1_000_000), int(p % 1_000_000))
            for p in unique
        ]
        # Labels can repeat for distinct pairs only if names repeat.
        categories = list(dict.fromkeys(labels))
        codes = np.array([categories.index(lbl) for lbl in labels])[inverse]
        return pd.Categorical.from_codes(codes, categories=categories)

    def _warn_out_of_range(
        self,
        task_type: np.ndarray,
        subtype: np.ndarray,
        cells: dict[str, np.ndarray],
    ) -> None:
        """Warn about indices the name tables do not cover."""
        names = self.names
        checks = [
            ("task type", task_type, len(names.task_types)),
            ("task subtype", subtype, len(names.subtypes)),
            (
                "cell type",
                np.concatenate([cells["ci_type"], cells["cj_type"]]),
                len(names.cell_types),
            ),
            (
                "cell subtype",
                np.concatenate([cells["ci_subtype"], cells["cj_subtype"]]),
                len(names.cell_subtypes),
            ),
        ]
        for kind, values, size in checks:
            bad = np.unique(values[values >= size])
            if bad.size:
                warn(
                    f"{kind} indices {', '.join(map(str, bad))} are outside "
                    f"the {size}-name table ({names.source}); shown as "
                    "unknown_<index>. The dump may come from another SWIFT "
                    "version (pass --swift-src)."
                )

    def step_duration_ms(self, rank: int) -> float:
        """Step length from the rank's header tic/toc."""
        tic, toc = self.step_ticks[rank]
        return (toc - tic) / self.cpu_clock_khz_by_rank[rank]

    def tick_to_ms(self, tick: int, rank: int) -> float:
        """Absolute tick to ms relative to the rank's step start."""
        return (tick - self.step_ticks[rank][0]) / self.cpu_clock_khz_by_rank[
            rank
        ]

    def rank_tasks(
        self, rank: int, task_filter: TaskFilter | None = None
    ) -> pd.DataFrame:
        """Tasks of one loaded rank after applying a filter."""
        if rank not in self.ranks:
            raise ValueError(
                f"rank {rank} was not loaded from {self.file_path}"
            )
        tasks = self.tasks
        mask = (tasks["rank"] == rank).to_numpy()
        if task_filter is not None:
            mask &= self.filter_mask(task_filter)
        return tasks.loc[mask]

    def filter_mask(self, task_filter: TaskFilter) -> np.ndarray:
        """Boolean mask of ``tasks`` rows passing a filter.

        Raises:
            ValueError: For unknown task labels or cell names, or a cell
                filter on a dump without cell columns.
        """
        tasks = self.tasks
        mask = np.ones(len(tasks), dtype=bool)
        if task_filter.tasks is not None:
            self.names.validate_labels(task_filter.tasks)
            mask &= tasks["label"].isin(task_filter.tasks).to_numpy()
        if task_filter.uses_cells and not self.layout.has_cells:
            raise ValueError(
                f"{self.file_path} has no cell metadata "
                f"({self.layout.ncols}-column dump); cell type, subtype and "
                "depth filters need a zoom-branch dump (19/21 columns)"
            )

        def col(name: str) -> np.ndarray:
            return tasks[name].to_numpy()

        for value, fields, subtype in (
            (task_filter.ci_type, ("ci_type",), False),
            (task_filter.cj_type, ("cj_type",), False),
            (task_filter.c_type, ("ci_type", "cj_type"), False),
            (task_filter.ci_subtype, ("ci_subtype",), True),
            (task_filter.cj_subtype, ("cj_subtype",), True),
            (task_filter.c_subtype, ("ci_subtype", "cj_subtype"), True),
        ):
            if value is None:
                continue
            index = self.names.cell_index(value, subtype)
            mask &= np.logical_or.reduce([col(f) == index for f in fields])
        if task_filter.depth is not None:
            mask &= (col("ci_depth") == task_filter.depth) | (
                col("cj_depth") == task_filter.depth
            )
        return mask


def _contiguous_threads(ranks: np.ndarray, threads: np.ndarray) -> np.ndarray:
    """Renumber thread ids 0..n-1 within each rank (sorted order kept)."""
    out = np.empty(threads.shape, dtype=np.int32)
    for rank in np.unique(ranks):
        mask = ranks == rank
        out[mask] = np.unique(threads[mask], return_inverse=True)[1]
    return out


def _rank_list(ranks: list[int]) -> str:
    """Compact rank list, e.g. ``0-15``."""
    if ranks == list(range(ranks[0], ranks[-1] + 1)) and len(ranks) > 2:
        return f"{ranks[0]}-{ranks[-1]}"
    return ", ".join(map(str, ranks))
