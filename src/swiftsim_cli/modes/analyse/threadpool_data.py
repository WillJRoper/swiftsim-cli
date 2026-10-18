"""Parser for SWIFT threadpool dump files (``-Y`` output).

Each row is ``map_function thread_id chunk_size tic toc``. Thread -1 is
the main (calling) thread, drawn as background occupancy; a function
name of ``(null)`` (dladdr found no symbol) is kept as is.
"""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from swiftsim_cli.utilities.messages import reading

if TYPE_CHECKING:
    import pandas as pd


class ThreadpoolData:
    """Parsed threadpool dump for one step and rank.

    Attributes:
        cpu_clock_khz: CPU frequency from the header, in kHz.
        thread_count: Number of worker threads seen.
        rows: One row per interval: ``function``, ``thread`` (contiguous
            worker ids, -1 for the main thread), ``chunk``, ``tic`` and
            ``toc`` (absolute int64 ticks).
    """

    def __init__(self, file_path: str | Path):
        """Read and parse a SWIFT threadpool dump file."""
        self.file_path = Path(file_path)
        with reading(self.file_path):
            self._parse()

    def _parse(self) -> None:
        """Parse the header and rows."""
        import pandas as pd

        with open(self.file_path, encoding="utf-8") as handle:
            head = [handle.readline() for _ in range(2)]
        try:
            header = literal_eval(head[1].lstrip("#").strip())
            self.cpu_clock_khz = float(header["cpufreq"]) / 1000.0
        except (IndexError, KeyError, SyntaxError, ValueError) as error:
            raise ValueError(
                "missing or malformed threadpool header (expected "
                "\"# {'num_threads': N, 'cpufreq': F}\")"
            ) from error

        try:
            rows = pd.read_csv(
                self.file_path,
                sep=r"\s+",
                comment="#",
                header=None,
                names=["function", "thread", "chunk", "tic", "toc"],
                dtype={
                    "function": str,
                    "thread": np.int64,
                    "chunk": np.int64,
                    "tic": np.int64,
                    "toc": np.int64,
                },
            )
        except pd.errors.EmptyDataError:
            rows = pd.DataFrame(
                columns=["function", "thread", "chunk", "tic", "toc"]
            )
        except (ValueError, pd.errors.ParserError) as error:
            raise ValueError(f"malformed threadpool row: {error}") from error
        if rows.empty:
            raise ValueError("threadpool file contains no data")

        rows["function"] = rows["function"].str.replace(
            "_mapper", "", regex=False
        )
        thread = rows["thread"].to_numpy()
        workers = thread >= 0
        mapped = np.full(len(rows), -1, dtype=np.int64)
        unique, inverse = np.unique(thread[workers], return_inverse=True)
        mapped[workers] = inverse
        rows["thread"] = mapped
        self.thread_count = int(unique.size)
        self.rows: pd.DataFrame = rows

    def tick_range(self) -> tuple[int, int]:
        """First tic and last toc in the file."""
        return int(self.rows["tic"].min()), int(self.rows["toc"].max())

    def to_ms(self, origin_tick: int) -> pd.DataFrame:
        """Rows with ``start_ms``/``end_ms`` relative to ``origin_tick``."""
        rows = self.rows
        return rows.assign(
            start_ms=(rows["tic"].to_numpy() - origin_tick)
            / self.cpu_clock_khz,
            end_ms=(rows["toc"].to_numpy() - origin_tick) / self.cpu_clock_khz,
        )
