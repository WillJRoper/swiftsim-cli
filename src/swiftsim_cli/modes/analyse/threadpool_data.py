"""Parser for SWIFT threadpool dump files."""

from __future__ import annotations

from ast import literal_eval
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .task_plot_metadata import COLOURS


@dataclass(frozen=True)
class ThreadpoolRecord:
    """A single threadpool interval."""

    thread: int
    function: str
    chunk: int
    start_ms: float
    end_ms: float
    colour: str
    is_background: bool = False


class ThreadpoolData:
    """Parsed threadpool dump data for one step."""

    def __init__(self, file_path: str | Path):
        """Read and parse a SWIFT threadpool dump file."""
        self.file_path = Path(file_path)
        self.cpu_clock_khz: float | None = None
        self.thread_count = 0
        self.records: list[ThreadpoolRecord] = []
        self._parse()

    def _parse(self) -> None:
        """Parse the threadpool file into structured records."""
        lines = self.file_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            raise ValueError(
                f"Threadpool file {self.file_path} is missing its header"
            )

        header = literal_eval(lines[1][2:].strip())
        self.thread_count = int(header["num_threads"])
        self.cpu_clock_khz = float(header["cpufreq"]) / 1000.0

        parsed_rows: list[tuple[str, int, int, int, int]] = []
        worker_threads: set[int] = set()
        functions: list[str] = []

        for line in lines:
            if not line or line.startswith("#"):
                continue

            columns = line.split()
            if len(columns) < 5:
                raise ValueError(
                    f"Malformed threadpool row in {self.file_path}: {line}"
                )

            function = columns[0].replace("_mapper", "")
            thread = int(columns[1])
            chunk = int(columns[2])
            tic = int(columns[3])
            toc = int(columns[4])
            parsed_rows.append((function, thread, chunk, tic, toc))
            functions.append(function)
            if thread >= 0:
                worker_threads.add(thread)

        thread_map = {
            thread_id: mapped_id
            for mapped_id, thread_id in enumerate(sorted(worker_threads))
        }
        colour_map = _function_colours(functions)

        for function, thread, chunk, tic, toc in parsed_rows:
            self.records.append(
                ThreadpoolRecord(
                    thread=thread_map.get(thread, -1),
                    function=function,
                    chunk=chunk,
                    start_ms=float(tic) / self.cpu_clock_khz,
                    end_ms=float(toc) / self.cpu_clock_khz,
                    colour=colour_map[function],
                    is_background=thread < 0,
                )
            )

        if worker_threads:
            self.thread_count = len(worker_threads)

    def mintic_to_ms(self, mintic: int) -> float:
        """Convert an absolute tick to milliseconds."""
        if self.cpu_clock_khz is None:
            raise ValueError("Threadpool file has not been parsed")
        return float(mintic) / self.cpu_clock_khz

    def bounds_ms(self) -> tuple[float, float]:
        """Return the min/max absolute times for the file."""
        if not self.records:
            raise ValueError(
                f"Threadpool file {self.file_path} contains no data"
            )
        return (
            min(record.start_ms for record in self.records),
            max(record.end_ms for record in self.records),
        )

    @property
    def background_records(self) -> list[ThreadpoolRecord]:
        """Return the background occupancy intervals."""
        return [record for record in self.records if record.is_background]

    @property
    def worker_records(self) -> list[ThreadpoolRecord]:
        """Return intervals associated with concrete worker threads."""
        return [record for record in self.records if not record.is_background]


def _function_colours(functions: list[str]) -> dict[str, str]:
    """Assign stable colours to threadpool function names."""
    ordered_functions = [
        function for function, _ in Counter(functions).most_common()
    ]
    return {
        function: COLOURS[index % len(COLOURS)]
        for index, function in enumerate(ordered_functions)
    }
