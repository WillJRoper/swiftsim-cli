"""Parser for SWIFT's per-step table.

SWIFT prints one row per step, both to stdout and to ``timesteps.txt``,
below a header line such as::

    #   Step           Time Scale-factor     Redshift      Time-step \
Time-bins      Updates    g-Updates    s-Updates sink-Updates    b-Updates \
 Wall-clock time [ms]  Props    Dead time [ms]

"Time-bins" spans two data columns (the minimum and maximum active bin).
Older SWIFT versions have no sink-Updates or b-Updates columns, so columns
must be located from the header rather than by a fixed index. Rows seen
before any header are parsed with the current layout
(:data:`DEFAULT_STEP_COLUMNS`).

Lines may carry an MPI rank prefix (``[0000] ...``), which is ignored.
"""

import re

WALLCLOCK_COLUMN = "Wall-clock time [ms]"
DEAD_TIME_COLUMN = "Dead time [ms]"
_TIME_BIN_COLUMNS = ("Min time-bin", "Max time-bin")

# The layout of current SWIFT versions, used until a header is seen.
DEFAULT_STEP_COLUMNS: tuple[str, ...] = (
    "Step",
    "Time",
    "Scale-factor",
    "Redshift",
    "Time-step",
    *_TIME_BIN_COLUMNS,
    "Updates",
    "g-Updates",
    "s-Updates",
    "sink-Updates",
    "b-Updates",
    WALLCLOCK_COLUMN,
    "Props",
    DEAD_TIME_COLUMN,
)

_RANK_PREFIX_RE = re.compile(r"\[\d+\]")
# A header label is one token, or a token (optionally followed by "time")
# with a bracketed unit, e.g. "Wall-clock time [ms]".
_HEADER_LABEL_RE = re.compile(r"\S+(?:\s+time)?\s+\[[^\]]*\]|\S+")

StepRow = dict[str, float]


def _strip_rank_prefix(line: str) -> str:
    """Remove a leading MPI rank prefix like ``[0000]``."""
    text = line.lstrip()
    match = _RANK_PREFIX_RE.match(text)
    return text[match.end() :] if match else text


def parse_step_header(line: str) -> tuple[str, ...] | None:
    """Parse a step-table header line into data-column names.

    Args:
        line: A line from a SWIFT log or timesteps file.

    Returns:
        One name per data column (with "Time-bins" expanded into
        "Min time-bin" and "Max time-bin"), or None if the line is not a
        step-table header.
    """
    # Cheap rejection first: this runs on every line of large logs.
    if "Step" not in line:
        return None
    text = _strip_rank_prefix(line).strip()
    if not text.startswith("#"):
        return None

    labels = _HEADER_LABEL_RE.findall(text[1:])
    # Rules out other comment lines such as "# Step Properties: ...".
    if labels[:2] != ["Step", "Time"]:
        return None

    columns: list[str] = []
    for label in labels:
        if label.lower() == "time-bins":
            columns.extend(_TIME_BIN_COLUMNS)
        else:
            columns.append(label)
    return tuple(columns)


def parse_step_row(
    line: str, columns: tuple[str, ...] = DEFAULT_STEP_COLUMNS
) -> StepRow | None:
    """Parse one step-table data row.

    A row must have exactly one numeric value per column and start with an
    integer step number, so log lines that merely begin with a number are
    rejected.

    Args:
        line: A line from a SWIFT log or timesteps file.
        columns: Column names from :func:`parse_step_header`.

    Returns:
        A dict mapping column name to value (in column order), or None if
        the line is not a data row for this layout.
    """
    tokens = line.split()
    if tokens and _RANK_PREFIX_RE.fullmatch(tokens[0]):
        tokens = tokens[1:]
    if len(tokens) != len(columns) or not tokens[0].isdigit():
        return None
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        return None
    return dict(zip(columns, values))


def step_wallclock_ms(row: StepRow) -> float | None:
    """Return a row's wall-clock time in ms, if the table has that column."""
    return row.get(WALLCLOCK_COLUMN)


class StepTableReader:
    """Track the column layout while reading a log line by line.

    Attributes:
        columns: The layout used for data rows; starts as
            :data:`DEFAULT_STEP_COLUMNS` and follows each header seen.
        header_seen: Whether a header line has been read.
    """

    def __init__(self) -> None:
        """Start with the default layout and no header seen."""
        self.columns = DEFAULT_STEP_COLUMNS
        self.header_seen = False

    def feed(self, line: str) -> StepRow | None:
        """Read one line, returning its parsed row if it is a data row.

        Args:
            line: The next line of the log or timesteps file.

        Returns:
            The parsed row, or None for headers and non-table lines.
        """
        header = parse_step_header(line)
        if header is not None:
            self.columns = header
            self.header_seen = True
            return None
        return parse_step_row(line, self.columns)
