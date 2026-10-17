"""Tests for the shared SWIFT step-table parser."""

import pytest

from swiftsim_cli.utilities.step_table import (
    DEFAULT_STEP_COLUMNS,
    StepTableReader,
    parse_step_header,
    parse_step_row,
    step_wallclock_ms,
)

HEADER_15 = (
    "#   Step           Time Scale-factor     Redshift      Time-step "
    "Time-bins      Updates    g-Updates    s-Updates sink-Updates    "
    "b-Updates  Wall-clock time [ms]  Props    Dead time [ms] "
)
ROW_15 = (
    "       0   1.448448e-04    0.0400000   24.0000000   0.000000e+00    "
    "1   56            0    286773236            0            0            "
    "0            263601.156    265          4652.056"
)
# Older SWIFT versions have no sink-Updates or b-Updates columns.
HEADER_13 = (
    "#   Step           Time Scale-factor     Redshift      Time-step "
    "Time-bins      Updates    g-Updates    s-Updates  Wall-clock time [ms]"
    "  Props    Dead time [ms]"
)
ROW_13 = (
    "      12   1.448448e-04    0.0400000   24.0000000   1.000000e-06    "
    "40   41          100          200            0              1234.5    "
    "1           5.25"
)


def test_header_with_sink_and_b_updates():
    """The modern header gives the 15-column default layout."""
    assert parse_step_header(HEADER_15) == DEFAULT_STEP_COLUMNS


def test_header_without_sink_and_b_updates():
    """An older header gives 13 columns with wall-clock 11th."""
    columns = parse_step_header(HEADER_13)

    assert columns is not None
    assert len(columns) == 13
    assert columns[10] == "Wall-clock time [ms]"
    assert columns[-1] == "Dead time [ms]"


def test_header_with_rank_prefix():
    """MPI rank prefixes are ignored on header lines."""
    assert parse_step_header("[0000] " + HEADER_15) == DEFAULT_STEP_COLUMNS


@pytest.mark.parametrize(
    "line",
    [
        "# Step Properties: Rebuild=1, Redistribute=2, Repartition=4",
        "#   Time-step  Step",
        "   Step  Time  Scale-factor",
        "[00035.2] engine_step: Writing step info to files took 0.031 ms",
        "",
    ],
)
def test_non_header_lines(line):
    """Other comment and log lines are not mistaken for headers."""
    assert parse_step_header(line) is None


def test_row_15_columns_wallclock():
    """The wall-clock column is read from a modern row by name."""
    row = parse_step_row(ROW_15)

    assert row is not None
    assert row["Step"] == 0
    assert row["Scale-factor"] == pytest.approx(0.04)
    assert step_wallclock_ms(row) == pytest.approx(263601.156)
    assert row["Dead time [ms]"] == pytest.approx(4652.056)


def test_row_13_columns_wallclock():
    """The wall-clock column of an older row comes from its header."""
    columns = parse_step_header(HEADER_13)
    assert columns is not None

    row = parse_step_row(ROW_13, columns)

    assert row is not None
    assert row["Step"] == 12
    assert step_wallclock_ms(row) == pytest.approx(1234.5)
    assert row["Dead time [ms]"] == pytest.approx(5.25)


def test_row_layout_mismatch_is_rejected():
    """Rows whose length does not match the layout do not parse."""
    assert parse_step_row(ROW_13) is None


@pytest.mark.parametrize(
    "line",
    [
        "12.345 unrelated header line",
        "[00035.2] engine_drift_all: took 163.988 ms.",
        "[0000] [00035.2] engine_drift_all: took 163.988 ms.",
        "",
        ROW_15.replace("263601.156", "nan_value"),
        ROW_15.replace("       0 ", "     0.5 ", 1),
    ],
)
def test_non_row_lines(line):
    """Log lines and malformed rows are not parsed as step rows."""
    assert parse_step_row(line) is None


def test_row_with_rank_prefix():
    """MPI rank prefixes are ignored on data rows."""
    row = parse_step_row("[0003] " + ROW_15)

    assert row is not None
    assert step_wallclock_ms(row) == pytest.approx(263601.156)


def test_wallclock_missing_from_layout():
    """Tables without a wall-clock column give None."""
    row = parse_step_row("  3 0.5", ("Step", "Time"))

    assert row is not None
    assert step_wallclock_ms(row) is None


def test_reader_follows_header():
    """The reader switches layout when it sees a header."""
    reader = StepTableReader()

    assert reader.feed(ROW_13) is None
    assert not reader.header_seen
    assert reader.feed(HEADER_13) is None
    assert reader.header_seen

    row = reader.feed(ROW_13)
    assert row is not None
    assert step_wallclock_ms(row) == pytest.approx(1234.5)
    assert reader.feed(ROW_15) is None


def test_reader_defaults_to_modern_layout():
    """Rows before any header use the 15-column layout."""
    row = StepTableReader().feed(ROW_15)

    assert row is not None
    assert step_wallclock_ms(row) == pytest.approx(263601.156)
