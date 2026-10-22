"""Tests for the gravity checks analysis mode."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from swiftsim_cli.modes.analyse.gravity_checks import (
    ForcePair,
    _find_counterpart_files,
    _load_force_pair,
    _relative_errors,
    _style,
    analyse_force_checks,
)


def write_force_file(path: Path, ids, accel_scale: float = 1.0) -> None:
    """Write a minimal gravity check file: id, pos, accel, potential."""
    rng = np.random.default_rng(int(sum(ids)))
    rows = []
    for pid in ids:
        pos = [pid % 7, pid % 5, pid % 3]
        accel = [accel_scale * (pid + 1), 2.0 * (pid + 1), 3.0 * (pid + 1)]
        rows.append([pid, *pos, *accel, -float(pid + 1)])
    data = np.asarray(rows, dtype=float)
    rng.shuffle(data)
    write_rows(path, data[:, 0].astype(np.int64), data[:, 1:])


def write_rows(path: Path, ids, values) -> None:
    """Write check rows as SWIFT does: integer IDs, then 7 floats."""
    with open(path, "w") as f:
        f.write("# Gravity accuracy test\n")
        for pid, row in zip(ids, np.asarray(values, dtype=float)):
            f.write(f"{int(pid):18d} " + " ".join(f"{v:.8e}" for v in row))
            f.write("\n")


class TestFindCounterpartFile:
    """Locating the exact/SWIFT counterpart of a check file."""

    def test_handles_steps_above_9999(self, tmp_path):
        """SWIFT's %.4d padding grows past four digits."""
        exact = tmp_path / "gravity_checks_exact_step12345.dat"
        swift = tmp_path / "gravity_checks_swift_step12345_order4.dat"
        exact.touch()
        swift.touch()

        assert _find_counterpart_files(str(exact)) == [
            (str(exact), str(swift))
        ]
        assert _find_counterpart_files(str(swift)) == [
            (str(exact), str(swift))
        ]

    def test_keeps_zero_padding(self, tmp_path):
        """The matched step text, padding included, builds the name."""
        exact = tmp_path / "gravity_checks_exact_periodic_step0007.dat"
        swift = tmp_path / "gravity_checks_swift_step0007_order3.dat"
        exact.touch()
        swift.touch()

        assert _find_counterpart_files(str(swift)) == [
            (str(exact), str(swift))
        ]

    def test_multiple_orders_pick_lowest_or_all(self, tmp_path, capsys):
        """Several orders: the lowest (numerically) unless all_orders."""
        exact = tmp_path / "gravity_checks_exact_step1000.dat"
        exact.touch()
        for n in (10, 5, 3):
            (tmp_path / f"gravity_checks_swift_step1000_order{n}.dat").touch()
        # Another step whose name starts with this one's must not match
        (tmp_path / "gravity_checks_swift_step10000_order2.dat").touch()

        [(_, swift_file)] = _find_counterpart_files(str(exact))
        assert swift_file.endswith("step1000_order3.dat")
        assert "3 SWIFT files" in capsys.readouterr().err

        pairs = _find_counterpart_files(str(exact), all_orders=True)
        assert [Path(s).name[-10:] for _, s in pairs] == [
            "order3.dat",
            "order5.dat",
            "rder10.dat",
        ]

    def test_swift_file_with_exact_in_name(self, tmp_path):
        """Only the gravity_checks_exact prefix marks an exact file."""
        exact = tmp_path / "gravity_checks_exact_step0002.dat"
        swift = tmp_path / "gravity_checks_swift_step0002_inexact.dat"
        exact.touch()
        swift.touch()

        assert _find_counterpart_files(str(swift)) == [
            (str(exact), str(swift))
        ]

    def test_missing_counterpart_raises(self, tmp_path):
        """No SWIFT file for an exact file is an error."""
        exact = tmp_path / "gravity_checks_exact_step0001.dat"
        exact.touch()

        with pytest.raises(FileNotFoundError):
            _find_counterpart_files(str(exact))


class TestLoadForcePair:
    """Shared exact/SWIFT loader."""

    def test_aligns_mismatched_ids(self, tmp_path, capsys):
        """Only common IDs are kept, in matching order."""
        exact = tmp_path / "exact.dat"
        swift = tmp_path / "swift.dat"
        write_force_file(exact, [1, 2, 3, 4])
        write_force_file(swift, [9, 3, 1, 4])

        pos, exact_a, swift_a, exact_pot, swift_pot = _load_force_pair(
            str(exact), str(swift)
        )

        assert len(pos) == 3
        np.testing.assert_array_equal(exact_a, swift_a)
        np.testing.assert_array_equal(exact_pot, swift_pot)
        np.testing.assert_array_equal(exact_pot, [-2.0, -4.0, -5.0])
        assert "particle IDs differ" in capsys.readouterr().err

    def test_no_common_ids_raises(self, tmp_path):
        """Disjoint files cannot be compared."""
        exact = tmp_path / "exact.dat"
        swift = tmp_path / "swift.dat"
        write_force_file(exact, [1, 2])
        write_force_file(swift, [3, 4])

        with pytest.raises(ValueError, match="No common particle IDs"):
            _load_force_pair(str(exact), str(swift))


def test_analyse_force_checks_with_shuffled_ids(tmp_path, capsys):
    """End to end: misaligned files give zero error and a plot."""
    exact = tmp_path / "gravity_checks_exact_step0001.dat"
    swift = tmp_path / "gravity_checks_swift_step0001_order3.dat"
    write_force_file(exact, list(range(10)))
    write_force_file(swift, list(range(12)))

    analyse_force_checks(
        [str(exact)],
        ["run"],
        output_path=str(tmp_path),
        show_plot=False,
    )

    out = capsys.readouterr().out
    assert "| |δa|/|a_exact|   | 0.000e+00 |" in out
    assert (
        tmp_path / "gravity_checks_analysis" / "gravity_checks.png"
    ).exists()


def test_analyse_force_checks_does_not_leak_rcparams(tmp_path):
    """Plot styling is scoped to the plot, not the whole process."""
    exact = tmp_path / "gravity_checks_exact_step0001.dat"
    swift = tmp_path / "gravity_checks_swift_step0001_order3.dat"
    write_force_file(exact, list(range(10)))
    write_force_file(swift, list(range(10)), accel_scale=1.01)
    # Use a sentinel value so an earlier leak can't mask this one.
    with plt.rc_context({"axes.labelsize": 3, "lines.linewidth": 0.5}):
        analyse_force_checks(
            [str(exact)], ["run"], output_path=str(tmp_path), show_plot=False
        )

        assert plt.rcParams["axes.labelsize"] == 3
        assert plt.rcParams["lines.linewidth"] == 0.5


def test_force_checks_skip_bad_inputs_and_fail_when_none_work(
    tmp_path, capsys
):
    """Bad inputs are warnings; no usable input at all is an error."""
    exact = tmp_path / "gravity_checks_exact_step0001.dat"
    swift = tmp_path / "gravity_checks_swift_step0001_order3.dat"
    write_force_file(exact, list(range(10)))
    write_force_file(swift, list(range(10)))
    lonely = tmp_path / "gravity_checks_exact_step0002.dat"
    write_force_file(lonely, [1, 2])

    analyse_force_checks(
        [str(exact), str(lonely)], ["ok", "bad"], output_path=str(tmp_path)
    )
    assert "warning: skipping" in capsys.readouterr().err

    png = tmp_path / "gravity_checks_analysis" / "gravity_checks.png"
    png.unlink()
    with pytest.raises(ValueError, match="none of the gravity check inputs"):
        analyse_force_checks([str(lonely)], ["bad"], output_path=str(tmp_path))
    assert not png.exists()


def test_force_checks_reject_bad_error_range():
    """--min-error must be below --max-error."""
    with pytest.raises(ValueError, match="--min-error < --max-error"):
        analyse_force_checks(["a"], ["a"], min_error=1.0, max_error=0.1)


def test_unparsable_file_is_named_in_the_error(tmp_path):
    """Parse errors say which input file was bad."""
    exact = tmp_path / "exact.dat"
    swift = tmp_path / "swift.dat"
    exact.write_text("1 2 three\n")
    write_force_file(swift, [1])

    with pytest.raises(ValueError, match="exact.dat: "):
        _load_force_pair(str(exact), str(swift))


def _pair_files(tmp_path, exact_values, swift_values, ids=None, step=1):
    """Write an exact/SWIFT pair from (N, 7) arrays: pos, accel, potential."""
    ids = np.arange(len(exact_values)) if ids is None else ids
    exact = tmp_path / f"gravity_checks_exact_step{step:04d}.dat"
    swift = tmp_path / f"gravity_checks_swift_step{step:04d}_order4.dat"
    write_rows(exact, ids, exact_values)
    write_rows(swift, ids, swift_values)
    return exact, swift


def _values(accel, pot=None):
    """Rows of (pos, accel, potential) for the given accelerations."""
    accel = np.asarray(accel, dtype=float)
    pos = np.tile([1.0, 2.0, 3.0], (len(accel), 1))
    pot = -np.ones(len(accel)) if pot is None else pot
    return np.column_stack([pos, accel, pot])


def test_ids_above_2_53_stay_distinct(tmp_path):
    """IDs are read as int64: 2**53 + 1 and 2**53 + 2 do not collide."""
    ids = np.array([2**53 + 1, 2**53 + 2], dtype=np.int64)
    exact, swift = _pair_files(
        tmp_path,
        _values([[1, 0, 0], [2, 0, 0]]),
        _values([[1, 0, 0], [2, 0, 0]]),
        ids=ids,
    )

    pair = _load_force_pair(str(exact), str(swift))

    np.testing.assert_array_equal(pair.exact_a[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(pair.swift_a, pair.exact_a)


def test_relative_errors_known_values():
    """Component, norm and offset-corrected potential errors."""
    exact_a = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 0]])
    swift_a = np.array([[1.1, 0, 0], [0, 2.0, 0.2], [1.0, 0, 0]])
    exact_pot = np.array([-1.0, -2.0, -3.0])
    # Same potential up to a constant zero-point offset: zero error
    swift_pot = exact_pot + 5.0

    errors = _relative_errors(
        ForcePair(np.zeros((3, 3)), exact_a, swift_a, exact_pot, swift_pot)
    )

    np.testing.assert_allclose(errors["x"][:2], [0.1, 0.0], atol=1e-12)
    np.testing.assert_allclose(errors["z"][:2], [0.0, 0.1], atol=1e-12)
    np.testing.assert_allclose(errors["norm"][:2], [0.1, 0.1], atol=1e-12)
    np.testing.assert_allclose(errors["pot"], 0.0, atol=1e-12)
    # A zero exact acceleration has an undefined (NaN) relative error
    assert np.isnan(errors["norm"][2]) and np.isnan(errors["x"][2])


def test_potential_error_after_offset(tmp_path):
    """A 10% potential difference shows up once the offset is removed."""
    exact_pot = np.array([-1.0, -3.0])
    swift_pot = np.array([-1.1, -2.9])  # mean offset 0: errors 10%, 3.3%
    errors = _relative_errors(
        ForcePair(
            np.zeros((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            exact_pot,
            swift_pot,
        )
    )
    np.testing.assert_allclose(errors["pot"], [0.1, 0.1 / 3])


def test_percentiles_and_out_of_range_counts(tmp_path, capsys):
    """Printed percentiles match known errors; out-of-range is counted."""
    rel = np.arange(1, 101) / 100.0  # errors 0.01 .. 1.00
    exact_a = np.column_stack([np.ones(100), np.zeros(100), np.zeros(100)])
    swift_a = exact_a.copy()
    swift_a[:, 0] += rel
    exact, _ = _pair_files(tmp_path, _values(exact_a), _values(swift_a))

    analyse_force_checks(
        [str(exact)], ["run"], output_path=tmp_path, max_error=0.5
    )

    out = capsys.readouterr().out
    median, p90, p99 = np.percentile(rel, [50, 90, 99])
    row = next(line for line in out.splitlines() if "|δa|/|a_exact| " in line)
    # The label itself contains "|": take the cells after it
    rest = row.split("|a_exact|", 1)[1].strip().strip("|")
    cells = [c.strip() for c in rest.split("|")]
    assert cells[:3] == [f"{median:.3e}", f"{p90:.3e}", f"{p99:.3e}"]
    assert cells[3:5] == ["0.0", "50.0"]  # none below 1e-7, half above 0.5


def test_non_finite_values_are_dropped_with_warning(tmp_path, capsys):
    """One NaN no longer wipes out every statistic."""
    accel = [[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]
    swift = np.array(accel)
    swift[1, 0] = np.nan
    exact, _ = _pair_files(tmp_path, _values(accel), _values(swift))

    analyse_force_checks([str(exact)], ["run"], output_path=tmp_path)

    captured = capsys.readouterr()
    assert "ignoring 1 particle with non-finite values" in captured.err
    assert "nan" not in captured.out.lower()


def test_position_mismatch_warns(tmp_path, capsys):
    """Exact and SWIFT files from different snapshots are flagged."""
    exact_values = _values([[1.0, 0, 0], [2.0, 0, 0]])
    swift_values = exact_values.copy()
    swift_values[0, 0] += 0.5
    exact, swift = _pair_files(tmp_path, exact_values, swift_values)

    _load_force_pair(str(exact), str(swift))

    assert "1 particle sit at different positions" in capsys.readouterr().err


def test_empty_file_is_an_error(tmp_path):
    """A check file with only a header has no data rows."""
    exact = tmp_path / "gravity_checks_exact_step0001.dat"
    swift = tmp_path / "gravity_checks_swift_step0001_order4.dat"
    exact.write_text("# header only\n")
    write_force_file(swift, [1])

    with pytest.raises(ValueError, match="no data rows"):
        _load_force_pair(str(exact), str(swift))


def test_all_orders_compares_each_order(tmp_path, capsys):
    """--all-orders gives one run per multipole order, labelled."""
    exact = tmp_path / "gravity_checks_exact_step0001.dat"
    write_force_file(exact, list(range(5)))
    for order in (4, 5):
        write_force_file(
            tmp_path / f"gravity_checks_swift_step0001_order{order}.dat",
            list(range(5)),
        )

    analyse_force_checks(
        [str(exact)], ["run"], output_path=tmp_path, all_orders=True
    )

    out = capsys.readouterr().out
    assert "run order 4:" in out and "run order 5:" in out


def test_style_never_repeats_for_many_runs():
    """Colour and line style together stay unique past the palette size."""
    styles = {
        (_style(i, 45)["color"], _style(i, 45)["linestyle"]) for i in range(45)
    }
    assert len(styles) == 45
