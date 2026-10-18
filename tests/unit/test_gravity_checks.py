"""Tests for the gravity checks analysis mode."""

from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from swiftsim_cli.modes.analyse import gravity_checks
from swiftsim_cli.modes.analyse.gravity_checks import (
    _find_counterpart_file,
    _load_force_pair,
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
    np.savetxt(path, data, header="Gravity accuracy test")


class TestFindCounterpartFile:
    """Locating the exact/SWIFT counterpart of a check file."""

    def test_handles_steps_above_9999(self, tmp_path):
        """SWIFT's %.4d padding grows past four digits."""
        exact = tmp_path / "gravity_checks_exact_step12345.dat"
        swift = tmp_path / "gravity_checks_swift_step12345_order4.dat"
        exact.touch()
        swift.touch()

        assert _find_counterpart_file(str(exact)) == (str(exact), str(swift))
        assert _find_counterpart_file(str(swift)) == (str(exact), str(swift))

    def test_keeps_zero_padding(self, tmp_path):
        """The matched step text, padding included, builds the name."""
        exact = tmp_path / "gravity_checks_exact_periodic_step0007.dat"
        swift = tmp_path / "gravity_checks_swift_step0007_order3.dat"
        exact.touch()
        swift.touch()

        assert _find_counterpart_file(str(swift)) == (str(exact), str(swift))

    def test_multiple_orders_pick_deterministically(self, tmp_path, capsys):
        """Several multipole orders: sorted choice plus a warning."""
        exact = tmp_path / "gravity_checks_exact_step0001.dat"
        exact.touch()
        names = [
            str(tmp_path / f"gravity_checks_swift_step0001_order{n}.dat")
            for n in (5, 3)
        ]

        with patch.object(gravity_checks.glob, "glob", return_value=names):
            _, swift_file = _find_counterpart_file(str(exact))

        assert swift_file.endswith("order3.dat")
        assert "Warning: 2 SWIFT files" in capsys.readouterr().out

    def test_missing_counterpart_raises(self, tmp_path):
        """No SWIFT file for an exact file is an error."""
        exact = tmp_path / "gravity_checks_exact_step0001.dat"
        exact.touch()

        with pytest.raises(FileNotFoundError):
            _find_counterpart_file(str(exact))


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
        assert "particle IDs differ" in capsys.readouterr().out

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
    assert "median: 0.00000" in out
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
