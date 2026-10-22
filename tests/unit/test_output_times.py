"""Tests for the output-times mode."""

import argparse

import numpy as np
import pytest

from swiftsim_cli.cli import main
from swiftsim_cli.cosmology import convert_redshift_to_time
from swiftsim_cli.modes.output_times import (
    Converter,
    _merge,
    add_arguments,
    generate_output_list,
    generate_times,
)
from swiftsim_cli.params import load_parameters

COSMO_PARAMS = """\
InternalUnitSystem:
  UnitLength_in_cgs: 3.08567758e24
  UnitVelocity_in_cgs: 1e5
Cosmology:
  h: 0.6777
  Omega_cdm: 0.2587481
  Omega_lambda: 0.693
  Omega_b: 0.0482519
  a_begin: 0.0078125
  a_end: 1.0
"""


def read_list(path):
    """Return (header, values, labels) of a written output list."""
    lines = path.read_text().splitlines()
    rows = [line.split(",") for line in lines[1:]]
    values = np.array([float(r[0]) for r in rows])
    labels = [r[1].strip() for r in rows if len(r) > 1]
    return lines[0], values, labels


@pytest.fixture
def cosmo(tmp_path):
    """Load a cosmological parameter file."""
    path = tmp_path / "params.yml"
    path.write_text(COSMO_PARAMS)
    load_parameters(path)
    return path


def run(tmp_path, **kwargs):
    """Generate a list from keyword arguments; return its path."""
    out = tmp_path / "out.txt"
    generate_output_list({"output_list": out, "force": True, **kwargs})
    return out


class TestGenerateTimes:
    """Native generation."""

    def test_non_divisor_delta_does_not_overshoot(self):
        """The last output never goes past the final one."""
        z = generate_times("z", 3.0, 0.7, 0.0)
        assert z == pytest.approx([3.0, 2.3, 1.6, 0.9, 0.2])
        a = generate_times("scale_factor", 0.1, 0.25, 1.0)
        assert a == pytest.approx([0.1, 0.35, 0.6, 0.85])
        t = generate_times("time", 0.0, 0.3, 1.0)
        assert t == pytest.approx([0.0, 0.3, 0.6, 0.9])

    def test_exact_divisor_includes_final(self):
        """An exact divisor reaches the final output despite rounding."""
        assert generate_times("scale_factor", 0.1, 0.1, 1.0)[-1] == (
            pytest.approx(1.0)
        )
        assert len(generate_times("z", 1.0, 0.1, 0.0)) == 11

    def test_log_scale_factor_is_log10(self):
        """The log spacing is in log10(a)."""
        a = generate_times("log_scale_factor", 0.1, 0.5, 1.0)
        assert a == pytest.approx([0.1, 10**-0.5, 1.0])

    def test_errors(self):
        """Backwards ranges and non-positive spacings are errors."""
        with pytest.raises(ValueError, match="comes before"):
            generate_times("z", 0.0, 0.1, 1.0)
        with pytest.raises(ValueError, match="positive"):
            generate_times("time", 0.0, 0.0, 1.0)


def test_merge_drops_near_duplicates_preferring_snapshots():
    """Outputs within tolerance collapse to one, snapshots win."""
    a, kind = _merge(
        np.array([0.5, 0.25]), np.array([0.25 * (1 + 1e-9), 0.3, 0.5])
    )
    assert a == pytest.approx([0.25, 0.3, 0.5])
    assert list(kind) == ["Snapshot", "Snipshot", "Snapshot"]


class TestCosmological:
    """Lists for a cosmological run."""

    def test_redshift_list(self, tmp_path, cosmo):
        """A non-divisor spacing stops above z = 0, and defaults a_end."""
        header, z, _ = read_list(run(tmp_path, first_snap_z=10.0, delta_z=0.7))
        assert header == "# Redshift"
        assert z[0] == 10.0
        assert z[-1] == pytest.approx(0.2)
        assert np.all(np.diff(z) < 0)

    def test_snipshots_have_no_duplicates(self, tmp_path, cosmo, capsys):
        """Snipshots on snapshot times are dropped (no repeated outputs)."""
        header, a, labels = read_list(
            run(
                tmp_path,
                first_snap_scale_factor=0.1,
                delta_scale_factor=0.1,
                snipshot_delta_scale_factor=0.02,
            )
        )
        assert header == "# Scale Factor, Select Output"
        assert np.all(np.diff(a) > 0)
        assert labels.count("Snapshot") == 10
        assert labels.count("Snipshot") == 46 - 10
        assert "select_output_on" in capsys.readouterr().err

    def test_time_snipshots_merge_in_scale_factor(self, tmp_path, cosmo):
        """Time-spaced snipshots that hit snapshot times are dropped too."""
        _, z, labels = read_list(
            run(
                tmp_path,
                first_snap_z=10.0,
                delta_time=0.5,
                snipshot_delta_time=0.1,
                list_units="redshift",
            )
        )
        assert z[0] == 10.0  # round trip through time is exact
        assert np.all(np.diff(z) < 0)
        assert labels.count("Snapshot") + labels.count("Snipshot") == len(z)

    def test_time_list_is_in_internal_units(self, tmp_path, cosmo):
        """A cosmological time list is written in internal time units."""
        header, t, _ = read_list(
            run(tmp_path, first_snap_z=1.0, delta_time=1.0)
        )
        assert header == "# Time"
        unit_gyr = 3.08567758e19 / (365.25 * 86400 * 1e9)
        age = float(convert_redshift_to_time(1.0))
        assert t[0] == pytest.approx(age / unit_gyr, rel=1e-10)
        assert np.diff(t) == pytest.approx(1.0 / unit_gyr)

    def test_written_with_12_significant_figures(self, tmp_path, cosmo):
        """Values keep 12 significant figures."""
        out = run(
            tmp_path,
            first_snap_scale_factor=0.01,
            delta_log_scale_factor=0.5,
        )
        assert "0.0316227766017\n" in out.read_text()

    def test_range_checks(self, tmp_path, cosmo):
        """Outputs outside [a_begin, a_end] are errors."""
        with pytest.raises(ValueError, match="before the start of the run"):
            run(tmp_path, first_snap_z=200.0, delta_z=1.0)
        with pytest.raises(ValueError, match="after the end of the run"):
            run(
                tmp_path,
                first_snap_z=1.0,
                final_snap_scale_factor=1.5,
                delta_z=0.1,
            )

    def test_single_output_is_an_error(self, tmp_path, cosmo):
        """SWIFT needs at least two outputs."""
        with pytest.raises(ValueError, match="at least 2"):
            run(tmp_path, first_snap_z=3.0, delta_z=5.0)

    def test_no_cosmology_flag(self, tmp_path, cosmo):
        """--no-cosmology ignores the Cosmology section."""
        with pytest.raises(ValueError, match="give the final output"):
            run(tmp_path, first_snap_z=3.0, delta_z=1.0, no_cosmology=True)


class TestNonCosmological:
    """Lists without a cosmology."""

    def test_time_list(self, tmp_path):
        """Times are written as given."""
        header, t, _ = read_list(
            run(
                tmp_path,
                first_snap_time=0.0,
                delta_time=0.3,
                final_snap_time=1.0,
            )
        )
        assert header == "# Time"
        assert t == pytest.approx([0.0, 0.3, 0.6, 0.9])

    def test_redshift_and_scale_factor_mix(self, tmp_path):
        """Redshift and scale factor convert without a cosmology."""
        header, a, _ = read_list(
            run(
                tmp_path,
                first_snap_z=1.0,
                delta_scale_factor=0.25,
                final_snap_z=0.0,
            )
        )
        assert header == "# Scale Factor"
        assert a == pytest.approx([0.5, 0.75, 1.0])

    def test_time_cannot_mix(self, tmp_path):
        """Time with redshift needs a cosmology."""
        with pytest.raises(ValueError, match="needs a cosmology"):
            run(
                tmp_path,
                first_snap_z=1.0,
                delta_time=0.1,
                final_snap_z=0.0,
            )

    def test_first_and_delta_are_required(self, tmp_path):
        """Missing first output or spacing is a clear error."""
        with pytest.raises(ValueError, match="--first-snap"):
            run(tmp_path, delta_time=0.1, final_snap_time=1.0)
        with pytest.raises(ValueError, match="--delta"):
            run(tmp_path, first_snap_time=0.0, final_snap_time=1.0)

    def test_final_is_required(self, tmp_path):
        """Without a cosmology there is no a_end to default to."""
        with pytest.raises(ValueError, match="--final-snap"):
            run(tmp_path, first_snap_time=0.0, delta_time=0.1)


class TestArguments:
    """Argument parsing."""

    def parse(self, *argv):
        """Parse output-times arguments."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        return parser.parse_args(argv)

    def test_exclusive_groups(self):
        """The first output and the spacing are each in one unit."""
        args = self.parse("--first-snap-z", "3", "--delta-z", "0.5")
        assert args.first_snap_z == 3.0 and args.delta_z == 0.5
        for argv in (
            [
                "--first-snap-z",
                "3",
                "--first-snap-time",
                "1",
                "--delta-z",
                "1",
            ],
            ["--first-snap-z", "3", "--delta-z", "1", "--delta-time", "1"],
            ["--first-snap-z", "3", "--delta-z", "0"],
            ["--first-snap-z", "3", "--delta-z", "-1"],
        ):
            with pytest.raises(SystemExit):
                self.parse(*argv)


class TestCLI:
    """Through cli.main."""

    def test_refuses_to_overwrite(self, tmp_path, capsys):
        """An existing list needs --force."""
        out = tmp_path / "list.txt"
        out.write_text("keep")
        argv = [
            "output-times",
            "--first-snap-time",
            "0",
            "--delta-time",
            "0.5",
            "--final-snap-time",
            "1",
            "--output-list",
            str(out),
        ]
        with pytest.raises(SystemExit):
            main(argv)
        assert "--force" in capsys.readouterr().err
        assert out.read_text() == "keep"
        main([*argv, "--force"])
        assert read_list(out)[1] == pytest.approx([0.0, 0.5, 1.0])

    def test_uses_params_cosmology(self, tmp_path):
        """-P supplies the cosmology and a_end."""
        params = tmp_path / "params.yml"
        params.write_text(COSMO_PARAMS.replace("a_end: 1.0", "a_end: 0.5"))
        out = tmp_path / "list.txt"
        main(
            [
                "output-times",
                "-P",
                str(params),
                "--first-snap-scale-factor",
                "0.1",
                "--delta-scale-factor",
                "0.1",
                "--output-list",
                str(out),
            ]
        )
        assert read_list(out)[1] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


def test_converter_without_cosmology_passes_time_through():
    """Time is its own ordering without a cosmology."""
    convert = Converter(cosmological=False)
    assert convert.to_a("time", 2.0) == 2.0
    assert convert.from_a("z", np.array([1.0]))[0] == 0.0
