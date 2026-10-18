"""Tests for the utils mode and the shared IC header / softening code."""

import argparse

import h5py
import numpy as np
import pytest
from ruamel.yaml import YAML

from swiftsim_cli.cli import main
from swiftsim_cli.modes.utils import (
    add_arguments,
    compute_softenings,
    ic_unit_parameters,
    read_ic_header,
)


def write(path, **attrs):
    """Write an IC file whose Header has the given attributes."""
    units = attrs.pop("units", None)
    with h5py.File(path, "w") as f:
        header = f.create_group("Header")
        for key, value in attrs.items():
            header.attrs[key] = value
        if units is not None:
            group = f.create_group("Units")
            for key, value in units.items():
                group.attrs[key] = value
    return path


class TestReadHeader:
    """read_ic_header copes with the IC variants found in the wild."""

    def test_scalar_attributes_and_no_units(self, tmp_path):
        """Scalar BoxSize/Time, no Redshift, no Units group."""
        header = read_ic_header(
            write(
                tmp_path / "a.hdf5",
                BoxSize=4.2,
                NumPart_Total=np.array([0, 10, 0, 0, 0, 0]),
                Time=0.0078125,
            )
        )
        assert list(header.box_size) == [4.2, 4.2, 4.2]
        assert header.redshift is None and header.units == {}
        assert header.is_cosmological and header.a_begin == 0.0078125

    def test_array_attributes_high_word_and_units(self, tmp_path):
        """1-element arrays, NumPart_Total_HighWord and a Units group."""
        header = read_ic_header(
            write(
                tmp_path / "b.hdf5",
                BoxSize=np.array([10.0]),
                NumPart_Total=np.array([5, 0, 0, 0, 0, 0, 0], dtype=np.uint32),
                NumPart_Total_HighWord=np.array([1, 0, 0, 0, 0, 0, 0]),
                Redshift=np.array([0.0]),
                Time=np.array([0.0]),
                units={
                    "Unit length in cgs (U_L)": 3.0e21,
                    "Unit time in cgs (U_t)": np.array([3.0e16]),
                },
            )
        )
        assert header.count(0) == 2**32 + 5
        assert header.count(6) == 0 and header.count(9) == 0
        assert not header.is_cosmological
        assert ic_unit_parameters(header) == {
            "UnitLength_in_cgs": 3.0e21,
            "UnitVelocity_in_cgs": pytest.approx(1e5),
        }

    @pytest.mark.parametrize(
        "attrs, message",
        [({}, "no BoxSize"), ({"BoxSize": 1.0}, "no NumPart_Total")],
    )
    def test_incomplete_header(self, tmp_path, attrs, message):
        """A header without the essentials is a clear error."""
        with pytest.raises(ValueError, match=message):
            read_ic_header(write(tmp_path / "c.hdf5", **attrs))

    def test_not_hdf5(self, tmp_path):
        """A non-HDF5 file is a clear error."""
        bad = tmp_path / "bad.hdf5"
        bad.write_text("text")
        with pytest.raises(ValueError, match="not a readable HDF5"):
            read_ic_header(bad)


def test_compute_softenings_species_and_2d(tmp_path):
    """DM, baryons (gas + stars) and neutrinos; 2D uses N^(1/2)."""
    header = read_ic_header(
        write(
            tmp_path / "d.hdf5",
            BoxSize=np.array([8.0, 8.0, 8.0]),
            NumPart_Total=np.array([4, 64, 0, 0, 4, 0, 8]),
        )
    )
    rows, gravity = compute_softenings(header, {}, 0.5, 2.0)
    assert [r["label"] for r in rows] == [
        "Dark matter",
        "Baryons",
        "Neutrinos",
    ]
    assert gravity == {
        "max_physical_DM_softening": 1.0,
        "max_physical_baryon_softening": 2.0,
        "max_physical_nu_softening": 2.0,
    }
    cosmo = {"Cosmology": {"h": 0.5}, "InitialConditions": {}}
    _, gravity = compute_softenings(header, cosmo, 0.5, 2.0)
    assert gravity["comoving_DM_softening"] == 1.0
    assert gravity["max_physical_DM_softening"] == pytest.approx(1 / 3)

    header.dimension = 2
    _, gravity = compute_softenings(header, {}, 0.5, 2.0)
    assert gravity["max_physical_DM_softening"] == 0.5  # sqrt(64/64) x 8 / 8


def test_no_particles(tmp_path):
    """ICs with only background particles cannot be softened."""
    header = read_ic_header(
        write(tmp_path / "e.hdf5", BoxSize=1.0, NumPart_Total=[0, 0, 5])
    )
    with pytest.raises(ValueError, match="no dark matter"):
        compute_softenings(header, {}, 0.04, 2.7)


def test_arguments_params_optional():
    """-P is optional for utils softenings."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["softenings", "--ics", __file__])
    assert args.params is None


class TestSofteningsCLI:
    """utils softenings agrees with what new writes."""

    def test_matches_new(self, tmp_path, monkeypatch, capsys):
        """Same ICs, same softenings, with and without -P."""
        monkeypatch.chdir(tmp_path)
        ics = write(
            tmp_path / "ics.hdf5",
            BoxSize=4.235625,
            NumPart_Total=np.array([0, 94**3, 0, 0, 0, 0]),
            Time=0.0078125,
        )
        main(["utils", "softenings", "--ics", str(ics)])
        report = capsys.readouterr().out
        assert "Parameters: template cosmo-dmo" in report
        assert "a_begin=0.0078125, z_begin=127" in report
        assert "2.6596e-03" in report  # 0.04 x 6.25 Mpc / 94

        main(["new", "-o", "run", "--ics", str(ics)])
        params = YAML(typ="safe").load(
            (tmp_path / "run/params.yml").read_text()
        )
        assert params["Gravity"]["comoving_DM_softening"] == pytest.approx(
            2.6596e-3, rel=1e-4
        )
        capsys.readouterr()

        main(
            [
                "utils",
                "softenings",
                "--ics",
                str(ics),
                "-P",
                "run/params.yml",
                "--softening-coeff",
                "0.08",
            ]
        )
        report = capsys.readouterr().out
        assert "Parameters: run/params.yml" in report
        assert "5.3191e-03" in report

    def test_isolated(self, tmp_path, capsys):
        """Non-cosmological ICs report physical softenings only."""
        ics = write(
            tmp_path / "ics.hdf5",
            BoxSize=100.0,
            NumPart_Total=np.array([1000, 0, 0, 0, 0, 0]),
        )
        main(["utils", "softenings", "--ics", str(ics)])
        out = capsys.readouterr().out
        assert "Non-cosmological" in out
        assert "Dark matter" not in out
        assert "4.0000e-01" in out
