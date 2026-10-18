"""Tests for the new mode (templates, IC-derived settings, checks)."""

import argparse
import stat
from typing import Any

import h5py
import numpy as np
import pytest
from ruamel.yaml import YAML

from swiftsim_cli.cli import main
from swiftsim_cli.modes.new import (
    _kv_pair,
    _tidy_blank_lines,
    add_arguments,
    apply_overrides,
    make_new_run_dir,
)
from swiftsim_cli.params import check_swift_parameter_text
from swiftsim_cli.templates import TEMPLATES, choose_template, template_path

EAGLE_BOX = 4.235625  # 6.25 Mpc in h^-1 Mpc (h = 0.6777)


def write_ics(
    path,
    counts=(0, 1000, 0, 0, 0, 0),
    box=EAGLE_BOX,
    time=None,
    redshift=None,
    units=None,
    masses=None,
):
    """Write a header-only IC file (plus particle masses if given)."""
    with h5py.File(path, "w") as f:
        header = f.create_group("Header")
        header.attrs["BoxSize"] = box
        header.attrs["NumPart_Total"] = np.array(counts, dtype=np.uint32)
        if time is not None:
            header.attrs["Time"] = time
        if redshift is not None:
            header.attrs["Redshift"] = redshift
        if units is not None:
            group = f.create_group("Units")
            for key, value in units.items():
                group.attrs[key] = value
        for part_type, mass in (masses or {}).items():
            f.create_dataset(
                f"PartType{part_type}/Masses",
                data=np.full(counts[part_type], mass),
            )
    return path


KPC_UNITS = {
    "Unit mass in cgs (U_M)": np.array([1.98848e43]),
    "Unit length in cgs (U_L)": np.array([3.085678e21]),
    "Unit time in cgs (U_t)": np.array([3.085678e16]),
    "Unit current in cgs (U_I)": np.array([1.0]),
    "Unit temperature in cgs (U_T)": np.array([1.0]),
}


def load(path):
    """Load a written parameter file."""
    return YAML(typ="safe").load(path.read_text())


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """Work in tmp_path; no SWIFT worktree is configured."""
    monkeypatch.chdir(tmp_path)
    return tmp_path / "run"


class TestArguments:
    """Argument parsing."""

    def test_kv_pair(self):
        """SECTION:KEY=VALUE splits on the first '='."""
        assert _kv_pair("Cosmology:h=0.7") == ("Cosmology:h", "0.7")
        assert _kv_pair("A:b=x=y") == ("A:b", "x=y")

    @pytest.mark.parametrize(
        "text",
        ["novalue", "key=1", "A:b:c=1", ":b=1", "A:=1", "A:b=", "A:b= "],
    )
    def test_kv_pair_rejects(self, text):
        """Missing sections, keys or values are usage errors."""
        with pytest.raises(argparse.ArgumentTypeError):
            _kv_pair(text)

    def test_template_and_params_are_exclusive(self, tmp_path):
        """--template and -P/--params cannot both be given."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        ics = write_ics(tmp_path / "ics.hdf5")
        base = ["-o", "run", "--ics", str(ics)]
        args = parser.parse_args([*base, "--template", "cosmo-dmo", "--check"])
        assert args.template == "cosmo-dmo" and args.check
        with pytest.raises(SystemExit):
            parser.parse_args(
                [*base, "--template", "cosmo-dmo", "-P", "own.yml"]
            )


class TestOverrides:
    """--set value handling."""

    def test_values(self, capsys):
        """Numbers become numbers, lists flow lists, the rest verbatim."""
        params: dict[str, Any] = {"A": {"i": 0, "f": 0.0, "s": "x"}}
        apply_overrides(
            params,
            {
                "A:i": "3",
                "A:f": "1e-2",
                "A:s": "2025-01-01 # not a comment",
                "A:l": "[1, 0, 2.5]",
                "B:new": "yes",
            },
        )
        assert params["A"]["i"] == 3 and isinstance(params["A"]["i"], int)
        assert params["A"]["f"] == 0.01
        assert params["A"]["s"] == "2025-01-01 # not a comment"
        assert list(params["A"]["l"]) == [1, 0, 2.5]
        assert params["B"]["new"] == "yes"
        err = capsys.readouterr().err
        assert "adding new parameter A:l" in err
        assert "adding new section B" in err

    def test_bad_key(self):
        """Keys must be SECTION:KEY."""
        with pytest.raises(ValueError, match="SECTION:KEY"):
            apply_overrides({}, {"nokey": "1"})


class TestTemplates:
    """The bundled templates."""

    @pytest.mark.parametrize("name", list(TEMPLATES))
    def test_templates_pass_parser_checks(self, name):
        """Every template is within SWIFT's parser limits."""
        text = template_path(name).read_text()
        assert check_swift_parameter_text(text) == []
        params = YAML(typ="safe").load(text)
        assert ("Cosmology" in params) == name.startswith("cosmo")
        assert ("SPH" in params) == name.endswith("hydro")

    def test_choose_template(self):
        """Templates are picked by cosmology and gas."""
        assert choose_template(True, False) == "cosmo-dmo"
        assert choose_template(True, True) == "cosmo-hydro"
        assert choose_template(False, False) == "isolated-gravity"
        assert choose_template(False, True) == "isolated-hydro"

    def test_unknown_template(self):
        """An unknown template name is a ValueError."""
        with pytest.raises(ValueError, match="unknown template"):
            template_path("nope")


class TestCosmological:
    """Cosmological ICs."""

    def test_gadget_style_dmo_ics(self, run_dir, capsys):
        """EAGLE-like ICs: no Units, Time = a_begin, h^-1 box."""
        ics = write_ics(
            run_dir.parent / "ics.hdf5", (0, 94**3, 0, 0, 0, 0), time=0.0078125
        )
        make_new_run_dir(run_dir, ics)

        params = load(run_dir / "params.yml")
        assert "SPH" not in params
        assert params["Cosmology"]["a_begin"] == 0.0078125
        assert params["Snapshots"]["scale_factor_first"] == 0.0078125
        assert params["Statistics"]["scale_factor_first"] == 0.0078125
        ic = params["InitialConditions"]
        assert ic["cleanup_h_factors"] == 1
        assert ic["cleanup_velocity_factors"] == 1
        # EAGLE: 0.04 x 6.25 Mpc / 94 = 2.66 ckpc, /3.7 physical
        grav = params["Gravity"]
        assert grav["comoving_DM_softening"] == pytest.approx(0.00265957)
        assert grav["max_physical_DM_softening"] == pytest.approx(
            0.00265957 / 3.7, rel=1e-5
        )
        assert "comoving_baryon_softening" not in grav
        out = capsys.readouterr().out
        assert "Starting from template cosmo-dmo" in out
        assert "swift --cosmology --self-gravity params.yml" in out

    def test_redshift_sets_a_begin_and_units_are_used(self, run_dir):
        """A positive Redshift wins; a Units group sets the unit system."""
        ics = write_ics(
            run_dir.parent / "ics.hdf5",
            (8, 8, 0, 0, 0, 0),
            box=np.array([10.0]),
            time=np.array([0.5]),
            redshift=np.array([9.0]),
            units=KPC_UNITS,
        )
        make_new_run_dir(run_dir, ics)

        params = load(run_dir / "params.yml")
        assert params["Cosmology"]["a_begin"] == pytest.approx(0.1)
        assert "SPH" in params  # gas -> cosmo-hydro
        units = params["InternalUnitSystem"]
        assert units["UnitLength_in_cgs"] == pytest.approx(3.085678e21)
        assert units["UnitVelocity_in_cgs"] == pytest.approx(1e5)
        assert "# 1 Mpc" not in (run_dir / "params.yml").read_text()
        assert params["InitialConditions"]["cleanup_h_factors"] == 0
        # 0.04 x 10 / 2 (no h-scaling with a Units group)
        assert params["Gravity"]["comoving_baryon_softening"] == 0.2

    def test_hydro_template_on_dmo_ics_generates_gas(self, run_dir):
        """A hydro run from DM-only ICs asks SWIFT to generate the gas."""
        ics = write_ics(run_dir.parent / "ics.hdf5", time=0.01)
        make_new_run_dir(run_dir, ics, template="cosmo-hydro")

        params = load(run_dir / "params.yml")
        assert params["InitialConditions"]["generate_gas_in_ics"] == 1
        grav = params["Gravity"]
        assert grav["comoving_baryon_softening"] == pytest.approx(
            grav["comoving_DM_softening"]
        )

    @pytest.mark.parametrize("zoom_support", [True, False])
    def test_zoom_ics(self, run_dir, tmp_path, zoom_support):
        """Zoom softening comes from the high-res particle mass."""
        swift_dir = tmp_path / "swift"
        (swift_dir / "src").mkdir(parents=True)
        if zoom_support:
            (swift_dir / "src" / "zoom_region").mkdir()
        h, omega_m = 0.6777, 0.2587481 + 0.0482519
        rho_crit = 2.775e1 * h**2  # 1e10 Msun / Mpc^3
        sep = 0.1  # Mpc, chosen
        mass = omega_m * rho_crit * sep**3 * h  # in 1e10 Msun / h
        ics = write_ics(
            run_dir.parent / "ics.hdf5",
            (0, 100, 50, 0, 0, 0),
            box=300.0,
            redshift=127.0,
            masses={1: mass, 2: 10 * mass},
        )
        make_new_run_dir(run_dir, ics, swift_dir=swift_dir)

        params = load(run_dir / "params.yml")
        grav = params["Gravity"]
        assert grav["comoving_DM_softening"] == pytest.approx(
            0.04 * sep, rel=1e-3
        )
        assert grav["softening_ratio_background"] == 0.04
        assert ("ZoomRegion" in params) == zoom_support


class TestNonCosmological:
    """Isolated / non-cosmological ICs."""

    def test_isolated_hydro(self, run_dir, capsys):
        """No Cosmology section; physical softening = coeff x separation."""
        ics = write_ics(
            run_dir.parent / "ics.hdf5",
            (1000, 0, 0, 0, 0, 0),
            box=np.array([100.0, 100.0, 100.0]),
            time=np.array([0.0]),
            redshift=np.array([0.0]),
            units=KPC_UNITS,
        )
        make_new_run_dir(run_dir, ics)

        params = load(run_dir / "params.yml")
        assert "Cosmology" not in params
        assert params["InitialConditions"]["periodic"] == 0
        assert params["TimeIntegration"]["time_begin"] == 0.0
        assert params["Snapshots"]["time_first"] == 0.0
        assert params["Gravity"]["max_physical_baryon_softening"] == 0.4
        assert "comoving_baryon_softening" not in params["Gravity"]
        assert "means little for an isolated object" in capsys.readouterr().err

    def test_stars_add_the_stars_flag(self, run_dir, capsys):
        """SWIFT drops stars without --stars, so it is suggested."""
        ics = write_ics(
            run_dir.parent / "ics.hdf5", (10, 0, 0, 0, 10, 0), box=10.0
        )
        make_new_run_dir(run_dir, ics)
        out = capsys.readouterr().out
        assert "swift --self-gravity --hydro --stars params.yml" in out


class TestOutputFile:
    """What is written, and when nothing is."""

    def test_long_values_stay_on_one_line(self, tmp_path, monkeypatch):
        """Every key: value is on one line, however long the paths."""
        deep = tmp_path / ("long_directory_name_" * 2) / "more"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)
        ics = write_ics(deep / ("ics_" + "x" * 20 + ".hdf5"), time=0.01)
        make_new_run_dir(deep / "run", ics)

        text = (deep / "run" / "params.yml").read_text()
        assert check_swift_parameter_text(text) == []
        long_lines = [x for x in text.splitlines() if len(x) > 80]
        assert long_lines  # the test really has long values
        for line in text.splitlines():
            content = line.split("#", 1)[0]
            if line.startswith(" ") and content.strip():
                key, _, value = content.partition(":")
                assert value.strip(), f"value wrapped off line: {line!r}"
        params = load(deep / "run" / "params.yml")
        assert params["InitialConditions"]["file_name"] == str(ics)

    def test_refuses_overwrite_without_force(self, run_dir, capsys):
        """An existing params.yml is only replaced with --force."""
        ics = write_ics(run_dir.parent / "ics.hdf5", time=0.01)
        argv = ["new", "-o", "run", "--ics", str(ics)]
        main(argv)
        (run_dir / "params.yml").write_text("sentinel")
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 1
        assert "--force" in capsys.readouterr().err
        main([*argv, "--force"])
        assert (run_dir / "params.yml").read_text() != "sentinel"

    def test_bad_ics_leave_no_run_dir(self, run_dir, capsys):
        """A failure before writing leaves no empty run directory."""
        bad = run_dir.parent / "bad.hdf5"
        bad.write_text("not hdf5")
        with pytest.raises(SystemExit):
            main(["new", "-o", "run", "--ics", str(bad)])
        assert "not a readable HDF5 file" in capsys.readouterr().err
        assert not run_dir.exists()

    def test_parser_limits_are_enforced(self, run_dir, capsys):
        """A value SWIFT cannot read fails before anything is written."""
        ics = write_ics(run_dir.parent / "ics.hdf5", time=0.01)
        with pytest.raises(SystemExit):
            main(
                [
                    "new",
                    "-o",
                    "run",
                    "--ics",
                    str(ics),
                    "--set",
                    f"MetaData:run_name={'x' * 300}",
                    "--set",
                    "Gravity:max_adaptive_softening=FLT_MAX",
                ]
            )
        err = capsys.readouterr().err
        assert "would be rejected by SWIFT" in err
        assert "characters long" in err
        assert "placeholder FLT_MAX" in err
        assert not run_dir.exists()

    def test_own_parameter_file(self, run_dir, capsys):
        """-P starts from the user's file (comments kept, Cosmology kept)."""
        own = run_dir.parent / "own.yml"
        own.write_text(
            "# my run\nCosmology:\n  h: 0.7\n  Omega_cdm: 0.25\n"
            "  Omega_lambda: 0.7\n  Omega_b: 0.05\n  a_begin: 0.01\n"
            "  a_end: 1.0\nMetaData:\n  run_name: mine  # keep me\n"
        )
        ics = write_ics(run_dir.parent / "ics.hdf5", time=0.02)
        main(["new", "-o", "run", "--ics", str(ics), "-P", str(own)])
        text = (run_dir / "params.yml").read_text()
        assert "# keep me" in text
        params = load(run_dir / "params.yml")
        assert params["MetaData"]["run_name"] == "mine"
        assert params["Cosmology"]["a_begin"] == 0.02
        assert f"Starting from {own}" in capsys.readouterr().out

    def test_profile_parameter_file(self, run_dir, monkeypatch):
        """The profile's parameter_file is used unless it is the example."""
        from types import SimpleNamespace

        own = run_dir.parent / "own.yml"
        own.write_text("MetaData:\n  run_name: from-profile\n")
        example = run_dir.parent / "parameter_example.yml"
        example.write_text("Bad: [\n")
        profile = SimpleNamespace(
            parameter_file=own, softening_coeff=0.04, softening_pivot_z=2.7
        )
        monkeypatch.setattr(
            "swiftsim_cli.modes.new.load_swift_profile", lambda: profile
        )
        ics = write_ics(run_dir.parent / "ics.hdf5", box=10.0)
        make_new_run_dir(run_dir, ics)
        assert load(run_dir / "params.yml")["MetaData"]["run_name"] == (
            "from-profile"
        )
        profile.parameter_file = example
        make_new_run_dir(run_dir, ics, force=True)
        assert (
            "isolated" in load(run_dir / "params.yml")["MetaData"]["run_name"]
        )


class TestCheck:
    """new --check runs swift --dry-run."""

    def _swift(self, tmp_path, script):
        swift_dir = tmp_path / "swift"
        swift_dir.mkdir()
        binary = swift_dir / "swift"
        binary.write_text("#!/bin/sh\n" + script)
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        return swift_dir

    def test_pass(self, run_dir, tmp_path, capsys):
        """A passing dry run is reported, run from the run directory."""
        swift_dir = self._swift(
            tmp_path,
            'echo "$@" > args.txt\necho "End of dry-run."\n',
        )
        ics = write_ics(run_dir.parent / "ics.hdf5", time=0.01)
        make_new_run_dir(run_dir, ics, swift_dir=swift_dir, check=True)
        assert (run_dir / "args.txt").read_text().split() == [
            "--dry-run",
            "--cosmology",
            "--self-gravity",
            "params.yml",
        ]
        assert (
            "SWIFT dry run passed: End of dry-run." in capsys.readouterr().out
        )

    def test_fail(self, run_dir, tmp_path):
        """A failing dry run is an error showing SWIFT's last lines."""
        swift_dir = self._swift(tmp_path, 'echo "bad parameter" >&2\nexit 1\n')
        ics = write_ics(run_dir.parent / "ics.hdf5", time=0.01)
        with pytest.raises(ValueError, match="bad parameter"):
            make_new_run_dir(run_dir, ics, swift_dir=swift_dir, check=True)

    def test_no_binary(self, run_dir):
        """Without a built SWIFT, --check says how to get one."""
        ics = write_ics(run_dir.parent / "ics.hdf5", time=0.01)
        with pytest.raises(ValueError, match="swift-cli make"):
            make_new_run_dir(run_dir, ics, check=True)


def test_tidy_blank_lines():
    """Blank lines end up between sections only."""
    text = "# head\n\nA:\n  x: 1\n\n  y: 2\nB:\n  z: 3\n\n\n"
    assert _tidy_blank_lines(text) == (
        "# head\n\nA:\n  x: 1\n  y: 2\n\nB:\n  z: 3\n"
    )
