"""Tests for the new mode module."""

import argparse
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from ruamel.yaml import YAML

from swiftsim_cli.modes.new import _kv_pair, add_arguments


class TestKVPair:
    """Test the _kv_pair function."""

    def test_kv_pair_valid(self):
        """Test parsing a valid KEY=VALUE pair."""
        key, val = _kv_pair("Cosmology:h=0.7")
        assert key == "Cosmology:h"
        assert val == "0.7"

    def test_kv_pair_with_equals_in_value(self):
        """Test parsing KEY=VALUE where VALUE contains =."""
        key, val = _kv_pair("key=value=with=equals")
        assert key == "key"
        assert val == "value=with=equals"

    def test_kv_pair_invalid_no_equals(self):
        """Test that invalid input without = raises error."""
        with pytest.raises(
            argparse.ArgumentTypeError, match="invalid parameter override"
        ):
            _kv_pair("invalid_no_equals")

    def test_kv_pair_empty_value(self):
        """Test parsing KEY= with empty value."""
        key, val = _kv_pair("key=")
        assert key == "key"
        assert val == ""


class TestAddArguments:
    """Test the add_arguments function."""

    def test_add_arguments(self):
        """Test that add_arguments adds all required arguments."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        # Test that required arguments are present
        # This should fail without --path and --ics
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_add_arguments_with_required_args(self):
        """Test parsing with required arguments."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        args = parser.parse_args(
            ["--path", "/test/path", "--ics", "/test/ic.hdf5"]
        )
        assert str(args.path) == "/test/path"
        assert str(args.ics) == "/test/ic.hdf5"

    def test_add_arguments_with_param_override(self):
        """Test parsing with parameter overrides."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        args = parser.parse_args(
            [
                "--path",
                "/test/path",
                "--ics",
                "/test/ic.hdf5",
                "--param",
                "Cosmology:h=0.7",
                "--param",
                "TimeIntegration:dt_min=0.0001",
            ]
        )
        assert args.param == [
            ("Cosmology:h", "0.7"),
            ("TimeIntegration:dt_min", "0.0001"),
        ]


class TestApplyOverrides:
    """Test the apply_overrides function."""

    def test_apply_overrides_valid(self):
        """Test applying valid parameter overrides."""
        from swiftsim_cli.modes.new import apply_overrides

        params = {
            "Cosmology": {"h": 0.67, "Omega_b": 0.05},
            "TimeIntegration": {"dt_min": 0.001},
        }
        overrides = {"Cosmology:h": "0.7", "TimeIntegration:dt_min": "0.0001"}

        apply_overrides(params, overrides)

        assert params["Cosmology"]["h"] == 0.7
        assert params["TimeIntegration"]["dt_min"] == 0.0001

    def test_apply_overrides_parses_yaml_scalars(self):
        """Numeric overrides become numbers, other strings stay strings."""
        from swiftsim_cli.modes.new import apply_overrides

        params = {"Snapshots": {"basename": "x", "delta_time": 1.0}}
        apply_overrides(
            params,
            {"Snapshots:basename": "snap", "Snapshots:delta_time": "1e-2"},
        )

        assert params["Snapshots"]["basename"] == "snap"
        assert params["Snapshots"]["delta_time"] == 0.01

    def test_apply_overrides_none(self):
        """Test that None overrides don't modify params."""
        from swiftsim_cli.modes.new import apply_overrides

        params = {"Cosmology": {"h": 0.67}}
        apply_overrides(params, None)
        assert params["Cosmology"]["h"] == 0.67

    def test_apply_overrides_invalid_key_format_no_colon(self):
        """Test error when key has no colon."""
        from swiftsim_cli.modes.new import apply_overrides

        params = {"Cosmology": {"h": 0.67}}
        overrides = {"invalid_key": "0.7"}

        with pytest.raises(
            ValueError, match="Keys should be provided in the format"
        ):
            apply_overrides(params, overrides)

    def test_apply_overrides_invalid_key_format_too_many_colons(self):
        """Test error when key has too many colons."""
        from swiftsim_cli.modes.new import apply_overrides

        params = {"Cosmology": {"h": 0.67}}
        overrides = {"Parent:Child:Extra": "0.7"}

        with pytest.raises(ValueError, match="Keys should be in the format"):
            apply_overrides(params, overrides)

    def test_apply_overrides_parent_not_found(self):
        """Test error when parent key doesn't exist."""
        from swiftsim_cli.modes.new import apply_overrides

        params = {"Cosmology": {"h": 0.67}}
        overrides = {"NonExistent:h": "0.7"}

        with pytest.raises(
            ValueError, match="Parent key 'NonExistent' not found"
        ):
            apply_overrides(params, overrides)

    def test_apply_overrides_child_not_found(self):
        """Test error when child key doesn't exist."""
        from swiftsim_cli.modes.new import apply_overrides

        params = {"Cosmology": {"h": 0.67}}
        overrides = {"Cosmology:nonexistent": "0.7"}

        with pytest.raises(
            ValueError, match="Key 'nonexistent' not found in parent"
        ):
            apply_overrides(params, overrides)


class TestDeriveParamsFromICs:
    """Regression tests for parameter derivation from IC files."""

    @patch("swiftsim_cli.modes.new.h5py.File")
    @patch("swiftsim_cli.modes.new.load_swift_profile")
    def test_redshift_only_header_does_not_require_time(
        self, mock_load_profile, mock_h5_file, tmp_path
    ):
        """IC files with Redshift but no Time should not crash."""
        inicond_file = tmp_path / "ics.hdf5"
        inicond_file.touch()

        mock_load_profile.return_value = Mock(
            softening_coeff=0.04,
            softening_pivot_z=2.7,
        )

        header_attrs = {
            "BoxSize": 100.0,
            "NumPart_Total": [64, 64, 0, 0, 0, 0],
            "Redshift": 9.0,
        }
        mock_hdf = {
            "Header": Mock(attrs=header_attrs),
        }
        mock_h5_file.return_value.__enter__.return_value = mock_hdf

        params = {
            "Gravity": {},
            "Cosmology": {},
            "TimeIntegration": {},
            "InternalUnitSystem": {},
        }

        from swiftsim_cli.modes.new import derive_params_from_ics

        result = derive_params_from_ics(inicond_file, params)

        assert result["Cosmology"]["a_begin"] == 0.1
        assert "time_begin" not in result["TimeIntegration"]


TEMPLATE_YML = """\
InternalUnitSystem:
  UnitMass_in_cgs: 1.0
  UnitLength_in_cgs: 1.0
  UnitVelocity_in_cgs: 1.0
  UnitCurrent_in_cgs: 1.0
  UnitTemp_in_cgs: 1.0
Cosmology:
  a_begin: 0.0078125
  a_end: 1.0
TimeIntegration:
  time_begin: 0.
  dt_max: 1e-2
Gravity:
  comoving_DM_softening: 0.0026994
  max_physical_DM_softening: 0.0007
  comoving_baryon_softening: 0.0026994
  max_physical_baryon_softening: 0.0007
Snapshots:
  subdir: dir
  basename: output
Restarts:
  subdir: restart
InitialConditions:
  file_name: SedovBlast/sedov.hdf5
"""


@pytest.fixture
def new_env(tmp_path, monkeypatch, write_tiny_ics):
    """A fake SWIFT checkout, a profile pointing at it and a tiny IC."""
    swift_dir = tmp_path / "swiftsim"
    (swift_dir / "examples").mkdir(parents=True)
    (swift_dir / "examples" / "parameter_example.yml").write_text(TEMPLATE_YML)
    profile = SimpleNamespace(
        swiftsim_dir=swift_dir,
        softening_coeff=0.04,
        softening_pivot_z=2.7,
    )
    monkeypatch.setattr(
        "swiftsim_cli.modes.new.load_swift_profile", lambda: profile
    )
    monkeypatch.setattr(
        "swiftsim_cli.modes.new.get_swiftsim_dir",
        lambda swift_dir=None: swift_dir or profile.swiftsim_dir,
    )
    write_tiny_ics(tmp_path / "ics.hdf5")
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestNewEndToEnd:
    """Run the new mode through the CLI with a real HDF5 IC file."""

    def test_units_group_no_dm_relative_paths(self, new_env):
        """Units/no-DM ICs work and give numeric values + absolute paths."""
        from swiftsim_cli.cli import main

        main(
            [
                "new",
                "--path",
                "run",
                "--ics",
                "ics.hdf5",
                "--param",
                "TimeIntegration:dt_max=0.5",
                "--param",
                "Snapshots:basename=snap",
            ]
        )

        text = (new_env / "run" / "params.yml").read_text()
        params = YAML(typ="safe").load(text)

        # Paths are absolute so SWIFT can run from inside the run dir
        assert params["InitialConditions"]["file_name"] == str(
            (new_env / "ics.hdf5").resolve()
        )
        assert params["Snapshots"]["subdir"] == str(
            (new_env / "run" / "snapshots").resolve()
        )

        # Units come through as plain numbers
        units = params["InternalUnitSystem"]
        assert units["UnitMass_in_cgs"] == pytest.approx(1.989e43)
        assert units["UnitLength_in_cgs"] == pytest.approx(3.085e24)

        # SWIFT has no time unit; velocity is derived as U_L / U_t
        assert "UnitTime_in_cgs" not in units
        assert units["UnitVelocity_in_cgs"] == pytest.approx(1e5)

        # Softenings are numbers; with no DM they mirror the gas
        grav = params["Gravity"]
        assert grav["comoving_baryon_softening"] == pytest.approx(1.0)
        assert grav["comoving_DM_softening"] == pytest.approx(1.0)

        # Overrides are parsed as scalars, numbers are never quoted
        assert params["TimeIntegration"]["dt_max"] == 0.5
        assert params["Snapshots"]["basename"] == "snap"
        assert "'" not in text and '"' not in text

        # Non-cosmological IC (Time=0): a_begin untouched, time_begin set
        assert params["Cosmology"]["a_begin"] == 0.0078125
        assert params["TimeIntegration"]["time_begin"] == 0.0

        # No temporary files are left behind
        assert sorted(p.name for p in (new_env / "run").iterdir()) == [
            "params.yml"
        ]

    def test_refuses_overwrite_without_force(self, new_env, capsys):
        """An existing params.yml is only replaced with --force."""
        from swiftsim_cli.cli import main

        argv = ["new", "--path", "run", "--ics", "ics.hdf5"]
        main(argv)
        param_file = new_env / "run" / "params.yml"
        param_file.write_text("sentinel")

        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 1
        assert "--force" in capsys.readouterr().err
        assert param_file.read_text() == "sentinel"

        main([*argv, "--force"])
        assert param_file.read_text() != "sentinel"

    def test_cosmological_time_sets_a_begin(self, new_env, write_tiny_ics):
        """A scale-factor Time (0 < Time <= 1) sets Cosmology:a_begin."""
        from swiftsim_cli.cli import main

        write_tiny_ics(new_env / "cosmo.hdf5", time=0.1)
        main(["new", "--path", "run", "--ics", "cosmo.hdf5"])

        params = YAML(typ="safe").load(
            (new_env / "run" / "params.yml").read_text()
        )
        assert params["Cosmology"]["a_begin"] == pytest.approx(0.1)
        assert params["TimeIntegration"]["time_begin"] == 0.0
