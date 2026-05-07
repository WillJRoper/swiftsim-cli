"""Tests for the new mode module."""

import argparse
from unittest.mock import Mock, patch

import pytest

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
        # This should fail without --path and --inic
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_add_arguments_with_required_args(self):
        """Test parsing with required arguments."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        args = parser.parse_args(
            ["--path", "/test/path", "--inic", "/test/ic.hdf5"]
        )
        assert str(args.path) == "/test/path"
        assert str(args.inic) == "/test/ic.hdf5"

    def test_add_arguments_with_param_override(self):
        """Test parsing with parameter overrides."""
        parser = argparse.ArgumentParser()
        add_arguments(parser)

        args = parser.parse_args(
            [
                "--path",
                "/test/path",
                "--inic",
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

        assert params["Cosmology"]["h"] == "0.7"
        assert params["TimeIntegration"]["dt_min"] == "0.0001"

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

        assert result["Cosmology"]["a_begin"] == "0.1"
        assert "time_begin" not in result["TimeIntegration"]
