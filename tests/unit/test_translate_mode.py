# ruff: noqa: D101, D102
"""Unit tests for the translate mode."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pytest

from swiftsim_cli.modes.translate import (
    _build_counts_offsets,
    _compute_cell_centres,
    _compute_cell_labels,
    _compute_cell_minmax,
    _parse_field_arg,
    _parse_fields,
    _parse_ndim,
    _resolve_input_files,
    _translate_snapshot,
    add_arguments,
    run,
)

# ---------------------------------------------------------------------------
# Field argument parsing
# ---------------------------------------------------------------------------


class TestParseFieldArg:
    def test_valid(self):
        assert _parse_field_arg("PartType0/Coordinates") == (
            "PartType0",
            "Coordinates",
        )

    def test_valid_leading_slash(self):
        assert _parse_field_arg("/PartType1/Velocities") == (
            "PartType1",
            "Velocities",
        )

    def test_invalid_no_parttype(self):
        with pytest.raises(
            argparse.ArgumentTypeError, match="Invalid --field argument"
        ):
            _parse_field_arg("Header/BoxSize")

    def test_invalid_too_many_parts(self):
        with pytest.raises(
            argparse.ArgumentTypeError, match="Invalid --field argument"
        ):
            _parse_field_arg("PartType0/Sub/Coordinates")

    def test_invalid_no_slash(self):
        with pytest.raises(
            argparse.ArgumentTypeError, match="Invalid --field argument"
        ):
            _parse_field_arg("Coordinates")


class TestParseFields:
    def test_single_field(self):
        result = _parse_fields(["PartType0/Coordinates"], ["PartType0"])
        assert result == {"PartType0": ["Coordinates"]}

    def test_multiple_fields_same_type(self):
        result = _parse_fields(
            ["PartType0/Coordinates", "PartType0/Masses"],
            ["PartType0"],
        )
        assert result == {"PartType0": ["Coordinates", "Masses"]}

    def test_multiple_types(self):
        result = _parse_fields(
            [
                "PartType0/Coordinates",
                "PartType1/Velocities",
            ],
            ["PartType0", "PartType1"],
        )
        assert result == {
            "PartType0": ["Coordinates"],
            "PartType1": ["Velocities"],
        }

    def test_duplicate_dedup(self):
        result = _parse_fields(
            ["PartType0/Coordinates", "PartType0/Coordinates"],
            ["PartType0"],
        )
        assert result == {"PartType0": ["Coordinates"]}

    def test_empty(self):
        assert _parse_fields([], ["PartType0"]) == {}

    def test_not_in_part_types(self):
        with pytest.raises(ValueError, match="not declared"):
            _parse_fields(["PartType0/Coordinates"], ["PartType1"])


class TestParseNdim:
    def test_scalar(self):
        result = _parse_ndim("box", ["10.0"], 3, float)
        np.testing.assert_array_equal(result, [10.0, 10.0, 10.0])

    def test_triplet(self):
        result = _parse_ndim("box", ["10", "20", "30"], 3, float)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_wrong_count(self):
        with pytest.raises(argparse.ArgumentTypeError, match="expects 1 or 3"):
            _parse_ndim("box", ["1", "2"], 3, float)

    def test_int_dtype(self):
        result = _parse_ndim("cdim", ["4"], 3, int)
        np.testing.assert_array_equal(result, [4, 4, 4])
        assert result.dtype == np.int64


class TestResolveInputFiles:
    def test_single_file(self, temp_dir):
        fp = temp_dir / "test.hdf5"
        fp.touch()
        result = _resolve_input_files([fp])
        assert result == [fp]

    def test_glob_pattern(self, temp_dir):
        (temp_dir / "snap_0.hdf5").touch()
        (temp_dir / "snap_1.hdf5").touch()
        (temp_dir / "snap_2.hdf5").touch()
        result = _resolve_input_files([temp_dir / "snap_*.hdf5"])
        assert len(result) == 3
        assert result == sorted(result)

    def test_glob_no_match(self, temp_dir):
        pattern = temp_dir / "nonexistent_*.hdf5"
        with pytest.raises(FileNotFoundError, match="No files matched"):
            _resolve_input_files([pattern])

    def test_multiple_explicit_paths(self, temp_dir):
        fp0 = temp_dir / "a.hdf5"
        fp1 = temp_dir / "b.hdf5"
        fp0.touch()
        fp1.touch()
        result = _resolve_input_files([fp0, fp1])
        assert set(result) == {fp0, fp1}

    def test_mixed_glob_and_explicit(self, temp_dir):
        fp = temp_dir / "explicit.hdf5"
        fp.touch()
        (temp_dir / "snap_0.hdf5").touch()
        result = _resolve_input_files([fp, temp_dir / "snap_*.hdf5"])
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Cell label computation
# ---------------------------------------------------------------------------


class TestComputeCellLabels:
    def test_single_cell(self):
        cdim = np.array([1, 1, 1], dtype=np.int64)
        boxsize = np.array([10.0, 10.0, 10.0])
        coords = np.array([[5.0, 5.0, 5.0]])
        labels = _compute_cell_labels(coords, boxsize, cdim)
        np.testing.assert_array_equal(labels, [0])

    def test_eight_cells_all_corners(self):
        cdim = np.array([2, 2, 2], dtype=np.int64)
        boxsize = np.array([10.0, 10.0, 10.0])
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 9.9],
                [0.0, 9.9, 0.0],
                [0.0, 9.9, 9.9],
                [9.9, 0.0, 0.0],
                [9.9, 0.0, 9.9],
                [9.9, 9.9, 0.0],
                [9.9, 9.9, 9.9],
            ]
        )
        labels = _compute_cell_labels(coords, boxsize, cdim)
        assert len(np.unique(labels)) == 8
        assert labels.min() == 0
        assert labels.max() == 7

    def test_periodic_wrapping(self):
        cdim = np.array([4, 4, 4], dtype=np.int64)
        boxsize = np.array([10.0, 10.0, 10.0])
        coords = np.array([[-1.0, 11.0, 20.0]])
        labels = _compute_cell_labels(coords, boxsize, cdim)
        wrapped = np.array([[9.0, 1.0, 0.0]])
        expected = _compute_cell_labels(wrapped, boxsize, cdim)
        np.testing.assert_array_equal(labels, expected)

    def test_c_order_flattening(self):
        cdim = np.array([2, 2, 3], dtype=np.int64)
        boxsize = np.array([2.0, 2.0, 3.0])
        coords = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 1.5],
                [0.5, 0.5, 2.5],
                [0.5, 1.5, 0.5],
                [0.5, 1.5, 1.5],
                [0.5, 1.5, 2.5],
                [1.5, 0.5, 0.5],
                [1.5, 0.5, 1.5],
                [1.5, 0.5, 2.5],
                [1.5, 1.5, 0.5],
                [1.5, 1.5, 1.5],
                [1.5, 1.5, 2.5],
            ]
        )
        labels = _compute_cell_labels(coords, boxsize, cdim)
        np.testing.assert_array_equal(labels, np.arange(12, dtype=np.int64))

    def test_clip_safety(self):
        cdim = np.array([2, 2, 2], dtype=np.int64)
        boxsize = np.array([10.0, 10.0, 10.0])
        coords = np.array([[1e15, -1e15, 5.0]])
        labels = _compute_cell_labels(coords, boxsize, cdim)
        assert 0 <= labels[0] < 8


# ---------------------------------------------------------------------------
# Counts and offsets
# ---------------------------------------------------------------------------


class TestBuildCountsOffsets:
    def test_simple(self):
        labels = np.array([0, 0, 0, 1, 1, 3, 3, 3], dtype=np.int64)
        counts, offsets = _build_counts_offsets(labels, n_cells=4)
        np.testing.assert_array_equal(counts, [3, 2, 0, 3])
        np.testing.assert_array_equal(offsets, [0, 3, 5, 5])

    def test_single_cell(self):
        labels = np.zeros(100, dtype=np.int64)
        counts, offsets = _build_counts_offsets(labels, n_cells=1)
        np.testing.assert_array_equal(counts, [100])
        np.testing.assert_array_equal(offsets, [0])

    def test_empty(self):
        labels = np.array([], dtype=np.int64)
        counts, offsets = _build_counts_offsets(labels, n_cells=5)
        np.testing.assert_array_equal(counts, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(offsets, [0, 0, 0, 0, 0])


# ---------------------------------------------------------------------------
# Cell centres
# ---------------------------------------------------------------------------


class TestComputeCellCentres:
    def test_single_cell(self):
        cdim = np.array([1, 1, 1], dtype=np.int64)
        boxsize = np.array([10.0, 10.0, 10.0])
        centres = _compute_cell_centres(cdim, boxsize, 1)
        np.testing.assert_array_equal(centres, [[5.0, 5.0, 5.0]])

    def test_rectangular(self):
        cdim = np.array([2, 1, 1], dtype=np.int64)
        boxsize = np.array([10.0, 8.0, 6.0])
        centres = _compute_cell_centres(cdim, boxsize, 2)
        expected = np.array([[2.5, 4.0, 3.0], [7.5, 4.0, 3.0]])
        np.testing.assert_array_equal(centres, expected)

    def test_multicell_c_order(self):
        cdim = np.array([2, 2, 2], dtype=np.int64)
        boxsize = np.array([10.0, 10.0, 10.0])
        centres = _compute_cell_centres(cdim, boxsize, 8)
        assert centres.shape == (8, 3)
        for i in range(8):
            assert np.all((centres[i] >= 0) & (centres[i] < boxsize))


# ---------------------------------------------------------------------------
# Cell min/max
# ---------------------------------------------------------------------------


class TestComputeCellMinmax:
    def test_basic(self):
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
            ]
        )
        counts = np.array([2, 2], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)
        min_pos, max_pos = _compute_cell_minmax(coords, counts, offsets, 2)
        np.testing.assert_array_equal(min_pos[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(max_pos[0], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(min_pos[1], [5.0, 5.0, 5.0])
        np.testing.assert_array_equal(max_pos[1], [6.0, 6.0, 6.0])

    def test_empty_cell(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        counts = np.array([0, 1], dtype=np.int64)
        offsets = np.array([0, 0], dtype=np.int64)
        min_pos, max_pos = _compute_cell_minmax(coords, counts, offsets, 2)
        np.testing.assert_array_equal(min_pos[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(max_pos[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(min_pos[1], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(max_pos[1], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_source_snapshot(
    path: Path,
    n_gas: int = 100,
    n_dm: int = 50,
    include_header: bool = True,
    include_units: bool = True,
    boxsize: float = 10.0,
):
    """Create a minimal HDF5 snapshot for testing."""
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as f:
        f.attrs["TestAttr"] = 42

        if include_header:
            hdr = f.create_group("Header")
            hdr.attrs["BoxSize"] = np.float64(boxsize)
            npart = np.zeros(6, dtype=np.int32)
            npart[0] = n_gas
            npart[1] = n_dm
            hdr.attrs["NumPart_Total"] = npart.astype(np.uint32)
            hdr.attrs["NumPart_ThisFile"] = npart.astype(np.uint32)
            hdr.attrs["NumPart_HighWord"] = np.zeros(6, dtype=np.int32)
            hdr.attrs["NumFilesPerSnapshot"] = 1
            hdr.attrs["ThisFile"] = 0
            hdr.attrs["Code"] = "Gadget2"
            hdr.attrs["TotalNumberOfParticles"] = npart.astype(np.uint64)
            hdr.attrs["RunName"] = "TestSim"

        if include_units:
            units = f.create_group("Units")
            units.attrs["Unit mass in cgs (U_M)"] = 1.989e43
            units.attrs["Unit length in cgs (U_L)"] = 3.086e21
            units.attrs["Unit time in cgs (U_t)"] = 3.156e16

        coords_gas = rng.uniform(0, boxsize, (n_gas, 3))
        masses_gas = rng.uniform(1e-5, 1e-4, n_gas)
        ids_gas = np.arange(n_gas, dtype=np.int64)

        g0 = f.create_group("PartType0")
        g0.create_dataset("Coordinates", data=coords_gas)
        g0.create_dataset("Masses", data=masses_gas)
        g0.create_dataset("ParticleIDs", data=ids_gas)

        if n_dm > 0:
            coords_dm = rng.uniform(0, boxsize, (n_dm, 3))
            masses_dm = rng.uniform(1e-3, 1e-2, n_dm)
            g1 = f.create_group("PartType1")
            g1.create_dataset("Coordinates", data=coords_dm)
            g1.create_dataset("Masses", data=masses_dm)


# ---------------------------------------------------------------------------
# Integration / end-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_translation_single_type(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        out_path = temp_dir / "output.hdf5"
        _make_source_snapshot(src_path, n_gas=200, n_dm=0)

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out_path,
            part_types=["PartType0"],
            coord_key="Coordinates",
            field_args=[
                "PartType0/Masses",
                "PartType0/ParticleIDs",
            ],
            boxsize=np.array([10.0, 10.0, 10.0]),
            cdim=np.array([4, 4, 4], dtype=np.int64),
            nthreads=1,
        )

        assert out_path.exists()
        with h5py.File(out_path, "r") as f:
            assert "Header" in f
            hdr = f["Header"]
            assert hdr.attrs["Code"] == "SWIFT"
            assert hdr.attrs["NumFilesPerSnapshot"] == 1
            assert hdr.attrs["NumPart_Total"][0] == 200

            assert "PartType0" in f
            g0 = f["PartType0"]
            assert g0["Coordinates"].shape == (200, 3)
            assert g0["Masses"].shape == (200,)
            assert g0["ParticleIDs"].shape == (200,)

            assert "Cells" in f
            cells = f["Cells"]
            assert "Meta-data" in cells
            assert cells["Meta-data"].attrs["nr_cells"] == 64
            np.testing.assert_array_equal(
                cells["Meta-data"].attrs["dimension"],
                [4, 4, 4],
            )
            assert "Centres" in cells
            assert cells["Centres"].shape == (64, 3)
            assert "Counts" in cells
            assert "PartType0" in cells["Counts"]
            assert cells["Counts/PartType0"].shape == (64,)
            assert cells["Counts/PartType0"][:].sum() == 200
            assert "OffsetsInFile" in cells
            assert "PartType0" in cells["OffsetsInFile"]
            assert "Files" in cells
            assert "PartType0" in cells["Files"]
            assert "MinPositions" in cells
            assert "PartType0" in cells["MinPositions"]
            assert "MaxPositions" in cells
            assert "PartType0" in cells["MaxPositions"]

            assert "Units" in f

    def test_two_part_types(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        out_path = temp_dir / "output.hdf5"
        _make_source_snapshot(src_path, n_gas=100, n_dm=50)

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out_path,
            part_types=["PartType0", "PartType1"],
            coord_key="Coordinates",
            field_args=[
                "PartType0/Masses",
                "PartType1/Masses",
            ],
            boxsize=np.array([10.0, 10.0, 10.0]),
            cdim=np.array([2, 2, 2], dtype=np.int64),
            nthreads=2,
        )

        with h5py.File(out_path, "r") as f:
            assert f["Header"].attrs["NumPart_Total"][0] == 100
            assert f["Header"].attrs["NumPart_Total"][1] == 50
            assert f["PartType0/Coordinates"].shape == (100, 3)
            assert f["PartType1/Coordinates"].shape == (50, 3)
            assert f["Cells/Counts/PartType0"][:].sum() == 100
            assert f["Cells/Counts/PartType1"][:].sum() == 50

    def test_cell_sorting_correct(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        out_path = temp_dir / "output.hdf5"
        _make_source_snapshot(src_path, n_gas=500, n_dm=0)

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out_path,
            part_types=["PartType0"],
            coord_key="Coordinates",
            field_args=[],
            boxsize=np.array([10.0, 10.0, 10.0]),
            cdim=np.array([4, 4, 4], dtype=np.int64),
            nthreads=1,
        )

        with h5py.File(out_path, "r") as f:
            coords = f["PartType0/Coordinates"][:]
            counts = f["Cells/Counts/PartType0"][:]
            offsets = f["Cells/OffsetsInFile/PartType0"][:]

            labels = _compute_cell_labels(
                coords,
                np.array([10.0, 10.0, 10.0]),
                np.array([4, 4, 4], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                labels, np.sort(labels, kind="stable")
            )

            for i in range(64):
                start = offsets[i]
                end = start + counts[i]
                if counts[i] > 0:
                    assert np.all(labels[start:end] == i)

    def test_no_header_in_source(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        out_path = temp_dir / "output.hdf5"
        _make_source_snapshot(src_path, n_gas=50, include_header=False)

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out_path,
            part_types=["PartType0"],
            coord_key="Coordinates",
            field_args=[],
            boxsize=np.array([10.0, 10.0, 10.0]),
            cdim=np.array([2, 2, 2], dtype=np.int64),
            nthreads=1,
        )

        with h5py.File(out_path, "r") as f:
            assert f["Header"].attrs["Code"] == "SWIFT"
            assert f["Header"].attrs["NumPart_Total"][0] == 50

    def test_no_units_in_source(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        out_path = temp_dir / "output.hdf5"
        _make_source_snapshot(src_path, n_gas=50, include_units=False)

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out_path,
            part_types=["PartType0"],
            coord_key="Coordinates",
            field_args=[],
            boxsize=np.array([10.0, 10.0, 10.0]),
            cdim=np.array([2, 2, 2], dtype=np.int64),
            nthreads=1,
        )

        with h5py.File(out_path, "r") as f:
            assert "Units" not in f

    def test_missing_part_type_raises(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        _make_source_snapshot(src_path)

        with pytest.raises(ValueError, match="not found in"):
            _translate_snapshot(
                input_paths=[src_path],
                output_path=temp_dir / "out.hdf5",
                part_types=["PartType0", "PartType2"],
                coord_key="Coordinates",
                field_args=[],
                boxsize=np.array([10.0] * 3),
                cdim=np.array([2, 2, 2], dtype=np.int64),
                nthreads=1,
            )

    def test_missing_coord_key_raises(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        _make_source_snapshot(src_path)

        with pytest.raises(ValueError, match="Coordinate field"):
            _translate_snapshot(
                input_paths=[src_path],
                output_path=temp_dir / "out.hdf5",
                part_types=["PartType0"],
                coord_key="Positions",
                field_args=[],
                boxsize=np.array([10.0] * 3),
                cdim=np.array([2, 2, 2], dtype=np.int64),
                nthreads=1,
            )

    def test_field_not_in_part_types_raises(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        _make_source_snapshot(src_path)

        with pytest.raises(ValueError, match="not declared"):
            _translate_snapshot(
                input_paths=[src_path],
                output_path=temp_dir / "out.hdf5",
                part_types=["PartType0"],
                coord_key="Coordinates",
                field_args=["PartType1/Masses"],
                boxsize=np.array([10.0] * 3),
                cdim=np.array([2, 2, 2], dtype=np.int64),
                nthreads=1,
            )

    def test_preserves_dataset_attributes(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        out_path = temp_dir / "output.hdf5"
        _make_source_snapshot(src_path, n_gas=50, n_dm=0)

        with h5py.File(src_path, "a") as f:
            f["PartType0/Masses"].attrs["units"] = "1e10 Msun"

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out_path,
            part_types=["PartType0"],
            coord_key="Coordinates",
            field_args=["PartType0/Masses"],
            boxsize=np.array([10.0, 10.0, 10.0]),
            cdim=np.array([2, 2, 2], dtype=np.int64),
            nthreads=1,
        )

        with h5py.File(out_path, "r") as f:
            assert f["PartType0/Masses"].attrs["units"] == "1e10 Msun"

    def test_rectangular_box(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        out_path = temp_dir / "output.hdf5"

        rng = np.random.default_rng(42)
        with h5py.File(src_path, "w") as f:
            hdr = f.create_group("Header")
            hdr.attrs["BoxSize"] = np.float64([10.0, 20.0, 5.0])
            npart = np.zeros(6, dtype=np.int32)
            npart[0] = 100
            hdr.attrs["NumPart_Total"] = npart.astype(np.uint32)
            hdr.attrs["NumPart_ThisFile"] = npart.astype(np.uint32)
            hdr.attrs["NumPart_HighWord"] = np.zeros(6, dtype=np.int32)

            g0 = f.create_group("PartType0")
            coords = np.column_stack(
                [
                    rng.uniform(0, 10, 100),
                    rng.uniform(0, 20, 100),
                    rng.uniform(0, 5, 100),
                ]
            )
            g0.create_dataset("Coordinates", data=coords)

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out_path,
            part_types=["PartType0"],
            coord_key="Coordinates",
            field_args=[],
            boxsize=np.array([10.0, 20.0, 5.0]),
            cdim=np.array([2, 4, 1], dtype=np.int64),
            nthreads=1,
        )

        with h5py.File(out_path, "r") as f:
            cells = f["Cells/Meta-data"]
            np.testing.assert_array_equal(cells.attrs["dimension"], [2, 4, 1])
            np.testing.assert_array_equal(cells.attrs["size"], [5.0, 5.0, 5.0])

    def test_coordinates_auto_included(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        out_path = temp_dir / "output.hdf5"
        _make_source_snapshot(src_path, n_gas=60, n_dm=0)

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out_path,
            part_types=["PartType0"],
            coord_key="Coordinates",
            field_args=["PartType0/ParticleIDs"],
            boxsize=np.array([10.0, 10.0, 10.0]),
            cdim=np.array([2, 2, 2], dtype=np.int64),
            nthreads=1,
        )

        with h5py.File(out_path, "r") as f:
            assert "Coordinates" in f["PartType0"]
            assert "ParticleIDs" in f["PartType0"]
            assert f["PartType0/Coordinates"].shape == (60, 3)

    def test_all_fields_flag(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        out_path = temp_dir / "output.hdf5"
        _make_source_snapshot(src_path, n_gas=30, n_dm=20)

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out_path,
            part_types=["PartType0", "PartType1"],
            coord_key="Coordinates",
            field_args=[],
            boxsize=np.array([10.0, 10.0, 10.0]),
            cdim=np.array([2, 2, 2], dtype=np.int64),
            nthreads=1,
            all_fields=True,
        )

        with h5py.File(out_path, "r") as f:
            assert set(f["PartType0"].keys()) == {
                "Coordinates",
                "Masses",
                "ParticleIDs",
            }
            assert set(f["PartType1"].keys()) == {
                "Coordinates",
                "Masses",
            }
            assert f["PartType0/Coordinates"].shape == (30, 3)
            assert f["PartType1/Coordinates"].shape == (20, 3)

    def test_multi_file_glob(self, temp_dir):
        """Two shard files, half the particles in each."""
        rng = np.random.default_rng(99)
        boxsize = 10.0

        # Shard 0: first 30 gas, first 15 DM
        fp0 = temp_dir / "snap_0000.hdf5"
        with h5py.File(fp0, "w") as f:
            hdr = f.create_group("Header")
            hdr.attrs["BoxSize"] = np.float64(boxsize)
            npart = np.array([30, 15, 0, 0, 0, 0], dtype=np.int32)
            hdr.attrs["NumPart_ThisFile"] = npart.astype(np.uint32)

            g0 = f.create_group("PartType0")
            coords0 = rng.uniform(0, boxsize, (30, 3))
            g0.create_dataset("Coordinates", data=coords0)
            g0.create_dataset("Masses", data=rng.uniform(0.1, 1.0, 30))

            g1 = f.create_group("PartType1")
            coords1 = rng.uniform(0, boxsize, (15, 3))
            g1.create_dataset("Coordinates", data=coords1)

        # Shard 1: next 40 gas, next 25 DM
        fp1 = temp_dir / "snap_0001.hdf5"
        with h5py.File(fp1, "w") as f:
            hdr = f.create_group("Header")
            hdr.attrs["BoxSize"] = np.float64(boxsize)
            npart = np.array([40, 25, 0, 0, 0, 0], dtype=np.int32)
            hdr.attrs["NumPart_ThisFile"] = npart.astype(np.uint32)

            g0 = f.create_group("PartType0")
            coords0b = rng.uniform(0, boxsize, (40, 3))
            g0.create_dataset("Coordinates", data=coords0b)
            g0.create_dataset("Masses", data=rng.uniform(0.1, 1.0, 40))

            g1 = f.create_group("PartType1")
            coords1b = rng.uniform(0, boxsize, (25, 3))
            g1.create_dataset("Coordinates", data=coords1b)

        out_path = temp_dir / "output.hdf5"
        glob_pattern = temp_dir / "snap_*.hdf5"

        _translate_snapshot(
            input_paths=[glob_pattern],
            output_path=out_path,
            part_types=["PartType0", "PartType1"],
            coord_key="Coordinates",
            field_args=[],
            boxsize=np.array([boxsize] * 3),
            cdim=np.array([4, 4, 4], dtype=np.int64),
            nthreads=1,
            all_fields=True,
        )

        with h5py.File(out_path, "r") as f:
            assert f["PartType0/Coordinates"].shape == (70, 3)
            assert f["PartType1/Coordinates"].shape == (40, 3)
            assert f["Cells/Counts/PartType0"][:].sum() == 70
            assert f["Cells/Counts/PartType1"][:].sum() == 40
            assert f["Header"].attrs["NumPart_Total"][0] == 70
            assert f["Header"].attrs["NumPart_Total"][1] == 40

    def test_thread_safety_nthreads(self, temp_dir):
        """Multi-threaded translation produces the same result as single."""
        src_path = temp_dir / "source.hdf5"
        out1 = temp_dir / "out_threaded.hdf5"
        out2 = temp_dir / "out_single.hdf5"
        _make_source_snapshot(src_path, n_gas=300, n_dm=0)

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out1,
            part_types=["PartType0"],
            coord_key="Coordinates",
            field_args=["PartType0/Masses"],
            boxsize=np.array([10.0, 10.0, 10.0]),
            cdim=np.array([4, 4, 4], dtype=np.int64),
            nthreads=4,
        )

        _translate_snapshot(
            input_paths=[src_path],
            output_path=out2,
            part_types=["PartType0"],
            coord_key="Coordinates",
            field_args=["PartType0/Masses"],
            boxsize=np.array([10.0, 10.0, 10.0]),
            cdim=np.array([4, 4, 4], dtype=np.int64),
            nthreads=1,
        )

        with h5py.File(out1, "r") as f1, h5py.File(out2, "r") as f2:
            np.testing.assert_array_equal(
                f1["PartType0/Coordinates"][:],
                f2["PartType0/Coordinates"][:],
            )
            np.testing.assert_array_equal(
                f1["PartType0/Masses"][:],
                f2["PartType0/Masses"][:],
            )
            np.testing.assert_array_equal(
                f1["Cells/Counts/PartType0"][:],
                f2["Cells/Counts/PartType0"][:],
            )


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


class TestCliArgs:
    def test_add_arguments(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(
            [
                "input.hdf5",
                "--output",
                "output.hdf5",
                "--part-type",
                "PartType0",
                "--part-type",
                "PartType1",
                "--coord-key",
                "Coordinates",
                "--field",
                "PartType0/Masses",
                "--field",
                "PartType1/Velocities",
                "--boxsize",
                "10",
                "--cdim",
                "4",
                "4",
                "4",
                "--nthreads",
                "2",
            ]
        )
        assert args.input == [Path("input.hdf5")]
        assert args.output == Path("output.hdf5")
        assert args.part_type == ["PartType0", "PartType1"]
        assert args.coord_key == "Coordinates"
        assert args.field == [
            "PartType0/Masses",
            "PartType1/Velocities",
        ]
        assert args.boxsize == ["10"]
        assert args.cdim == ["4", "4", "4"]
        assert args.nthreads == 2
        assert args.all_fields is False

    def test_add_arguments_all_fields(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(
            [
                "input.hdf5",
                "--output",
                "output.hdf5",
                "--part-type",
                "PartType0",
                "--coord-key",
                "Coordinates",
                "--all-fields",
                "--boxsize",
                "10",
                "--cdim",
                "4",
                "4",
                "4",
            ]
        )
        assert args.all_fields is True

    def test_glob_input_arg(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(
            [
                "snapdir/snapshot_*.hdf5",
                "--output",
                "output.hdf5",
                "--part-type",
                "PartType0",
                "--coord-key",
                "Coordinates",
                "--boxsize",
                "10",
                "--cdim",
                "4",
                "4",
                "4",
            ]
        )
        assert args.input == [Path("snapdir/snapshot_*.hdf5")]

    def test_run_no_part_types(self, temp_dir):
        src_path = temp_dir / "source.hdf5"
        _make_source_snapshot(src_path, n_gas=10)
        out_path = temp_dir / "output.hdf5"

        args = argparse.Namespace(
            input=[src_path],
            output=out_path,
            part_type=[],
            coord_key="Coordinates",
            field=["PartType0/Masses"],
            boxsize=["10"],
            cdim=["2", "2", "2"],
            nthreads=1,
            all_fields=False,
        )
        with pytest.raises(SystemExit):
            run(args)
