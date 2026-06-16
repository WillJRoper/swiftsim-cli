"""Translate mode for converting external snapshots to SWIFT format.

This mode reads an external HDF5 snapshot (single file or glob for
distributed snapshots), reorders particles into SWIFT's top-level cell
structure, and writes a SWIFT-compliant snapshot file with Cell
hash-table metadata that can be read by swiftsimio and other SWIFT
tooling.
"""

import argparse
import glob
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from socket import gethostname
from typing import Dict, List, Tuple

import h5py  # type: ignore[import-untyped]
import numpy as np

_GADGET_PARTTYPE_NAMES: Dict[int, str] = {
    0: "Gas",
    1: "Dark Matter",
    2: "Background Dark Matter",
    3: "Sinks",
    4: "Stars",
    5: "Black Holes",
    6: "Neutrino Dark Matter",
}
_GADGET_NUM_PART_TYPES = 6


def _parse_field_arg(arg: str) -> Tuple[str, str]:
    """Parse ``PartTypeN/FieldName`` into *(part_type, field_name)*.

    Raises:
    ------
    argparse.ArgumentTypeError
        If *arg* does not match the expected format.
    """
    stripped = arg.lstrip("/")
    parts = stripped.split("/")
    if len(parts) != 2 or not parts[0].startswith("PartType"):
        raise argparse.ArgumentTypeError(
            f"Invalid --field argument '{arg}'. "
            f"Expected format: PartTypeN/FieldName"
        )
    return parts[0], parts[1]


def _parse_fields(
    field_args: List[str],
    part_types: List[str],
) -> Dict[str, List[str]]:
    """Parse repeated ``--field`` args into {part_type: [field_names]}.

    Every *field_args* entry is validated against *part_types*.
    """
    fields: Dict[str, List[str]] = {}
    for arg in field_args:
        pt, name = _parse_field_arg(arg)
        if pt not in part_types:
            raise ValueError(
                f"--field '{arg}' references '{pt}' which was not "
                f"declared with --part-type"
            )
        fields.setdefault(pt, [])
        if name not in fields[pt]:
            fields[pt].append(name)
    return fields


def _parse_ndim(
    name: str,
    vals: List[str],
    dim: int,
    dtype: type = float,
) -> np.ndarray:
    """Parse a scalar or N-vector CLI argument into a NumPy array.

    *vals* must contain either 1 element (broadcast to *dim*) or exactly
    *dim* elements.
    """
    if len(vals) == 1:
        return np.full(dim, dtype(vals[0]), dtype=dtype)
    if len(vals) == dim:
        return np.array(vals, dtype=dtype)
    raise argparse.ArgumentTypeError(
        f"--{name} expects 1 or {dim} value(s), got {len(vals)}"
    )


def _resolve_input_files(input_path: Path) -> List[Path]:
    """Expand *input_path* to a sorted list of concrete HDF5 files.

    When *input_path* contains glob wildcards (``*``, ``?``, ``[``) the
    pattern is expanded with :func:`glob.glob`.  Otherwise the single path
    is returned as-is.
    """
    pattern = str(input_path)
    if any(ch in pattern for ch in "*?["):
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No files matched the glob pattern '{pattern}'"
            )
        return [Path(m) for m in matches]
    return [input_path]


def _compute_cell_labels(
    coords: np.ndarray,
    boxsize: np.ndarray,
    cdim: np.ndarray,
) -> np.ndarray:
    """Assign each particle to a top-level cell with periodic wrapping.

    Cell indices follow SWIFT's C-style ordering (z fastest, x slowest).

    Parameters
    ----------
    coords : (N, 3) float64
        Particle positions in the same units as *boxsize*.
    boxsize : (3,) float64
        Simulation box dimensions.
    cdim : (3,) int64
        Number of top-level cells along each axis.

    Returns:
    -------
    labels : (N,) int64
        Flat cell index for each particle, ranging ``[0, prod(cdim))``.
    """
    norm = (coords / boxsize) % 1.0
    cell_ix = np.floor(norm * cdim).astype(np.int64)
    np.clip(cell_ix, 0, cdim - 1, out=cell_ix)
    ny, nz = cdim[1], cdim[2]
    return cell_ix[:, 0] * (ny * nz) + cell_ix[:, 1] * nz + cell_ix[:, 2]


def _build_counts_offsets(
    cell_labels: np.ndarray,
    n_cells: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive per-cell particle counts and file offsets from sorted labels.

    Parameters
    ----------
    cell_labels : (N,) int64
        Cell labels for particles already sorted by cell index.
    n_cells : int
        Total number of top-level cells.

    Returns:
    -------
    counts : (n_cells,) int64
        Number of particles in each cell.
    offsets : (n_cells,) int64
        Starting offset of each cell in the sorted particle array.
    """
    counts = np.bincount(cell_labels, minlength=n_cells)
    offsets = np.zeros(n_cells, dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    return counts, offsets


def _compute_cell_centres(
    cdim: np.ndarray,
    boxsize: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Return the centre coordinate of every top-level cell (C-order).

    Returns (n_cells, 3) float64 array.
    """
    cell_size = boxsize / cdim
    ix, iy, iz = np.unravel_index(
        np.arange(n_cells, dtype=np.int64), cdim, order="C"
    )
    return np.column_stack(
        [
            (ix + 0.5) * cell_size[0],
            (iy + 0.5) * cell_size[1],
            (iz + 0.5) * cell_size[2],
        ]
    )


def _compute_cell_minmax(
    sorted_coords: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    n_cells: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-cell bounding-box coordinates from sorted positions."""
    min_pos = np.full((n_cells, 3), np.inf, dtype=np.float64)
    max_pos = np.full((n_cells, 3), -np.inf, dtype=np.float64)

    for i in range(n_cells):
        cnt = counts[i]
        if cnt == 0:
            min_pos[i] = 0.0
            max_pos[i] = 0.0
            continue
        start = offsets[i]
        chunk = sorted_coords[start : start + cnt]
        min_pos[i] = chunk.min(axis=0)
        max_pos[i] = chunk.max(axis=0)

    return min_pos, max_pos


def _process_part_type_arrays(
    datasets: Dict[str, np.ndarray],
    coord_field: str,
    boxsize: np.ndarray,
    cdim: np.ndarray,
) -> Tuple[
    Dict[str, np.ndarray],  # reordered datasets
    np.ndarray,  # sorted coords
    np.ndarray,  # sorted cell_labels
]:
    """Cell-sort pre-loaded per-part-type data (pure NumPy – thread-safe).

    Parameters
    ----------
    datasets : dict[str, ndarray]
        All field arrays for one particle type; all must have the same
        leading dimension *N*.
    coord_field : str
        Key in *datasets* that holds the (N, 3) coordinate array.
    boxsize : (3,) ndarray
    cdim : (3,) ndarray

    Returns:
    -------
    reordered : dict[str, ndarray]
        Cell-sorted copies of every dataset in *datasets*.
    sorted_coords : (N, 3) ndarray
    sorted_labels : (N,) int64 ndarray
    """
    coords = datasets[coord_field]
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"Coordinate field '{coord_field}' must be "
            f"(N, 3); got shape {coords.shape}"
        )

    labels = _compute_cell_labels(coords, boxsize, cdim)
    sort_order = np.argsort(labels, kind="stable")

    reordered: Dict[str, np.ndarray] = {}
    for field_name, data in datasets.items():
        reordered[field_name] = data[sort_order]

    return reordered, coords[sort_order], labels[sort_order]


def _copy_or_synthesise_header(
    src: h5py.File,
    out: h5py.File,
    boxsize: np.ndarray,
    npart: np.ndarray,
):
    """Copy the source ``/Header`` or create a minimal one.

    *npart* is a (6,) int32 array with total particle counts per type.
    """
    if "Header" in src:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            src.copy("Header", out, "Header")
        hdr = out["Header"]
    else:
        hdr = out.create_group("Header")

    hdr.attrs["Code"] = "SWIFT"
    hdr.attrs["BoxSize"] = boxsize
    hdr.attrs["NumFilesPerSnapshot"] = 1
    hdr.attrs["ThisFile"] = 0

    for attr in (
        "NumPart_Total",
        "NumPart_ThisFile",
        "NumPart_HighWord",
    ):
        if attr not in hdr.attrs:
            hdr.attrs[attr] = np.zeros(6, dtype=np.int32)  # type: ignore[assignment]

    hdr.attrs["NumPart_Total"] = npart.astype(np.uint32)
    hdr.attrs["NumPart_ThisFile"] = npart.astype(np.uint32)

    for attr in ("NumPart_HighWord",):
        if attr not in hdr.attrs or len(hdr.attrs[attr]) < 6:
            hdr.attrs[attr] = np.zeros(6, dtype=np.int32)  # type: ignore[assignment]

    total_field = "TotalNumberOfParticles"
    if total_field in hdr.attrs:
        hdr.attrs[total_field] = npart.astype(np.uint64)
    else:
        hdr.attrs.create(total_field, npart.astype(np.uint64))

    used_types = [i for i in range(6) if npart[i] > 0]
    part_type_names = [
        _GADGET_PARTTYPE_NAMES.get(i, f"PartType{i}") for i in used_types
    ]
    if "NumPartTypes" in hdr.attrs:
        hdr.attrs["NumPartTypes"] = len(used_types)
    else:
        hdr.attrs["NumPartTypes"] = len(used_types)

    names_key = "PartTypeNames"
    names_arr = np.array(part_type_names, dtype=h5py.string_dtype())
    if names_key in hdr:
        del hdr[names_key]
    hdr.create_dataset(names_key, data=names_arr)

    if "RunName" not in hdr.attrs:
        hdr.attrs["RunName"] = "Translated Snapshot"
    if "SnapshotDate" not in hdr.attrs:
        hdr.attrs["SnapshotDate"] = datetime.now().isoformat()
    if "System" not in hdr.attrs:
        hdr.attrs["System"] = gethostname()


def _write_cells_group(
    out: h5py.File,
    cdim: np.ndarray,
    boxsize: np.ndarray,
    n_cells: int,
    centres: np.ndarray,
    part_type_info: Dict[
        str,
        Tuple[
            np.ndarray,  # counts
            np.ndarray,  # offsets
            np.ndarray,  # sorted_coords
        ],
    ],
):
    """Write the SWIFT ``/Cells`` group including per-type metadata."""
    cells = out.create_group("Cells")

    meta = cells.create_group("Meta-data")
    meta.attrs["dimension"] = cdim
    meta.attrs["size"] = boxsize / cdim
    meta.attrs["nr_cells"] = n_cells

    cells.create_dataset("Centres", data=centres)

    files_grp = cells.create_group("Files")
    counts_grp = cells.create_group("Counts")
    offsets_grp = cells.create_group("OffsetsInFile")
    min_grp = cells.create_group("MinPositions")
    max_grp = cells.create_group("MaxPositions")

    for pt, (counts, offsets, sorted_coords) in part_type_info.items():
        files_grp.create_dataset(pt, data=np.zeros(n_cells, dtype=np.int32))
        counts_grp.create_dataset(pt, data=counts)
        offsets_grp.create_dataset(pt, data=offsets)

        min_pos, max_pos = _compute_cell_minmax(
            sorted_coords, counts, offsets, n_cells
        )
        min_grp.create_dataset(pt, data=min_pos)
        max_grp.create_dataset(pt, data=max_pos)


def _translate_snapshot(
    input_path: Path,
    output_path: Path,
    part_types: List[str],
    coord_key: str,
    field_args: List[str],
    boxsize: np.ndarray,
    cdim: np.ndarray,
    nthreads: int,
    all_fields: bool = False,
):
    """Orchestrate the full snapshot translation.

    Parameters
    ----------
    input_path : Path
        Source HDF5 snapshot path.  May be a single file or a glob
        pattern (e.g. ``snapdir/snapshot_*.hdf5``) to combine
        distributed shards.
    output_path : Path
        Destination HDF5 file (overwritten if it exists).
    part_types : list[str]
        HDF5 group names to include (e.g. ``["PartType0"]``).
    coord_key : str
        Dataset name used for coordinates inside each *part_types* group.
    field_args : list[str]
        Raw ``--field`` values (e.g. ``["PartType0/Density"]``).
    boxsize : (3,) ndarray
        Box size in whatever units the snapshot uses.
    cdim : (3,) ndarray
        Top-level cell grid dimensions.
    nthreads : int
        Number of threads for per-part-type parallel processing.
    all_fields : bool
        When True, copy every dataset from each part type group
        instead of only those listed in *field_args*.
    """
    fields = _parse_fields(field_args, part_types)

    for pt in part_types:
        if pt not in fields:
            fields[pt] = []

    n_cells = int(np.prod(cdim))

    if nthreads < 1:
        nthreads = 1

    # ---- resolve input files ----
    input_files = _resolve_input_files(input_path)
    n_files = len(input_files)
    print(f"Found {n_files} source file(s) matching: {input_path}")

    # ---- peek at first file for field discovery & metadata ----
    with h5py.File(input_files[0], "r") as peek:
        metadata_groups = sorted(
            name
            for name in peek.keys()
            if name not in ("Header", "Cells", "Units")
            and not name.startswith("PartType")
        )
        for pt in part_types:
            if pt not in peek:
                raise ValueError(
                    f"Group '{pt}' not found in '{input_files[0]}'"
                )
            if all_fields:
                fields[pt] = sorted(peek[pt].keys())
            if coord_key not in peek[pt]:
                raise ValueError(
                    f"Coordinate field '{coord_key}' not found in '{pt}'"
                )

    # Ensure coord_key is always included
    for pt in part_types:
        if coord_key not in fields[pt]:
            fields[pt].insert(0, coord_key)

    # ---- read all particle data in the main thread ----
    # all_datasets : {part_type: {field_name: list_of_arrays}}
    all_datasets: Dict[str, Dict[str, List[np.ndarray]]] = {
        pt: {fn: [] for fn in fields[pt]} for pt in part_types
    }

    for fp in input_files:
        print(f"  Reading: {fp}")
        with h5py.File(fp, "r") as src:
            for pt in part_types:
                if pt not in src:
                    continue
                for fn in fields[pt]:
                    if fn in src[pt]:
                        all_datasets[pt][fn].append(src[pt][fn][:])

    # Validate: coordinate field must have been found in at least one file
    for pt in part_types:
        if not all_datasets[pt].get(coord_key):
            raise ValueError(
                f"Coordinate field '{coord_key}' not found in '{pt}' "
                f"across {n_files} source file(s)"
            )

    # ---- concatenate per-part-type arrays ----
    concatenated: Dict[str, Dict[str, np.ndarray]] = {}
    for pt in part_types:
        concatenated[pt] = {}
        for fn in fields[pt]:
            parts = all_datasets[pt].get(fn, [])
            concatenated[pt][fn] = (
                np.concatenate(parts) if parts else np.array([])
            )

    # ---- process each part type (parallel; pure NumPy – thread-safe) ----
    results: Dict[
        str,
        Tuple[
            Dict[str, np.ndarray],  # reordered
            np.ndarray,  # sorted_coords
            np.ndarray,  # sorted_labels
        ],
    ] = {}

    if nthreads > 1:
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            futures = {}
            for pt in part_types:
                f = executor.submit(
                    _process_part_type_arrays,
                    concatenated[pt],
                    coord_key,
                    boxsize,
                    cdim,
                )
                futures[f] = pt
            for f in futures:
                results[futures[f]] = f.result()
    else:
        for pt in part_types:
            results[pt] = _process_part_type_arrays(
                concatenated[pt],
                coord_key,
                boxsize,
                cdim,
            )

    # ---- build cell metadata per part type ----
    part_type_info: Dict[
        str,
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for pt in part_types:
        _, sorted_coords, sorted_labels = results[pt]
        counts, offsets = _build_counts_offsets(sorted_labels, n_cells)
        part_type_info[pt] = (counts, offsets, sorted_coords)

    centres = _compute_cell_centres(cdim, boxsize, n_cells)

    # ---- particle counts for the header ----
    npart = np.zeros(_GADGET_NUM_PART_TYPES, dtype=np.int32)
    for pt in part_types:
        pidx = int(pt.replace("PartType", ""))
        npart[pidx] = len(list(concatenated[pt].values())[0])

    # ---- write output ----
    print(f"Writing translated snapshot to: {output_path}")
    with h5py.File(input_files[0], "r") as src_meta, h5py.File(
        output_path, "w"
    ) as out:
        if "Units" in src_meta:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                src_meta.copy("Units", out, "Units")

        for name in metadata_groups:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    src_meta.copy(name, out, name)
            except Exception:
                print(f"  Warning: could not copy group '{name}'")

        _copy_or_synthesise_header(src_meta, out, boxsize, npart)

        # Particle data (cell-sorted)
        for pt in part_types:
            grp = out.create_group(pt)
            reordered, _, _ = results[pt]
            for fn in fields[pt]:
                data = reordered[fn]
                grp.create_dataset(fn, data=data)

            # Restore per-dataset attributes from first file
            if pt in src_meta:
                for fn in fields[pt]:
                    if fn in src_meta[pt]:
                        for k, v in src_meta[pt][fn].attrs.items():
                            grp[fn].attrs[k] = v

        _write_cells_group(
            out,
            cdim,
            boxsize,
            n_cells,
            centres,
            part_type_info,
        )

    nfields = sum(len(v) for v in fields.values())
    print(
        f"Done. Translated {n_files} file(s), {len(part_types)} particle "
        f"type(s), {nfields} field(s), {n_cells} cells."
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the ``translate`` mode."""
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to the input HDF5 snapshot to translate "
            "(accepts globs, e.g. 'snapdir/snapshot_*.hdf5')."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path for the output SWIFT-compliant snapshot.",
    )
    parser.add_argument(
        "--part-type",
        action="append",
        default=[],
        metavar="GROUP",
        help=(
            "HDF5 particle group to include (e.g. PartType0). May be repeated."
        ),
    )
    parser.add_argument(
        "--coord-key",
        type=str,
        required=True,
        help=(
            "Dataset name for coordinates within each particle group "
            "(e.g. Coordinates). Applied to all --part-type entries."
        ),
    )
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Dataset to include, relative to a --part-type group "
            "(e.g. PartType0/Density). May be repeated."
        ),
    )
    parser.add_argument(
        "--boxsize",
        nargs="+",
        required=True,
        help="Simulation box size (scalar or 3 floats).",
    )
    parser.add_argument(
        "--cdim",
        nargs="+",
        required=True,
        help="Top-level cell grid dimensions (scalar or 3 ints).",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="Number of threads for parallel processing (default: 1).",
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        default=False,
        help="Copy every dataset from each --part-type group.",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the translate mode."""
    boxsize = _parse_ndim("boxsize", args.boxsize, 3, float)
    cdim = _parse_ndim("cdim", args.cdim, 3, int)

    if not args.part_type:
        print(
            "No --part-type arguments provided. "
            "At least one particle type is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    _translate_snapshot(
        input_path=args.input,
        output_path=args.output,
        part_types=args.part_type,
        coord_key=args.coord_key,
        field_args=args.field,
        boxsize=boxsize,
        cdim=cdim,
        nthreads=args.nthreads,
        all_fields=args.all_fields,
    )
