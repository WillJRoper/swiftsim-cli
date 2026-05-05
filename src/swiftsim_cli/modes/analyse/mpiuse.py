"""MPI communication usage analysis for SWIFT mpiuse reports."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from pandas.errors import EmptyDataError  # type: ignore[import-untyped]

from swiftsim_cli.utilities import create_ascii_table, create_output_path

MPIUSE_COLUMNS = [
    "stic",
    "etic",
    "dtic",
    "step",
    "rank",
    "otherrank",
    "type",
    "itype",
    "subtype",
    "isubtype",
    "activation",
    "tag",
    "size",
    "sum",
]

MPIUSE_FILENAME_RE = re.compile(
    r"mpiuse_report-rank(?P<rank>\d+)-step(?P<step>\d+)\.dat$"
)


class MpiuseData(TypedDict):
    """Type definition for aggregated mpiuse input data."""

    input_path: str
    label: str
    files: list[str]
    skipped_files: int
    ranks: list[int]
    steps: list[int]
    total_bytes: int
    total_count: int
    per_step_detail: pd.DataFrame
    per_step_totals: pd.DataFrame
    per_subtype: pd.DataFrame
    inflight_per_step: pd.DataFrame
    per_rank_detail: pd.DataFrame | None
    per_rank_totals: pd.DataFrame | None


__all__ = [
    "add_mpiuse_arguments",
    "run_swift_mpiuse",
    "analyse_swift_mpiuse",
]


def add_mpiuse_arguments(subparsers) -> None:
    """Add CLI arguments for SWIFT mpiuse report analysis."""
    mpiuse_parser = subparsers.add_parser(
        "mpiuse",
        help=(
            "Analyse SWIFT mpiuse_report-rank*-step*.dat files and "
            "compare communication volume across one or more inputs."
        ),
    )

    mpiuse_parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Input directories, glob patterns, or individual "
            "mpiuse_report files to analyse."
        ),
        type=str,
    )
    mpiuse_parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        help="Labels for each input (same order as inputs).",
        type=str,
        default=None,
    )
    mpiuse_parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Where to save analysis outputs (default: current directory).",
        default=None,
    )
    mpiuse_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="Prefix for output files and output directory.",
        default=None,
    )
    mpiuse_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively.",
        default=False,
    )
    mpiuse_parser.add_argument(
        "--subtypes",
        nargs="+",
        type=str,
        help="Optional subtype filter (for example: gpart grav_counts).",
        default=None,
    )
    mpiuse_parser.add_argument(
        "--steps",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive step range to analyse.",
        default=None,
    )
    mpiuse_parser.add_argument(
        "--per-rank",
        action="store_true",
        help="Also produce per-rank breakdown tables and plots.",
        default=False,
    )
    mpiuse_parser.add_argument(
        "--format",
        choices=("ascii", "csv", "both"),
        default="ascii",
        help="Write ASCII tables, CSV output, or both.",
    )


def run_swift_mpiuse(args: argparse.Namespace) -> None:
    """Entry point for the ``mpiuse`` CLI subcommand."""
    step_range = None
    if args.steps is not None:
        step_range = (args.steps[0], args.steps[1])

    analyse_swift_mpiuse(
        inputs=args.inputs,
        labels=args.labels,
        output_path=str(args.output_path) if args.output_path else None,
        prefix=args.prefix,
        show_plot=args.show,
        subtype_filter=args.subtypes,
        step_range=step_range,
        per_rank=args.per_rank,
        output_format=args.format,
    )


def analyse_swift_mpiuse(
    inputs: list[str],
    labels: list[str] | None = None,
    output_path: str | None = None,
    prefix: str | None = None,
    show_plot: bool = True,
    subtype_filter: list[str] | None = None,
    step_range: tuple[int, int] | None = None,
    per_rank: bool = False,
    output_format: str = "ascii",
) -> None:
    """Analyse SWIFT mpiuse report files.

    Args:
        inputs: Input directories, glob patterns, or files to analyse.
        labels: Optional labels matching ``inputs``.
        output_path: Directory where output files should be written.
        prefix: Optional prefix for files and output directory.
        show_plot: Whether to display plots interactively.
        subtype_filter: Optional subtype names to include.
        step_range: Optional inclusive ``(start, end)`` step range.
        per_rank: Whether to produce per-rank summaries and plots.
        output_format: ``ascii``, ``csv``, or ``both``.
    """
    print(f"Analyzing mpiuse reports from {len(inputs)} input(s)")

    if labels is not None and len(labels) != len(inputs):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of "
            f"inputs ({len(inputs)})"
        )

    if step_range is not None and step_range[0] > step_range[1]:
        raise ValueError("Step range must be ordered as START END")

    if labels is None:
        labels = [_derive_label(input_path) for input_path in inputs]

    out_dir = (
        "mpiuse_analysis" if prefix is None else f"{prefix}_mpiuse_analysis"
    )
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    cmap = plt.get_cmap("tab20")

    all_data: list[MpiuseData] = []
    csv_rows: list[pd.DataFrame] = []

    for input_path, label in zip(inputs, labels):
        files = _discover_files(input_path)
        if not files:
            print(f"Warning: no mpiuse files matched input '{input_path}'")
            continue

        aggregated = _aggregate(
            files=files,
            step_range=step_range,
            subtype_filter=subtype_filter,
            per_rank=per_rank,
        )
        if aggregated is None:
            print(
                f"Warning: no usable mpiuse data remained for '{label}' "
                "after filtering"
            )
            continue

        data: MpiuseData = {
            "input_path": input_path,
            "label": label,
            "files": [str(path) for path in aggregated["files"]],
            "skipped_files": int(aggregated["skipped_files"]),
            "ranks": aggregated["ranks"],
            "steps": aggregated["steps"],
            "total_bytes": int(aggregated["total_bytes"]),
            "total_count": int(aggregated["total_count"]),
            "per_step_detail": aggregated["per_step_detail"],
            "per_step_totals": aggregated["per_step_totals"],
            "per_subtype": aggregated["per_subtype"],
            "inflight_per_step": aggregated["inflight_per_step"],
            "per_rank_detail": aggregated["per_rank_detail"],
            "per_rank_totals": aggregated["per_rank_totals"],
        }
        all_data.append(data)

        headline = _format_headline(data)
        summary_table = _format_summary_table(data)
        output_text = f"{headline}\n\n{summary_table}\n"

        print()
        print(headline)
        print(summary_table)

        table_path = create_output_path(
            output_path,
            prefix,
            f"mpiuse_summary_{_slugify(label)}.txt",
            out_dir,
        )
        table_path.write_text(output_text, encoding="utf-8")

        csv_rows.append(
            data["per_step_detail"].assign(input=input_path, label=label)
        )

        if per_rank and data["per_rank_detail"] is not None:
            rank_table = _format_per_rank_table(data)
            print()
            print(rank_table)
            rank_path = create_output_path(
                output_path,
                prefix,
                f"mpiuse_per_rank_{_slugify(label)}.txt",
                out_dir,
            )
            rank_path.write_text(rank_table + "\n", encoding="utf-8")

    if not all_data:
        print("No usable mpiuse data found in any input")
        return

    if len(all_data) >= 2:
        comparison_table = _format_comparison_table(all_data)
        print()
        print(comparison_table)
        comparison_path = create_output_path(
            output_path,
            prefix,
            "mpiuse_comparison.txt",
            out_dir,
        )
        comparison_path.write_text(comparison_table + "\n", encoding="utf-8")

    if output_format in {"csv", "both"}:
        _write_csv(csv_rows, output_path, prefix, out_dir)

    colors = cmap(np.linspace(0, 1, max(1, len(all_data))))

    _plot_total_per_step(
        all_data,
        colors,
        markers,
        output_path,
        prefix,
        out_dir,
        show_plot,
        value_column="bytes",
        filename="mpiuse_total_bytes_per_step.png",
        ylabel="Total bytes per step",
        title="MPI communication volume per step",
        force_log=True,
    )
    _plot_total_per_step(
        all_data,
        colors,
        markers,
        output_path,
        prefix,
        out_dir,
        show_plot,
        value_column="count",
        filename="mpiuse_total_count_per_step.png",
        ylabel="MPI requests per step",
        title="MPI request count per step",
        force_log=False,
    )
    _plot_subtype_bars(
        all_data,
        colors,
        output_path,
        prefix,
        out_dir,
        show_plot,
        value_column="bytes",
        filename="mpiuse_bytes_by_subtype.png",
        ylabel="Total bytes",
        title="MPI bytes by subtype",
    )
    _plot_subtype_bars(
        all_data,
        colors,
        output_path,
        prefix,
        out_dir,
        show_plot,
        value_column="count",
        filename="mpiuse_count_by_subtype.png",
        ylabel="Request count",
        title="MPI request count by subtype",
    )
    if _has_gpart(all_data):
        _plot_gpart_per_step(
            all_data,
            colors,
            markers,
            output_path,
            prefix,
            out_dir,
            show_plot,
        )
    _plot_inflight_max(
        all_data,
        colors,
        markers,
        output_path,
        prefix,
        out_dir,
        show_plot,
    )
    if per_rank:
        _plot_per_rank_totals(
            all_data,
            output_path,
            prefix,
            out_dir,
            show_plot,
        )


def _derive_label(input_path: str) -> str:
    path = Path(input_path)
    if path.exists() and path.is_dir():
        return path.name
    if path.exists() and path.is_file():
        return path.stem
    if path.name:
        return path.name
    return input_path


def _discover_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    pattern = str(input_path)
    has_glob = any(char in pattern for char in "*?[")

    if path.exists():
        if path.is_dir():
            return sorted(path.glob("mpiuse_report-rank*-step*.dat"))
        if path.is_file() and MPIUSE_FILENAME_RE.search(path.name) is not None:
            return [path]
        return []

    if has_glob:
        parent = path.parent if str(path.parent) != "" else Path.cwd()
        return sorted(
            match
            for match in parent.glob(path.name)
            if match.is_file()
            and MPIUSE_FILENAME_RE.search(match.name) is not None
        )

    return []


def _parse_rank_step(path: Path) -> tuple[int, int]:
    match = MPIUSE_FILENAME_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse rank/step from filename: {path}")
    return int(match.group("rank")), int(match.group("step"))


def _read_mpiuse_raw_file(path: Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            names=MPIUSE_COLUMNS,
            header=None,
            dtype={
                "stic": np.int64,
                "etic": np.int64,
                "dtic": np.int64,
                "step": np.int64,
                "rank": np.int64,
                "otherrank": np.int64,
                "type": str,
                "itype": np.int64,
                "subtype": str,
                "isubtype": np.int64,
                "activation": np.int64,
                "tag": np.int64,
                "size": np.int64,
                "sum": np.int64,
            },
        )
    except EmptyDataError:
        data = pd.DataFrame(columns=MPIUSE_COLUMNS)

    if data.empty:
        return pd.DataFrame(columns=MPIUSE_COLUMNS)

    _parse_rank_step(path)
    return data


def _read_mpiuse_file(path: Path) -> pd.DataFrame:
    raw = _read_mpiuse_raw_file(path)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "step",
                "rank",
                "otherrank",
                "type",
                "subtype",
                "tag",
                "size",
                "dtic",
            ]
        )

    # Each request is logged once at activation and once at handoff/free.
    # Keep activation rows only for volume statistics to avoid cancellation.
    return raw.loc[
        raw["activation"] == 1,
        [
            "step",
            "rank",
            "otherrank",
            "type",
            "subtype",
            "tag",
            "size",
            "dtic",
        ],
    ].copy()


def _aggregate(
    files: list[Path],
    step_range: tuple[int, int] | None,
    subtype_filter: list[str] | None,
    per_rank: bool,
) -> dict[str, Any] | None:
    activation_frames: list[pd.DataFrame] = []
    inflight_frames: list[pd.DataFrame] = []
    used_files: list[Path] = []
    skipped_files = 0

    for path in files:
        raw = _read_mpiuse_raw_file(path)
        if raw.empty:
            skipped_files += 1
            continue

        if step_range is not None:
            raw = raw.loc[
                (raw["step"] >= step_range[0]) & (raw["step"] <= step_range[1])
            ].copy()

        if raw.empty:
            skipped_files += 1
            continue

        activation = raw.loc[raw["activation"] == 1].copy()
        if subtype_filter is not None:
            activation = activation.loc[
                activation["subtype"].isin(subtype_filter)
            ].copy()

        if activation.empty:
            skipped_files += 1
            continue

        activation_frames.append(activation)
        inflight_frames.append(
            raw.groupby(["step", "rank"], as_index=False)["sum"]
            .max()
            .rename(columns={"sum": "inflight_max"})
        )
        used_files.append(path)

    if not activation_frames:
        return None

    activation_data = pd.concat(activation_frames, ignore_index=True)
    inflight_data = pd.concat(inflight_frames, ignore_index=True)

    per_step_detail = (
        activation_data.groupby(["step", "type", "subtype"], as_index=False)
        .agg(bytes=("size", "sum"), count=("size", "size"))
        .sort_values(["step", "bytes"], ascending=[True, False])
        .reset_index(drop=True)
    )
    per_step_totals = (
        activation_data.groupby("step", as_index=False)
        .agg(bytes=("size", "sum"), count=("size", "size"))
        .sort_values("step")
        .reset_index(drop=True)
    )
    per_subtype = (
        activation_data.groupby(["type", "subtype"], as_index=False)
        .agg(
            bytes=("size", "sum"),
            count=("size", "size"),
            mean_bytes=("size", "mean"),
            max_bytes=("size", "max"),
        )
        .sort_values("bytes", ascending=False)
        .reset_index(drop=True)
    )
    inflight_per_step = (
        inflight_data.groupby("step", as_index=False)["inflight_max"]
        .max()
        .sort_values("step")
        .reset_index(drop=True)
    )

    per_rank_detail: pd.DataFrame | None = None
    per_rank_totals: pd.DataFrame | None = None
    if per_rank:
        per_rank_detail = (
            activation_data.groupby(
                ["rank", "type", "subtype"], as_index=False
            )
            .agg(bytes=("size", "sum"), count=("size", "size"))
            .sort_values(["rank", "bytes"], ascending=[True, False])
            .reset_index(drop=True)
        )
        per_rank_totals = (
            activation_data.groupby("rank", as_index=False)
            .agg(bytes=("size", "sum"), count=("size", "size"))
            .sort_values("rank")
            .reset_index(drop=True)
        )

    return {
        "files": used_files,
        "skipped_files": skipped_files,
        "ranks": sorted(
            int(value) for value in activation_data["rank"].unique()
        ),
        "steps": sorted(
            int(value) for value in activation_data["step"].unique()
        ),
        "total_bytes": int(activation_data["size"].sum()),
        "total_count": int(len(activation_data)),
        "per_step_detail": per_step_detail,
        "per_step_totals": per_step_totals,
        "per_subtype": per_subtype,
        "inflight_per_step": inflight_per_step,
        "per_rank_detail": per_rank_detail,
        "per_rank_totals": per_rank_totals,
    }


def _format_headline(data: MpiuseData) -> str:
    step_min = min(data["steps"])
    step_max = max(data["steps"])
    return (
        f"mpiuse summary: {data['label']}\n"
        f"Files parsed: {len(data['files'])} "
        f"(skipped: {data['skipped_files']}), "
        f"steps: {step_min}-{step_max}, ranks: {data['ranks']}\n"
        f"Total bytes communicated: {data['total_bytes']:,} "
        f"({data['total_bytes'] / 1024**2:.3f} MB)\n"
        f"Total MPI request count: {data['total_count']:,}"
    )


def _format_summary_table(data: MpiuseData) -> str:
    total_bytes = max(data["total_bytes"], 1)
    headers = [
        "Type",
        "Subtype",
        "Count",
        "Total bytes",
        "Total MB",
        "Mean bytes",
        "Max bytes",
        "% of total",
    ]
    rows: list[list[str]] = []
    for row in data["per_subtype"].itertuples(index=False):
        rows.append(
            [
                str(row.type),
                str(row.subtype),
                f"{int(row.count):,}",
                f"{int(row.bytes):,}",
                f"{float(row.bytes) / 1024**2:.3f}",
                f"{float(row.mean_bytes):.1f}",
                f"{int(row.max_bytes):,}",
                f"{100.0 * float(row.bytes) / total_bytes:.2f}",
            ]
        )

    rows.append(
        [
            "TOTAL",
            "",
            f"{data['total_count']:,}",
            f"{data['total_bytes']:,}",
            f"{data['total_bytes'] / 1024**2:.3f}",
            "-",
            "-",
            "100.00",
        ]
    )
    return create_ascii_table(
        headers,
        rows,
        title=f"mpiuse summary: {data['label']}",
    )


def _format_per_rank_table(data: MpiuseData) -> str:
    assert data["per_rank_totals"] is not None
    headers = ["Rank", "Count", "Total bytes", "Total MB"]
    rows = [
        [
            str(int(row.rank)),
            f"{int(row.count):,}",
            f"{int(row.bytes):,}",
            f"{float(row.bytes) / 1024**2:.3f}",
        ]
        for row in data["per_rank_totals"].itertuples(index=False)
    ]
    rows.append(
        [
            "TOTAL",
            f"{data['total_count']:,}",
            f"{data['total_bytes']:,}",
            f"{data['total_bytes'] / 1024**2:.3f}",
        ]
    )
    return create_ascii_table(
        headers,
        rows,
        title=f"mpiuse per-rank totals: {data['label']}",
    )


def _format_comparison_table(all_data: list[MpiuseData]) -> str:
    reference = all_data[0]
    union_keys = sorted(
        {
            (str(row.type), str(row.subtype))
            for data in all_data
            for row in data["per_subtype"].itertuples(index=False)
        },
        key=lambda key: sum(
            _lookup_subtype_value(data, key) for data in all_data
        ),
        reverse=True,
    )

    headers = ["Type", "Subtype"]
    for data in all_data:
        headers.append(f"{data['label']} bytes")
    for data in all_data[1:]:
        headers.extend(
            [
                f"Δ bytes ({data['label']})",
                f"Δ % ({data['label']})",
            ]
        )

    rows: list[list[str]] = []
    ref_total = reference["total_bytes"]
    for key in union_keys:
        row = [key[0], key[1]]
        values = [_lookup_subtype_value(data, key) for data in all_data]
        row.extend(f"{value:,}" for value in values)
        for value in values[1:]:
            delta = value - values[0]
            delta_pct = 0.0 if values[0] == 0 else 100.0 * delta / values[0]
            row.extend([f"{delta:,}", f"{delta_pct:.2f}"])
        rows.append(row)

    total_row = ["TOTAL", ""]
    totals = [data["total_bytes"] for data in all_data]
    total_row.extend(f"{value:,}" for value in totals)
    for value in totals[1:]:
        delta = value - ref_total
        delta_pct = 0.0 if ref_total == 0 else 100.0 * delta / ref_total
        total_row.extend([f"{delta:,}", f"{delta_pct:.2f}"])
    rows.append(total_row)

    return create_ascii_table(headers, rows, title="mpiuse comparison")


def _lookup_subtype_value(data: MpiuseData, key: tuple[str, str]) -> int:
    matched = data["per_subtype"].loc[
        (data["per_subtype"]["type"] == key[0])
        & (data["per_subtype"]["subtype"] == key[1]),
        "bytes",
    ]
    if matched.empty:
        return 0
    return int(matched.iloc[0])


def _write_csv(
    rows: list[pd.DataFrame],
    output_path: str | None,
    prefix: str | None,
    out_dir: str,
) -> None:
    if not rows:
        return
    csv_data = pd.concat(rows, ignore_index=True)
    csv_data = csv_data[
        ["input", "label", "step", "type", "subtype", "count", "bytes"]
    ]
    csv_path = create_output_path(
        output_path,
        prefix,
        "mpiuse_per_step.csv",
        out_dir,
    )
    csv_data.to_csv(csv_path, index=False)


def _plot_total_per_step(
    all_data: list[MpiuseData],
    colors: Any,
    markers: list[str],
    output_path: str | None,
    prefix: str | None,
    out_dir: str,
    show_plot: bool,
    value_column: str,
    filename: str,
    ylabel: str,
    title: str,
    force_log: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    values: list[float] = []

    for index, data in enumerate(all_data):
        series = data["per_step_totals"]
        yvals = series[value_column].to_numpy(dtype=float)
        values.extend(yvals[yvals > 0])
        plot_values = np.where(yvals > 0, yvals, np.nan)
        ax.plot(
            series["step"],
            plot_values,
            marker=markers[index % len(markers)],
            markersize=4,
            color=colors[index],
            label=data["label"],
        )

    if len(all_data) > 1:
        ax.legend()
    ax.set_xlabel("Step number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if force_log or _should_use_log(values):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    out_file = create_output_path(output_path, prefix, filename, out_dir)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def _plot_subtype_bars(
    all_data: list[MpiuseData],
    colors: Any,
    output_path: str | None,
    prefix: str | None,
    out_dir: str,
    show_plot: bool,
    value_column: str,
    filename: str,
    ylabel: str,
    title: str,
) -> None:
    subtype_keys = sorted(
        {
            (str(row.type), str(row.subtype))
            for data in all_data
            for row in data["per_subtype"].itertuples(index=False)
        },
        key=lambda key: sum(
            int(
                data["per_subtype"]
                .loc[
                    (data["per_subtype"]["type"] == key[0])
                    & (data["per_subtype"]["subtype"] == key[1]),
                    value_column,
                ]
                .sum()
            )
            for data in all_data
        ),
        reverse=True,
    )
    if not subtype_keys:
        return

    labels = [_format_subtype_label(key) for key in subtype_keys]
    x = np.arange(len(subtype_keys), dtype=float)
    width = 0.8 / max(1, len(all_data))
    fig, ax = plt.subplots(figsize=(max(10, len(subtype_keys) * 0.8), 6))
    values: list[float] = []

    for index, data in enumerate(all_data):
        heights = [
            float(
                data["per_subtype"]
                .loc[
                    (data["per_subtype"]["type"] == key[0])
                    & (data["per_subtype"]["subtype"] == key[1]),
                    value_column,
                ]
                .sum()
            )
            for key in subtype_keys
        ]
        values.extend([value for value in heights if value > 0])
        ax.bar(
            x + (index - (len(all_data) - 1) / 2) * width,
            heights,
            width=width,
            color=colors[index],
            label=data["label"],
        )

    if len(all_data) > 1:
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if _should_use_log(values):
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", which="both")

    out_file = create_output_path(output_path, prefix, filename, out_dir)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def _plot_gpart_per_step(
    all_data: list[MpiuseData],
    colors: Any,
    markers: list[str],
    output_path: str | None,
    prefix: str | None,
    out_dir: str,
    show_plot: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    values: list[float] = []

    for index, data in enumerate(all_data):
        series = (
            data["per_step_detail"]
            .loc[data["per_step_detail"]["subtype"] == "gpart"]
            .groupby("step", as_index=False)["bytes"]
            .sum()
            .sort_values("step")
        )
        yvals = series["bytes"].to_numpy(dtype=float)
        values.extend(yvals[yvals > 0])
        plot_values = np.where(yvals > 0, yvals, np.nan)
        ax.plot(
            series["step"],
            plot_values,
            marker=markers[index % len(markers)],
            markersize=4,
            color=colors[index],
            label=data["label"],
        )

    if len(all_data) > 1:
        ax.legend()
    ax.set_xlabel("Step number")
    ax.set_ylabel("gpart bytes per step")
    ax.set_title("MPI gpart communication volume per step")
    if _should_use_log(values):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    out_file = create_output_path(
        output_path,
        prefix,
        "mpiuse_gpart_per_step.png",
        out_dir,
    )
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def _plot_inflight_max(
    all_data: list[MpiuseData],
    colors: Any,
    markers: list[str],
    output_path: str | None,
    prefix: str | None,
    out_dir: str,
    show_plot: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    values: list[float] = []

    for index, data in enumerate(all_data):
        series = data["inflight_per_step"]
        yvals = series["inflight_max"].to_numpy(dtype=float)
        values.extend(yvals[yvals > 0])
        plot_values = np.where(yvals > 0, yvals, np.nan)
        ax.plot(
            series["step"],
            plot_values,
            marker=markers[index % len(markers)],
            markersize=4,
            color=colors[index],
            label=data["label"],
        )

    if len(all_data) > 1:
        ax.legend()
    ax.set_xlabel("Step number")
    ax.set_ylabel("Max in-flight bytes")
    ax.set_title("MPI in-flight high-water mark per step")
    if _should_use_log(values):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    out_file = create_output_path(
        output_path,
        prefix,
        "mpiuse_inflight_max.png",
        out_dir,
    )
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def _plot_per_rank_totals(
    all_data: list[MpiuseData],
    output_path: str | None,
    prefix: str | None,
    out_dir: str,
    show_plot: bool,
) -> None:
    with_rank_data = [
        data for data in all_data if data["per_rank_detail"] is not None
    ]
    if not with_rank_data:
        return

    fig, axes = plt.subplots(
        len(with_rank_data),
        1,
        figsize=(12, max(4, 4 * len(with_rank_data))),
        squeeze=False,
    )

    subtype_keys_set: set[tuple[str, str]] = set()
    for data in with_rank_data:
        rank_detail = data["per_rank_detail"]
        if rank_detail is None:
            continue
        subtype_keys_set.update(
            (str(row.type), str(row.subtype))
            for row in rank_detail.itertuples(index=False)
        )
    subtype_keys = sorted(subtype_keys_set)
    colors = plt.get_cmap("tab20")(
        np.linspace(0, 1, max(1, len(subtype_keys)))
    )

    for axis, data in zip(axes.flat, with_rank_data):
        _draw_rank_stack(axis, data, subtype_keys, colors)

    out_file = create_output_path(
        output_path,
        prefix,
        "mpiuse_bytes_per_rank.png",
        out_dir,
    )
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def _draw_rank_stack(
    axis: Axes,
    data: MpiuseData,
    subtype_keys: list[tuple[str, str]],
    colors: Any,
) -> None:
    assert data["per_rank_detail"] is not None
    rank_df = data["per_rank_detail"]
    ranks = sorted(int(value) for value in rank_df["rank"].unique())
    x = np.arange(len(ranks))
    bottom = np.zeros(len(ranks), dtype=float)

    for index, key in enumerate(subtype_keys):
        heights = np.array(
            [
                float(
                    rank_df.loc[
                        (rank_df["rank"] == rank)
                        & (rank_df["type"] == key[0])
                        & (rank_df["subtype"] == key[1]),
                        "bytes",
                    ].sum()
                )
                for rank in ranks
            ]
        )
        if not np.any(heights > 0):
            continue
        axis.bar(
            x,
            heights,
            bottom=bottom,
            color=colors[index],
            label=_format_subtype_label(key),
        )
        bottom += heights

    axis.set_xticks(x)
    axis.set_xticklabels([str(rank) for rank in ranks])
    axis.set_xlabel("Rank")
    axis.set_ylabel("Total bytes")
    axis.set_title(f"MPI bytes per rank: {data['label']}")
    axis.grid(True, axis="y", alpha=0.3, linestyle="--")
    if len(subtype_keys) <= 12:
        axis.legend(fontsize=8, ncol=2)


def _should_use_log(values: list[float]) -> bool:
    positive_values = [value for value in values if value > 0]
    if len(positive_values) < 2:
        return False
    return max(positive_values) / min(positive_values) >= 100


def _has_gpart(all_data: list[MpiuseData]) -> bool:
    return any(
        not data["per_step_detail"]
        .loc[data["per_step_detail"]["subtype"] == "gpart"]
        .empty
        for data in all_data
    )


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "mpiuse"


def _format_subtype_label(key: tuple[str, str]) -> str:
    return f"{key[0]}:{key[1]}"
