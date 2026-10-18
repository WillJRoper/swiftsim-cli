"""MPI communication usage analysis for SWIFT mpiuse reports."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, TypedDict

import numpy as np

from swiftsim_cli.modes.common_args import (
    add_labels_argument,
    add_output_arguments,
    add_steps_argument,
    resolve_labels,
)
from swiftsim_cli.utilities.io import create_output_path, slugify
from swiftsim_cli.utilities.messages import plural, warn
from swiftsim_cli.utilities.plotting import save_figure, should_use_log_scale
from swiftsim_cli.utilities.tables import create_ascii_table

from .task_plot_metadata import distinct_colours

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

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
    direction: str
    total_bytes: int
    total_count: int
    recv_bytes: int
    recv_count: int
    per_step_detail: pd.DataFrame
    per_step_totals: pd.DataFrame
    per_subtype: pd.DataFrame
    inflight_per_step: pd.DataFrame
    per_rank_detail: pd.DataFrame | None
    per_rank_totals: pd.DataFrame | None


# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "mpiuse_analysis"

__all__ = [
    "add_mpiuse_arguments",
    "run_swift_mpiuse",
    "analyse_swift_mpiuse",
]


def add_mpiuse_arguments(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add CLI arguments for SWIFT mpiuse report analysis."""
    mpiuse_parser = subparsers.add_parser(
        "mpiuse",
        help=(
            "Analyse SWIFT mpiuse_report-rank*-step*.dat files and compare "
            "communication volume across inputs."
        ),
        description="Analyse SWIFT mpiuse_report-rank*-step*.dat files and "
        "compare communication volume across inputs. Requires an MPI "
        "build of SWIFT configured with --enable-mpiuse-reports.",
    )
    mpiuse_parser.set_defaults(func=run_swift_mpiuse)

    mpiuse_parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help=(
            "Input directories, glob patterns, or individual "
            "mpiuse_report files to analyse."
        ),
        type=str,
    )
    add_labels_argument(mpiuse_parser)
    add_output_arguments(mpiuse_parser, OUT_SUBDIR)
    mpiuse_parser.add_argument(
        "--subtypes",
        nargs="+",
        type=str,
        metavar="NAME",
        help="Subtypes to include, e.g. gpart grav_counts (default: all).",
        default=None,
    )
    add_steps_argument(mpiuse_parser)
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
        help="Write ASCII tables, CSV output, or both (default: ascii).",
    )


def run_swift_mpiuse(args: argparse.Namespace) -> None:
    """Entry point for the ``mpiuse`` CLI subcommand."""
    analyse_swift_mpiuse(
        inputs=args.inputs,
        labels=args.labels,
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        subtype_filter=args.subtypes,
        step_range=args.steps,
        per_rank=args.per_rank,
        output_format=args.format,
    )


def analyse_swift_mpiuse(
    inputs: list[str],
    labels: list[str] | None = None,
    output_path: str | Path | None = None,
    prefix: str | None = None,
    show_plot: bool = False,
    subtype_filter: list[str] | None = None,
    step_range: Sequence[int] | None = None,
    per_rank: bool = False,
    output_format: str = "ascii",
) -> None:
    """Analyse SWIFT mpiuse report files.

    Args:
        inputs: Input directories, glob patterns, or files to analyse.
        labels: Optional labels matching ``inputs``.
        output_path: Directory whose ``mpiuse_analysis`` subdirectory
            receives the outputs. If None, the current directory is used.
        prefix: Optional prefix added to each output filename.
        show_plot: Whether to display plots interactively.
        subtype_filter: Optional subtype names to include.
        step_range: Optional inclusive ``(start, end)`` step range.
        per_rank: Whether to produce per-rank summaries and plots.
        output_format: ``ascii``, ``csv``, or ``both``.
    """
    print(f"Analysing mpiuse reports from {plural(len(inputs), 'input')}")

    if step_range is not None and step_range[0] > step_range[1]:
        raise ValueError("Step range must be ordered as START END")

    labels = resolve_labels(inputs, labels)

    out_dir = OUT_SUBDIR
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    all_data: list[MpiuseData] = []
    csv_rows: list[pd.DataFrame] = []
    seen_subtypes: set[str] = set()

    for input_path, label in zip(inputs, labels):
        files = _discover_files(input_path)
        if not files:
            warn(f"no mpiuse_report files matched '{input_path}'; skipping it")
            continue

        aggregated = _aggregate(
            files=files,
            step_range=step_range,
            subtype_filter=subtype_filter,
            per_rank=per_rank,
            seen_subtypes=seen_subtypes,
        )
        if aggregated is None:
            warn(
                f"no usable mpiuse data remained for '{label}' after "
                "filtering; skipping it"
            )
            continue

        data: MpiuseData = {
            "input_path": input_path,
            "label": label,
            "files": [str(path) for path in aggregated["files"]],
            "skipped_files": int(aggregated["skipped_files"]),
            "ranks": aggregated["ranks"],
            "steps": aggregated["steps"],
            "direction": aggregated["direction"],
            "total_bytes": int(aggregated["total_bytes"]),
            "total_count": int(aggregated["total_count"]),
            "recv_bytes": int(aggregated["recv_bytes"]),
            "recv_count": int(aggregated["recv_count"]),
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
        label_slug = slugify(label, fallback="mpiuse", keep="._-")

        print()
        print(headline)
        print(summary_table)

        table_path = create_output_path(
            output_path,
            prefix,
            f"mpiuse_summary_{label_slug}.txt",
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
                f"mpiuse_per_rank_{label_slug}.txt",
                out_dir,
            )
            rank_path.write_text(rank_table + "\n", encoding="utf-8")

    if subtype_filter is not None and seen_subtypes:
        unknown = sorted(set(subtype_filter) - seen_subtypes)
        if unknown:
            raise ValueError(
                f"unknown subtype(s) {', '.join(unknown)}; the reports "
                "list: " + ", ".join(sorted(seen_subtypes))
            )
    if not all_data:
        raise ValueError(
            "no usable mpiuse data found in any input (expected "
            "mpiuse_report-rank*-step*.dat files)"
        )

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

    colors = distinct_colours(len(all_data))
    side = "sent" if all_data[0]["direction"] == "send" else "received"

    plot_args = (colors, markers, output_path, prefix, out_dir, show_plot)
    step_totals = [data["per_step_totals"] for data in all_data]
    _plot_step_series(
        all_data,
        step_totals,
        "bytes",
        *plot_args,
        filename="mpiuse_total_bytes_per_step.png",
        ylabel=f"Bytes {side} per step",
        title=f"MPI communication volume per step (bytes {side}, each "
        "message once)",
        force_log=True,
    )
    _plot_step_series(
        all_data,
        step_totals,
        "count",
        *plot_args,
        filename="mpiuse_total_count_per_step.png",
        ylabel=f"Messages {side} per step",
        title=f"MPI message count per step ({side})",
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
        ylabel=f"Bytes {side}",
        title=f"MPI bytes {side} by subtype",
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
        ylabel=f"Messages {side}",
        title=f"MPI messages {side} by subtype",
    )
    if _has_gpart(all_data):
        _plot_step_series(
            all_data,
            [_gpart_bytes_per_step(data) for data in all_data],
            "bytes",
            *plot_args,
            filename="mpiuse_gpart_per_step.png",
            ylabel=f"gpart bytes {side} per step",
            title=f"MPI gpart communication volume per step ({side})",
        )
    _plot_step_series(
        all_data,
        [data["inflight_per_step"] for data in all_data],
        "inflight_max",
        *plot_args,
        filename="mpiuse_inflight_max.png",
        ylabel="Max in-flight bytes (any rank)",
        title="MPI in-flight high-water mark per step"
        + (f" ({', '.join(subtype_filter)})" if subtype_filter else ""),
    )
    if per_rank:
        _plot_per_rank_totals(
            all_data,
            output_path,
            prefix,
            out_dir,
            show_plot,
        )


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
    import pandas as pd
    from pandas.errors import EmptyDataError

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
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e

    if data.empty:
        return pd.DataFrame(columns=MPIUSE_COLUMNS)

    _parse_rank_step(path)
    return data


def _aggregate(
    files: list[Path],
    step_range: Sequence[int] | None,
    subtype_filter: list[str] | None,
    per_rank: bool,
    seen_subtypes: set[str] | None = None,
) -> dict[str, Any] | None:
    """Aggregate activation rows of the given report files.

    Each message appears twice (a send on one rank, a recv on another),
    so volumes are reported per direction: the totals, per-step and
    per-subtype plots use the send side (the recv side when no sends were
    logged), never the sum of both.

    The in-flight high-water mark is the largest running total of message
    sizes on any rank in a step: SWIFT's ``sum`` column, or, with a
    subtype filter, the running total of the selected subtypes alone.
    """
    import pandas as pd

    activation_frames: list[pd.DataFrame] = []
    inflight_frames: list[pd.DataFrame] = []
    used_files: list[Path] = []
    skipped_files = 0

    for path in files:
        raw = _read_mpiuse_raw_file(path)
        if seen_subtypes is not None:
            seen_subtypes.update(raw["subtype"].astype(str))
        if step_range is not None:
            raw = raw.loc[
                (raw["step"] >= step_range[0]) & (raw["step"] <= step_range[1])
            ]
        if subtype_filter is not None:
            raw = raw.loc[raw["subtype"].isin(subtype_filter)]
        activation = raw.loc[raw["activation"] == 1]
        if activation.empty:
            skipped_files += 1
            continue

        activation_frames.append(activation)
        if subtype_filter is None:
            running = raw["sum"]
        else:
            # ponytail: starts from 0 at the report start, so messages
            # still in flight from before it are not counted.
            running = raw.groupby(["step", "rank"])["size"].cumsum()
        inflight_frames.append(
            raw.assign(inflight_max=running)
            .groupby(["step", "rank"], as_index=False)["inflight_max"]
            .max()
        )
        used_files.append(path)

    if not activation_frames:
        return None

    data = pd.concat(activation_frames, ignore_index=True)
    inflight_data = pd.concat(inflight_frames, ignore_index=True)
    direction = "send" if (data["type"] == "send").any() else "recv"
    primary = data.loc[data["type"] == direction]
    recv = data.loc[data["type"] == "recv"]

    def totals(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return (
            frame.groupby(keys, as_index=False)
            .agg(bytes=("size", "sum"), count=("size", "size"))
            .sort_values(keys)
            .reset_index(drop=True)
        )

    per_step_detail = (
        totals(data, ["step", "type", "subtype"])
        .sort_values(["step", "bytes"], ascending=[True, False])
        .reset_index(drop=True)
    )
    per_subtype = (
        data.groupby(["type", "subtype"], as_index=False)
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
        per_rank_detail = totals(primary, ["rank", "type", "subtype"])
        per_rank_totals = (
            data.pivot_table(
                index="rank",
                columns="type",
                values="size",
                aggfunc=["sum", "size"],
                fill_value=0,
            )
            .reindex(
                columns=pd.MultiIndex.from_product(
                    [["sum", "size"], ["send", "recv"]]
                ),
                fill_value=0,
            )
            .set_axis(
                ["sent_bytes", "recv_bytes", "sent_count", "recv_count"],
                axis=1,
            )
            .reset_index()
        )

    return {
        "files": used_files,
        "skipped_files": skipped_files,
        "ranks": sorted(int(value) for value in data["rank"].unique()),
        "steps": sorted(int(value) for value in data["step"].unique()),
        "direction": direction,
        "total_bytes": int(primary["size"].sum()),
        "total_count": int(len(primary)),
        "recv_bytes": int(recv["size"].sum()),
        "recv_count": int(len(recv)),
        "per_step_detail": per_step_detail,
        "per_step_totals": totals(primary, ["step"]),
        "per_subtype": per_subtype,
        "inflight_per_step": inflight_per_step,
        "per_rank_detail": per_rank_detail,
        "per_rank_totals": per_rank_totals,
    }


def _format_headline(data: MpiuseData) -> str:
    step_min = min(data["steps"])
    step_max = max(data["steps"])
    sent = (
        f"Bytes sent: {data['total_bytes']:,} "
        f"({data['total_bytes'] / 1024**2:.3f} MB) in "
        f"{data['total_count']:,} messages"
        if data["direction"] == "send"
        else "Bytes sent: none logged"
    )
    return (
        f"mpiuse summary: {data['label']}\n"
        f"Files parsed: {len(data['files'])} "
        f"(skipped: {data['skipped_files']}), "
        f"steps: {step_min}-{step_max}, ranks: {data['ranks']}\n"
        f"{sent}\n"
        f"Bytes received: {data['recv_bytes']:,} "
        f"({data['recv_bytes'] / 1024**2:.3f} MB) in "
        f"{data['recv_count']:,} messages\n"
        "(each message is logged as a send and a recv; totals and plots "
        f"use the {data['direction']} side)"
    )


def _format_summary_table(data: MpiuseData) -> str:
    direction_totals = {
        "send": max(
            data["total_bytes"] if data["direction"] == "send" else 0, 1
        ),
        "recv": max(data["recv_bytes"], 1),
    }
    headers = [
        "Type",
        "Subtype",
        "Count",
        "Total bytes",
        "Total MB",
        "Mean bytes",
        "Max bytes",
        "% of type",
    ]
    rows: list[list[str]] = []
    for row in data["per_subtype"].itertuples(index=False):
        share = direction_totals.get(str(row.type), 1)
        rows.append(
            [
                str(row.type),
                str(row.subtype),
                f"{int(row.count):,}",
                f"{int(row.bytes):,}",
                f"{float(row.bytes) / 1024**2:.3f}",
                f"{float(row.mean_bytes):.1f}",
                f"{int(row.max_bytes):,}",
                f"{100.0 * float(row.bytes) / share:.2f}",
            ]
        )
    per_type = data["per_subtype"].groupby("type")
    for kind in ("send", "recv"):
        if kind not in per_type.groups:
            continue
        group = per_type.get_group(kind)
        rows.append(
            [
                f"TOTAL {kind}",
                "",
                f"{int(group['count'].sum()):,}",
                f"{int(group['bytes'].sum()):,}",
                f"{float(group['bytes'].sum()) / 1024**2:.3f}",
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
    totals = data["per_rank_totals"]
    assert totals is not None
    headers = ["Rank", "Sent", "Sent bytes", "Received", "Received bytes"]
    rows = [
        [
            str(int(row.rank)),
            f"{int(row.sent_count):,}",
            f"{int(row.sent_bytes):,}",
            f"{int(row.recv_count):,}",
            f"{int(row.recv_bytes):,}",
        ]
        for row in totals.itertuples(index=False)
    ]
    rows.append(
        [
            "TOTAL",
            f"{int(totals['sent_count'].sum()):,}",
            f"{int(totals['sent_bytes'].sum()):,}",
            f"{int(totals['recv_count'].sum()):,}",
            f"{int(totals['recv_bytes'].sum()):,}",
        ]
    )
    return create_ascii_table(
        headers,
        rows,
        title=f"mpiuse per-rank messages: {data['label']}",
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

    total_row = [f"TOTAL {reference['direction']}", ""]
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
    output_path: str | Path | None,
    prefix: str | None,
    out_dir: str,
) -> None:
    import pandas as pd

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


def _plot_step_series(
    all_data: list[MpiuseData],
    step_frames: list[pd.DataFrame],
    value_column: str,
    colors: Any,
    markers: list[str],
    output_path: str | Path | None,
    prefix: str | None,
    out_dir: str,
    show_plot: bool,
    filename: str,
    ylabel: str,
    title: str,
    force_log: bool = False,
) -> None:
    """Plot one per-step line per input, from ``step_frames[i]``."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    values: list[float] = []

    for index, (data, series) in enumerate(zip(all_data, step_frames)):
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
    ax.xaxis.get_major_locator().set_params(integer=True)  # type: ignore[call-arg]
    ax.set_xlabel("Step number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if force_log or should_use_log_scale(values):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    save_figure(
        fig,
        create_output_path(output_path, prefix, filename, out_dir),
        show=show_plot,
    )


def _plot_subtype_bars(
    all_data: list[MpiuseData],
    colors: Any,
    output_path: str | Path | None,
    prefix: str | None,
    out_dir: str,
    show_plot: bool,
    value_column: str,
    filename: str,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    # One direction only, so each message is counted once.
    all_data = [
        {
            **data,
            "per_subtype": data["per_subtype"].loc[
                data["per_subtype"]["type"] == data["direction"]
            ],
        }
        for data in all_data
    ]

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
    if should_use_log_scale(values):
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", which="both")

    save_figure(
        fig,
        create_output_path(output_path, prefix, filename, out_dir),
        show=show_plot,
    )


def _gpart_bytes_per_step(data: MpiuseData) -> pd.DataFrame:
    """Sum gpart communication bytes per step for one input."""
    return (
        data["per_step_detail"]
        .loc[
            (data["per_step_detail"]["subtype"] == "gpart")
            & (data["per_step_detail"]["type"] == data["direction"])
        ]
        .groupby("step", as_index=False)["bytes"]
        .sum()
        .sort_values("step")
    )


def _plot_per_rank_totals(
    all_data: list[MpiuseData],
    output_path: str | Path | None,
    prefix: str | None,
    out_dir: str,
    show_plot: bool,
) -> None:
    import matplotlib.pyplot as plt

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
    colors = distinct_colours(len(subtype_keys))

    for axis, data in zip(axes.flat, with_rank_data):
        _draw_rank_stack(axis, data, subtype_keys, colors)

    save_figure(
        fig,
        create_output_path(
            output_path, prefix, "mpiuse_bytes_per_rank.png", out_dir
        ),
        show=show_plot,
    )


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
    side = "sent" if data["direction"] == "send" else "received"
    axis.set_ylabel(f"Bytes {side}")
    axis.set_title(f"MPI bytes {side} per rank: {data['label']}")
    axis.grid(True, axis="y", alpha=0.3, linestyle="--")
    axis.legend(
        fontsize="small",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=max(1, -(-len(subtype_keys) // 20)),
        borderaxespad=0.0,
    )


def _has_gpart(all_data: list[MpiuseData]) -> bool:
    return any(
        not data["per_step_detail"]
        .loc[data["per_step_detail"]["subtype"] == "gpart"]
        .empty
        for data in all_data
    )


def _format_subtype_label(key: tuple[str, str]) -> str:
    return f"{key[0]}:{key[1]}"
