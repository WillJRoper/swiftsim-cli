"""Scaling analysis for SWIFT timing logs."""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np

from swiftsim_cli.src_parser import (
    TimerDef,
    TimerInstance,
    compile_site_patterns,
    load_timer_db,
    scan_log_instances_by_step,
)
from swiftsim_cli.utilities import create_ascii_table, create_output_path

MPI_RANKS_RE = re.compile(
    r"MPI is up and running with\s+(?P<ranks>\d+)\s+node\(s\)"
)


@dataclass
class ScalingLogData:
    """Aggregated scaling data for one log file."""

    log_file: str
    label: str
    rank_count: int
    timer_totals: dict[str, float]
    timer_percentages: dict[str, float]
    total_time: float
    per_rank_totals: dict[int, dict[str, float]]
    timer_call_counts: dict[str, int]
    total_call_count: int
    per_rank_call_counts: dict[int, dict[str, int]]
    emitted_rank_count: int


def add_scaling_arguments(subparsers) -> None:
    """Add CLI arguments for scaling analysis."""
    scaling_parser = subparsers.add_parser(
        "scaling",
        help=(
            "Analyse how SWIFT timers scale across multiple log files with "
            "different MPI rank counts."
        ),
    )

    scaling_parser.add_argument(
        "log_files",
        nargs="+",
        type=Path,
        help="SWIFT log files to analyse together.",
    )
    scaling_parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Where to save analysis outputs (default: current directory).",
        default=None,
    )
    scaling_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="Prefix for output files and output directory.",
        default=None,
    )
    scaling_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively.",
        default=False,
    )
    scaling_parser.add_argument(
        "--steps",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive step range to analyse.",
        default=None,
    )
    scaling_parser.add_argument(
        "--min-percent-threshold",
        type=float,
        default=10.0,
        help=(
            "Minimum per-log percentage contribution required for a timer to "
            "be included (default: 10)."
        ),
    )
    scaling_parser.add_argument(
        "--rank-source",
        choices=("root", "all"),
        default="root",
        help=(
            "Which ranks to use for the main scaling view: 'root' uses rank 0 "
            "only, 'all' combines all emitting ranks present in each log."
        ),
    )


def run_swift_scaling(args: argparse.Namespace) -> None:
    """Entry point for the ``scaling`` CLI subcommand."""
    step_range = None
    if args.steps is not None:
        step_range = (args.steps[0], args.steps[1])

    rank_source = cast(Literal["root", "all"], args.rank_source)

    analyse_swift_scaling(
        log_files=[str(path) for path in args.log_files],
        output_path=str(args.output_path) if args.output_path else None,
        prefix=args.prefix,
        show_plot=args.show,
        step_range=step_range,
        min_percent_threshold=args.min_percent_threshold,
        rank_source=rank_source,
    )


def analyse_swift_scaling(
    log_files: list[str],
    output_path: str | None = None,
    prefix: str | None = None,
    show_plot: bool = True,
    step_range: tuple[int, int] | None = None,
    min_percent_threshold: float = 10.0,
    rank_source: Literal["root", "all"] = "root",
) -> None:
    """Analyse timer scaling across multiple SWIFT log files."""
    if len(log_files) < 2:
        raise ValueError("Scaling analysis requires at least two log files")
    if step_range is not None and step_range[0] > step_range[1]:
        raise ValueError("Step range must be ordered as START END")
    if min_percent_threshold < 0:
        raise ValueError("min_percent_threshold must be non-negative")
    if rank_source not in {"root", "all"}:
        raise ValueError("rank_source must be one of 'root' or 'all'")

    print(f"Analyzing timer scaling across {len(log_files)} log files")

    out_dir = (
        "scaling_analysis" if prefix is None else f"{prefix}_scaling_analysis"
    )

    timer_db = load_timer_db()
    compiled = compile_site_patterns(timer_db)

    scaling_data: list[ScalingLogData] = []

    for log_file in log_files:
        print(f"\nProcessing scaling input: {log_file}")
        rank_count = _extract_rank_count_from_log(log_file)
        instances_by_step, _ = scan_log_instances_by_step(
            log_file, compiled, timer_db
        )
        (
            timer_totals,
            timer_call_counts,
            per_rank_totals,
            per_rank_call_counts,
        ) = _aggregate_timer_totals(instances_by_step, step_range)
        total_time = sum(timer_totals.values())
        total_call_count = sum(timer_call_counts.values())
        timer_percentages = {
            timer_id: (100.0 * total / total_time if total_time > 0 else 0.0)
            for timer_id, total in timer_totals.items()
        }

        scaling_data.append(
            ScalingLogData(
                log_file=log_file,
                label=Path(log_file).name,
                rank_count=rank_count,
                timer_totals=timer_totals,
                timer_percentages=timer_percentages,
                total_time=total_time,
                per_rank_totals=per_rank_totals,
                timer_call_counts=timer_call_counts,
                total_call_count=total_call_count,
                per_rank_call_counts=per_rank_call_counts,
                emitted_rank_count=len(per_rank_totals),
            )
        )

        if len(per_rank_totals) < rank_count:
            print(
                "Warning: only "
                f"{len(per_rank_totals)}/{rank_count} ranks emitted timer "
                f"lines in {Path(log_file).name}. This often indicates a "
                "log written with -v 1 rather than -v 2."
            )

    scaling_data.sort(key=lambda item: (item.rank_count, item.label))
    selected_timers = _select_timer_ids(scaling_data, min_percent_threshold)
    if not selected_timers:
        raise ValueError(
            "No timers satisfied the selected minimum percentage threshold"
        )

    run_statistics: list[Literal["sum", "mean"]] = ["sum", "mean"]
    plot_paths: list[Path] = []
    summary_sections: list[str] = []

    for _plot_index, run_statistic in enumerate(run_statistics, start=1):
        used_log_scale = _plot_would_use_log_scale(
            scaling_data=scaling_data,
            timer_ids=selected_timers,
            run_statistic=run_statistic,
            rank_source=rank_source,
        )

        plot_path = _create_scaling_plot(
            scaling_data=scaling_data,
            timer_ids=selected_timers,
            timer_db=timer_db,
            output_path=output_path,
            prefix=prefix,
            show_plot=show_plot,
            out_dir=out_dir,
            filename=f"timer_scaling_{run_statistic}_{rank_source}.png",
            title=(
                "Timer Scaling by MPI Rank Count "
                f"({_run_statistic_title(run_statistic)}, "
                f"{_rank_source_title(rank_source)})"
            ),
            run_statistic=run_statistic,
            rank_source=rank_source,
        )
        if plot_path is not None:
            plot_paths.append(plot_path)

        table_text = _build_scaling_table(
            scaling_data,
            selected_timers,
            timer_db,
            min_percent_threshold,
            run_statistic,
            rank_source,
        )
        analysis_text = _build_scaling_analysis(
            scaling_data,
            selected_timers,
            timer_db,
            min_percent_threshold,
            used_log_scale,
            run_statistic,
            rank_source,
        )

        print("\n" + table_text)
        print("\n" + analysis_text)
        summary_sections.append(f"{table_text}\n\n{analysis_text}")

    summary_path = create_output_path(
        output_path,
        prefix,
        "scaling_summary.txt",
        out_dir,
    )
    summary_path.write_text(
        "\n\n".join(summary_sections) + "\n", encoding="utf-8"
    )

    per_rank_paths: list[Path] = []
    per_timer_rank_paths: list[Path] = []
    verbose_ranks = _collect_verbose_ranks(scaling_data)
    for rank in verbose_ranks:
        rank_path = _create_scaling_plot(
            scaling_data=scaling_data,
            timer_ids=selected_timers,
            timer_db=timer_db,
            output_path=output_path,
            prefix=prefix,
            show_plot=show_plot,
            out_dir=out_dir,
            filename=f"timer_scaling_rank_{rank:04d}.png",
            title=f"Timer Scaling by MPI Rank Count for Rank {rank}",
            per_rank=rank,
            run_statistic="sum",
            rank_source="root",
        )
        if rank_path is not None:
            per_rank_paths.append(rank_path)

    comparable_ranks = _collect_comparable_ranks(scaling_data)
    if comparable_ranks:
        for _timer_index, timer_id in enumerate(selected_timers, start=1):
            timer_rank_path = _create_timer_rank_scaling_plot(
                scaling_data=scaling_data,
                timer_id=timer_id,
                timer_db=timer_db,
                ranks=comparable_ranks,
                output_path=output_path,
                prefix=prefix,
                show_plot=show_plot,
                out_dir=out_dir,
                filename=(
                    "timer_rank_scaling_"
                    f"{_slugify(_timer_display_name(timer_id, timer_db))}.png"
                ),
            )
            if timer_rank_path is not None:
                per_timer_rank_paths.append(timer_rank_path)

    if verbose_ranks:
        imbalance_text = _build_rank_imbalance_table(
            scaling_data, selected_timers, timer_db
        )
        if imbalance_text is not None:
            print("\n" + imbalance_text)
            imbalance_path = create_output_path(
                output_path,
                prefix,
                "rank_imbalance_summary.txt",
                out_dir,
            )
            imbalance_path.write_text(imbalance_text + "\n", encoding="utf-8")

    print("\nCreated outputs:")
    for plot_path in plot_paths:
        print(f"  - {plot_path}")
    print(f"  - {summary_path}")
    for rank_path in per_rank_paths:
        print(f"  - {rank_path}")
    for timer_rank_path in per_timer_rank_paths:
        print(f"  - {timer_rank_path}")


def _extract_rank_count_from_log(log_file: str) -> int:
    """Extract the MPI rank count from a SWIFT log file."""
    with open(log_file, "r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(200):
            line = handle.readline()
            if not line:
                break
            match = MPI_RANKS_RE.search(line)
            if match is not None:
                return int(match.group("ranks"))
    raise ValueError(
        f"Could not determine MPI rank count from log '{log_file}'"
    )


def _aggregate_timer_totals(
    instances_by_step: dict[int | None, list[TimerInstance]],
    step_range: tuple[int, int] | None,
) -> tuple[
    dict[str, float],
    dict[str, int],
    dict[int, dict[str, float]],
    dict[int, dict[str, int]],
]:
    """Aggregate timer totals overall and per emitting rank."""
    timer_totals: dict[str, float] = defaultdict(float)
    timer_call_counts: dict[str, int] = defaultdict(int)
    per_rank_totals: dict[int, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    per_rank_call_counts: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for step, instances in instances_by_step.items():
        if step_range is not None:
            if step is None or step < step_range[0] or step > step_range[1]:
                continue
        for instance in instances:
            timer_totals[instance.timer_id] += instance.time_ms
            timer_call_counts[instance.timer_id] += 1
            per_rank_totals[instance.rank][instance.timer_id] += (
                instance.time_ms
            )
            per_rank_call_counts[instance.rank][instance.timer_id] += 1

    return (
        dict(timer_totals),
        dict(timer_call_counts),
        {
            rank: dict(rank_totals)
            for rank, rank_totals in per_rank_totals.items()
        },
        {
            rank: dict(rank_counts)
            for rank, rank_counts in per_rank_call_counts.items()
        },
    )


def _select_timer_ids(
    scaling_data: list[ScalingLogData], min_percent_threshold: float
) -> list[str]:
    """Select timers that exceed the threshold in at least one input."""
    selected: list[tuple[str, float, float]] = []
    seen_timer_ids: set[str] = set()

    for log_data in scaling_data:
        seen_timer_ids.update(log_data.timer_totals.keys())

    for timer_id in seen_timer_ids:
        max_percent = max(
            log_data.timer_percentages.get(timer_id, 0.0)
            for log_data in scaling_data
        )
        max_total = max(
            log_data.timer_totals.get(timer_id, 0.0)
            for log_data in scaling_data
        )
        if max_percent >= min_percent_threshold:
            selected.append((timer_id, max_percent, max_total))

    selected.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    return [timer_id for timer_id, _, _ in selected]


def _timer_display_name(timer_id: str, timer_db: dict[str, TimerDef]) -> str:
    """Build a readable label for a timer."""
    timer_def = timer_db[timer_id]
    label_text = timer_def.label_text.strip()
    if label_text in {"took %.3f %s.", "took %.3f %s", "(%s)"}:
        return timer_def.function

    label = _clean_timer_label_text(label_text)
    if not label:
        return timer_def.function
    return f"{timer_def.function}: {label}"


def _clean_timer_label_text(label_text: str) -> str:
    """Remove timing boilerplate and format tokens from timer labels."""
    label = label_text.strip()

    # Prefer the descriptive part before the timing suffix when present.
    if " took " in label:
        label = label.split(" took ", maxsplit=1)[0]
    elif label.startswith("took "):
        label = label[5:]

    label = label.replace("%.3f %s", "")
    label = label.replace("%s", "")
    label = label.replace("%d", "")
    label = label.replace("%f", "")

    # If the remaining text is just a parenthesized qualifier, keep the
    # content rather than the wrapper punctuation.
    paren_match = re.fullmatch(r"\((.*)\)\.??", label.strip())
    if paren_match is not None:
        label = paren_match.group(1)

    label = re.sub(r"^took\b", "", label).strip()
    label = re.sub(r"\s+", " ", label).strip(" .:;,-()")
    return label


def _create_scaling_plot(
    scaling_data: list[ScalingLogData],
    timer_ids: list[str],
    timer_db: dict[str, TimerDef],
    output_path: str | None,
    prefix: str | None,
    show_plot: bool,
    out_dir: str,
    filename: str,
    title: str,
    per_rank: int | None = None,
    run_statistic: Literal["sum", "mean"] = "sum",
    rank_source: Literal["root", "all"] = "root",
) -> Path | None:
    """Create a scaling line plot for the selected timers."""
    include_total_reference = run_statistic == "sum"
    legend_labels = _build_legend_labels(
        timer_ids, timer_db, include_total_reference
    )
    series_count = len(legend_labels)
    legend_columns = _legend_columns(legend_labels)
    legend_rows = math.ceil(series_count / legend_columns)
    plot_size = 10.5
    speedup_height = 3.0
    legend_height = 0.9 + max(0, legend_rows - 1) * 0.65
    fig = plt.figure(
        figsize=(plot_size, plot_size + speedup_height + legend_height)
    )
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[plot_size, speedup_height, legend_height],
        hspace=0.04,
    )
    ax = fig.add_subplot(grid[0])
    speedup_ax = fig.add_subplot(grid[1], sharex=ax)
    legend_ax = fig.add_subplot(grid[2])
    legend_ax.axis("off")
    ax.set_box_aspect(1)

    all_values: list[float] = []
    plotted = 0

    total_x_values, total_y_values = _build_total_series(
        scaling_data, per_rank, run_statistic, rank_source
    )
    if include_total_reference and len(total_x_values) >= 2:
        perfect_y_values = _build_perfect_scaling_series(
            total_x_values, total_y_values
        )
        ax.plot(
            total_x_values,
            total_y_values,
            marker="o",
            linewidth=3,
            markersize=6,
            color="black",
            label=legend_labels[0],
            zorder=4,
        )
        speedup_ax.plot(
            total_x_values,
            _compute_speedup_series(total_y_values),
            marker="o",
            linewidth=2.5,
            markersize=5,
            color="black",
        )
        speedup_ax.plot(
            total_x_values,
            _build_perfect_speedup_series(total_x_values),
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.45,
        )
        ax.plot(
            total_x_values,
            perfect_y_values,
            linestyle="--",
            linewidth=2,
            color="0.35",
            label=legend_labels[1],
            zorder=3,
        )
        all_values.extend(total_y_values)
        all_values.extend(perfect_y_values)
        plotted += 2

    for timer_id in timer_ids:
        x_values: list[int] = []
        y_values: list[float] = []

        for log_data in scaling_data:
            totals = (
                _build_rank_series_dict(log_data, per_rank, run_statistic)
                if per_rank is not None
                else _build_main_series_dict(
                    log_data, run_statistic, rank_source
                )
            )
            value = totals.get(timer_id)
            if value is None or value <= 0:
                continue
            x_values.append(log_data.rank_count)
            y_values.append(value)

        if len(x_values) < 2:
            continue

        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            markersize=5,
            label=legend_labels[plotted],
        )
        color = ax.lines[-1].get_color()
        speedup_ax.plot(
            x_values,
            _compute_speedup_series(y_values),
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
        )
        speedup_ax.plot(
            x_values,
            _build_perfect_speedup_series(x_values),
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.35,
        )
        all_values.extend(y_values)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_ylabel("Accumulated Timer Time (ms)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    speedup_ax.set_xlabel("Number of Ranks")
    speedup_ax.set_ylabel("Relative Speedup")
    speedup_ax.grid(True, alpha=0.3, linestyle="--")
    speedup_ax.axhline(1.0, color="0.5", linewidth=1.0, alpha=0.6)

    x_ticks = sorted({item.rank_count for item in scaling_data})
    ax.set_xticks(x_ticks)
    speedup_ax.set_xticks(x_ticks)
    plt.setp(ax.get_xticklabels(), visible=False)

    if _should_use_log_scale(all_values):
        ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=legend_columns,
        fontsize=9,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        labelspacing=0.8,
        borderaxespad=0.0,
    )

    output = create_output_path(output_path, prefix, filename, out_dir)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    return output


def _legend_columns(labels: list[str]) -> int:
    """Use a fixed legend column count for predictable layout."""
    return min(3, max(1, len(labels)))


def _build_legend_labels(
    timer_ids: list[str],
    timer_db: dict[str, TimerDef],
    include_total_reference: bool = True,
) -> list[str]:
    """Build wrapped legend labels in plotting order."""
    labels: list[str] = []
    if include_total_reference:
        labels.extend(["Total", "Perfect scaling"])
    labels.extend(
        _timer_display_name(timer_id, timer_db) for timer_id in timer_ids
    )
    return [_wrap_legend_label(label) for label in labels]


def _wrap_legend_label(label: str, width: int = 30) -> str:
    """Wrap a legend label onto multiple lines for readability."""
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def _should_use_log_scale(values: list[float]) -> bool:
    """Use a log y-axis when the value range spans two orders of magnitude."""
    positive_values = [value for value in values if value > 0]
    if len(positive_values) < 2:
        return False
    return max(positive_values) / min(positive_values) >= 100.0


def _build_scaling_table(
    scaling_data: list[ScalingLogData],
    timer_ids: list[str],
    timer_db: dict[str, TimerDef],
    min_percent_threshold: float,
    run_statistic: Literal["sum", "mean"],
    rank_source: Literal["root", "all"],
) -> str:
    """Build the main scaling summary table."""
    headers = ["Timer", "Max %", "Slope", "Scaling"]
    headers.extend(_log_column_labels(scaling_data))

    rows: list[list[str]] = []
    if run_statistic == "sum":
        total_values = [
            _main_total_value(log_data, run_statistic, rank_source)
            for log_data in scaling_data
        ]
        total_slope = _fit_scaling_slope(
            [log_data.rank_count for log_data in scaling_data], total_values
        )
        rows.append(
            [
                "Total",
                "100.0%",
                _format_slope(total_slope),
                _describe_scaling(total_slope),
                *[_format_time(value) for value in total_values],
            ]
        )

    for timer_id in timer_ids:
        values = [
            _main_timer_value(log_data, timer_id, run_statistic, rank_source)
            for log_data in scaling_data
        ]
        slope = _fit_scaling_slope(
            [log_data.rank_count for log_data in scaling_data], values
        )
        max_percent = max(
            log_data.timer_percentages.get(timer_id, 0.0)
            for log_data in scaling_data
        )
        rows.append(
            [
                _truncate(_timer_display_name(timer_id, timer_db), 64),
                f"{max_percent:.1f}%",
                _format_slope(slope),
                _describe_scaling(slope),
                *[
                    _format_time(
                        _main_timer_value(
                            log_data, timer_id, run_statistic, rank_source
                        )
                    )
                    for log_data in scaling_data
                ],
            ]
        )

    title = (
        "TIMER SCALING SUMMARY "
        "("
        f"{_run_statistic_title(run_statistic)}, "
        f"{_rank_source_title(rank_source)}"
        ") "
        f"(timers above {min_percent_threshold:.1f}% in at least one log)"
    )
    return create_ascii_table(headers, rows, title)


def _build_scaling_analysis(
    scaling_data: list[ScalingLogData],
    timer_ids: list[str],
    timer_db: dict[str, TimerDef],
    min_percent_threshold: float,
    used_log_scale: bool,
    run_statistic: Literal["sum", "mean"],
    rank_source: Literal["root", "all"],
) -> str:
    """Create a concise textual analysis of the scaling results."""
    rank_counts = [log_data.rank_count for log_data in scaling_data]
    total_times = [
        _main_total_value(log_data, run_statistic, rank_source)
        for log_data in scaling_data
    ]
    total_slope = _fit_scaling_slope(rank_counts, total_times)

    timer_slopes: list[tuple[str, float]] = []
    for timer_id in timer_ids:
        slope = _fit_scaling_slope(
            rank_counts,
            [
                _main_timer_value(
                    log_data, timer_id, run_statistic, rank_source
                )
                for log_data in scaling_data
            ],
        )
        if slope is not None:
            timer_slopes.append((timer_id, slope))

    timer_slopes.sort(key=lambda item: item[1])
    strongest = timer_slopes[:3]
    weakest = [
        item
        for item in sorted(
            timer_slopes, key=lambda item: item[1], reverse=True
        )
        if item not in strongest
    ][:3]

    lines = [
        "Scaling analysis:",
        (
            f"- Included {len(timer_ids)} timer(s) using a "
            f"{min_percent_threshold:.1f}% minimum contribution threshold."
        ),
        (
            f"- This view shows {_run_statistic_description(run_statistic)} "
            f"using {_rank_source_description(rank_source)}."
        ),
    ]

    if run_statistic == "sum":
        lines.extend(
            [
                (
                    f"- Total measured timer time changes from "
                    f"{_format_time(total_times[0])} at {rank_counts[0]} "
                    f"ranks to {_format_time(total_times[-1])} at "
                    f"{rank_counts[-1]} ranks."
                ),
                (
                    "- Overall strong-scaling slope: "
                    f"{_format_slope(total_slope)}."
                ),
                (
                    "- The plot includes a solid Total line and a dashed "
                    "perfect 1/N scaling reference anchored to the "
                    "smallest-rank run."
                ),
            ]
        )
    else:
        lines.append(
            "- The mean-per-call view does not show a Total line because "
            "different timers have different call counts, so a single total "
            "mean would be misleading."
        )

    if strongest:
        lines.append(
            "- Strongest improving timers: "
            + ", ".join(
                f"{_timer_display_name(timer_id, timer_db)} "
                f"({_format_slope(slope)})"
                for timer_id, slope in strongest
            )
            + "."
        )

    if weakest:
        lines.append(
            "- Weakest or regressing timers: "
            + ", ".join(
                f"{_timer_display_name(timer_id, timer_db)} "
                f"({_format_slope(slope)})"
                for timer_id, slope in weakest
            )
            + "."
        )

    if _collect_verbose_ranks(scaling_data):
        lines.append(
            "- Per-rank timer plots were generated because multiple emitting "
            "ranks were detected in the input logs."
        )

    incomplete_logs = [
        (
            f"{log_data.label} "
            f"({log_data.emitted_rank_count}/{log_data.rank_count})"
        )
        for log_data in scaling_data
        if log_data.emitted_rank_count < log_data.rank_count
    ]
    if incomplete_logs:
        lines.append(
            "- Some logs did not contain timer lines from every rank. This "
            f"matters when using {_rank_source_description(rank_source)}: "
            + ", ".join(incomplete_logs)
            + "."
        )

    if used_log_scale:
        lines.append(
            "- The y-axis switches to log scale when selected timer values "
            "span at least two orders of magnitude."
        )

    return "\n".join(lines)


def _build_rank_imbalance_table(
    scaling_data: list[ScalingLogData],
    timer_ids: list[str],
    timer_db: dict[str, TimerDef],
) -> str | None:
    """Summarize per-rank imbalance for the largest run."""
    verbose_logs = [
        log_data
        for log_data in scaling_data
        if len(log_data.per_rank_totals) > 1
    ]
    if not verbose_logs:
        return None

    reference = max(verbose_logs, key=lambda item: item.rank_count)
    headers = ["Timer", "Mean/rank", "Max/rank", "Min/rank", "Max/Min"]
    rows: list[list[str]] = []

    for timer_id in timer_ids:
        values = [
            rank_totals.get(timer_id, 0.0)
            for rank_totals in reference.per_rank_totals.values()
        ]
        positive_values = [value for value in values if value > 0]
        if not positive_values:
            continue
        rows.append(
            [
                _truncate(_timer_display_name(timer_id, timer_db), 64),
                _format_time(float(np.mean(positive_values))),
                _format_time(max(positive_values)),
                _format_time(min(positive_values)),
                f"{max(positive_values) / min(positive_values):.2f}x",
            ]
        )

    if not rows:
        return None

    title = f"PER-RANK IMBALANCE AT {reference.rank_count} RANKS"
    return create_ascii_table(headers, rows, title)


def _create_timer_rank_scaling_plot(
    scaling_data: list[ScalingLogData],
    timer_id: str,
    timer_db: dict[str, TimerDef],
    ranks: list[int],
    output_path: str | None,
    prefix: str | None,
    show_plot: bool,
    out_dir: str,
    filename: str,
) -> Path | None:
    """Create one scaling plot for a single timer with one line per rank."""
    fig = plt.figure(figsize=(10.5, 14.0))
    grid = fig.add_gridspec(3, 1, height_ratios=[10.5, 3.0, 1.0], hspace=0.04)
    ax = fig.add_subplot(grid[0])
    speedup_ax = fig.add_subplot(grid[1], sharex=ax)
    legend_ax = fig.add_subplot(grid[2])
    legend_ax.axis("off")
    ax.set_box_aspect(1)

    all_values: list[float] = []
    plotted = 0

    for rank in ranks:
        x_values: list[int] = []
        y_values: list[float] = []
        for log_data in scaling_data:
            value = log_data.per_rank_totals.get(rank, {}).get(timer_id)
            if value is None or value <= 0:
                continue
            x_values.append(log_data.rank_count)
            y_values.append(value)

        if len(x_values) < 2:
            continue

        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            markersize=5,
            label=f"Rank {rank}",
        )
        color = ax.lines[-1].get_color()
        speedup_ax.plot(
            x_values,
            _compute_speedup_series(y_values),
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
        )
        perfect_y_values = _build_perfect_scaling_series(x_values, y_values)
        ax.plot(
            x_values,
            perfect_y_values,
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            color=color,
        )
        speedup_ax.plot(
            x_values,
            _build_perfect_speedup_series(x_values),
            linestyle="--",
            linewidth=1.2,
            alpha=0.35,
            color=color,
        )
        all_values.extend(y_values)
        all_values.extend(perfect_y_values)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_ylabel("Summed Timer Time Over Run (ms)")
    ax.set_title(
        f"{_timer_display_name(timer_id, timer_db)} Across Ranks",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(sorted({item.rank_count for item in scaling_data}))
    speedup_ax.set_xlabel("Number of Ranks")
    speedup_ax.set_ylabel("Relative Speedup")
    speedup_ax.grid(True, alpha=0.3, linestyle="--")
    speedup_ax.axhline(1.0, color="0.5", linewidth=1.0, alpha=0.6)
    speedup_ax.set_xticks(sorted({item.rank_count for item in scaling_data}))
    plt.setp(ax.get_xticklabels(), visible=False)

    if _should_use_log_scale(all_values):
        ax.set_yscale("log")

    legend_columns = min(3, max(1, plotted))
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=legend_columns,
        frameon=False,
    )

    output = create_output_path(output_path, prefix, filename, out_dir)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    return output


def _collect_verbose_ranks(scaling_data: list[ScalingLogData]) -> list[int]:
    """Collect ranks that have enough data for per-rank scaling plots."""
    if not any(len(log_data.per_rank_totals) > 1 for log_data in scaling_data):
        return []

    rank_occurrences: dict[int, set[int]] = defaultdict(set)
    for log_data in scaling_data:
        for rank in log_data.per_rank_totals:
            rank_occurrences[rank].add(log_data.rank_count)

    return sorted(
        rank
        for rank, rank_counts in rank_occurrences.items()
        if len(rank_counts) >= 2
    )


def _collect_comparable_ranks(scaling_data: list[ScalingLogData]) -> list[int]:
    """Collect ranks that appear in at least two runs for timer-rank plots."""
    rank_occurrences: dict[int, int] = defaultdict(int)
    for log_data in scaling_data:
        for rank in log_data.per_rank_totals:
            rank_occurrences[rank] += 1

    return sorted(
        rank for rank, count in rank_occurrences.items() if count >= 2
    )


def _plot_would_use_log_scale(
    scaling_data: list[ScalingLogData],
    timer_ids: list[str],
    run_statistic: Literal["sum", "mean"],
    rank_source: Literal["root", "all"],
    per_rank: int | None = None,
) -> bool:
    """Return whether the corresponding scaling plot should use log scale."""
    values: list[float] = []
    for timer_id in timer_ids:
        for log_data in scaling_data:
            totals = (
                _build_rank_series_dict(log_data, per_rank, run_statistic)
                if per_rank is not None
                else _build_main_series_dict(
                    log_data, run_statistic, rank_source
                )
            )
            value = totals.get(timer_id)
            if value is not None and value > 0:
                values.append(value)
    if run_statistic == "sum":
        _, total_values = _build_total_series(
            scaling_data, per_rank, run_statistic, rank_source
        )
        values.extend(value for value in total_values if value > 0)
    return _should_use_log_scale(values)


def _build_total_series(
    scaling_data: list[ScalingLogData],
    per_rank: int | None = None,
    run_statistic: Literal["sum", "mean"] = "sum",
    rank_source: Literal["root", "all"] = "root",
) -> tuple[list[int], list[float]]:
    """Build the total-time series for the overall or per-rank plot."""
    x_values: list[int] = []
    y_values: list[float] = []

    for log_data in scaling_data:
        if per_rank is None:
            value = _main_total_value(log_data, run_statistic, rank_source)
        else:
            value = _rank_total_value(log_data, per_rank, run_statistic)
        if value <= 0:
            continue
        x_values.append(log_data.rank_count)
        y_values.append(value)

    return x_values, y_values


def _build_perfect_scaling_series(
    x_values: list[int], y_values: list[float]
) -> list[float]:
    """Build a perfect 1/N scaling reference anchored to the first point."""
    if not x_values or not y_values:
        return []

    anchor_ranks = x_values[0]
    anchor_value = y_values[0]
    return [
        anchor_value * anchor_ranks / rank_count for rank_count in x_values
    ]


def _build_perfect_speedup_series(x_values: list[int]) -> list[float]:
    """Build the ideal speedup series corresponding to perfect scaling."""
    if not x_values:
        return []
    anchor_ranks = x_values[0]
    return [rank_count / anchor_ranks for rank_count in x_values]


def _compute_speedup_series(y_values: list[float]) -> list[float]:
    """Convert a timing series into speedup against its first point."""
    if not y_values:
        return []
    anchor_value = y_values[0]
    return [anchor_value / value for value in y_values]


def _build_main_series_dict(
    log_data: ScalingLogData,
    run_statistic: Literal["sum", "mean"],
    rank_source: Literal["root", "all"],
) -> dict[str, float]:
    """Build the timer series used in the main scaling plot/table."""
    if rank_source == "root":
        return _build_rank_series_dict(log_data, 0, run_statistic)
    if rank_source == "all":
        if run_statistic == "sum":
            return log_data.timer_totals
        if run_statistic == "mean":
            return {
                timer_id: log_data.timer_totals[timer_id]
                / log_data.timer_call_counts[timer_id]
                for timer_id in log_data.timer_totals
                if log_data.timer_call_counts.get(timer_id, 0) > 0
            }
    raise ValueError(
        "Unsupported combination: "
        f"run_statistic={run_statistic}, rank_source={rank_source}"
    )


def _build_rank_series_dict(
    log_data: ScalingLogData,
    rank: int,
    run_statistic: Literal["sum", "mean"],
) -> dict[str, float]:
    """Build a timer series for a single rank."""
    rank_totals = log_data.per_rank_totals.get(rank, {})
    if run_statistic == "sum":
        return rank_totals
    rank_counts = log_data.per_rank_call_counts.get(rank, {})
    return {
        timer_id: total / rank_counts[timer_id]
        for timer_id, total in rank_totals.items()
        if rank_counts.get(timer_id, 0) > 0
    }


def _main_timer_value(
    log_data: ScalingLogData,
    timer_id: str,
    run_statistic: Literal["sum", "mean"],
    rank_source: Literal["root", "all"],
) -> float:
    """Get the value of one timer for the main scaling view."""
    return _build_main_series_dict(log_data, run_statistic, rank_source).get(
        timer_id, 0.0
    )


def _main_total_value(
    log_data: ScalingLogData,
    run_statistic: Literal["sum", "mean"],
    rank_source: Literal["root", "all"],
) -> float:
    """Get the total value for the main scaling view."""
    if rank_source == "root":
        return _rank_total_value(log_data, 0, run_statistic)
    if run_statistic == "sum":
        return log_data.total_time
    if run_statistic == "mean":
        if log_data.total_call_count == 0:
            return 0.0
        return log_data.total_time / log_data.total_call_count
    raise ValueError(
        "Unsupported combination: "
        f"run_statistic={run_statistic}, rank_source={rank_source}"
    )


def _rank_total_value(
    log_data: ScalingLogData,
    rank: int,
    run_statistic: Literal["sum", "mean"],
) -> float:
    """Get a total value for one rank."""
    rank_totals = log_data.per_rank_totals.get(rank, {})
    if run_statistic == "sum":
        return sum(rank_totals.values())
    rank_counts = log_data.per_rank_call_counts.get(rank, {})
    total_calls = sum(rank_counts.values())
    if total_calls == 0:
        return 0.0
    return sum(rank_totals.values()) / total_calls


def _run_statistic_title(run_statistic: Literal["sum", "mean"]) -> str:
    """Return a short title fragment for the run statistic."""
    if run_statistic == "sum":
        return "Summed Over Run"
    return "Mean Per Call"


def _run_statistic_description(run_statistic: Literal["sum", "mean"]) -> str:
    """Return a sentence fragment for the run statistic."""
    if run_statistic == "sum":
        return "summed timer time over the run"
    return "average timer time per call over the run"


def _rank_source_title(rank_source: Literal["root", "all"]) -> str:
    """Return a short title fragment for rank selection."""
    if rank_source == "root":
        return "Rank 0 Only"
    return "All Emitting Ranks"


def _rank_source_description(rank_source: Literal["root", "all"]) -> str:
    """Return a sentence fragment for rank selection."""
    if rank_source == "root":
        return "rank 0 only"
    return "all emitting ranks present in the log"


def _fit_scaling_slope(
    rank_counts: list[int], values: list[float]
) -> float | None:
    """Fit log(time) against log(ranks)."""
    points = [
        (rank_count, value)
        for rank_count, value in zip(rank_counts, values, strict=False)
        if rank_count > 0 and value > 0
    ]
    unique_ranks = {rank_count for rank_count, _ in points}
    if len(points) < 2 or len(unique_ranks) < 2:
        return None

    x = np.log10([rank_count for rank_count, _ in points])
    y = np.log10([value for _, value in points])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _format_slope(slope: float | None) -> str:
    """Format a scaling slope."""
    if slope is None:
        return "n/a"
    return f"{slope:.2f}"


def _describe_scaling(slope: float | None) -> str:
    """Map a slope to a short qualitative description."""
    if slope is None:
        return "insufficient"
    if slope <= -1.15:
        return "superlinear"
    if slope <= -0.75:
        return "strong"
    if slope <= -0.25:
        return "moderate"
    if slope <= 0.10:
        return "weak"
    return "regressing"


def _log_column_labels(scaling_data: list[ScalingLogData]) -> list[str]:
    """Build compact per-log column labels."""
    counts_seen: dict[int, int] = defaultdict(int)
    labels: list[str] = []
    for log_data in scaling_data:
        counts_seen[log_data.rank_count] += 1
        suffix = ""
        if (
            sum(
                1
                for item in scaling_data
                if item.rank_count == log_data.rank_count
            )
            > 1
        ):
            suffix = f"#{counts_seen[log_data.rank_count]}"
        labels.append(f"{log_data.rank_count}r{suffix}")
    return labels


def _format_time(value: float) -> str:
    """Format time values for tables."""
    if value >= 1000.0:
        return f"{value / 1000.0:.2f}s"
    return f"{value:.1f}ms"


def _truncate(text: str, limit: int) -> str:
    """Truncate a string for ASCII table output."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _slugify(text: str) -> str:
    """Convert display text into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return slug.strip("_") or "timer"
