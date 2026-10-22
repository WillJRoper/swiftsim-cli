"""``analyse scaling timers``: how each timer scales with the rank count."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from swiftsim_cli.modes.analyse.scaling.data import (
    LogTimers,
    RankSource,
    Statistic,
    clear_outputs,
    describe_slope,
    fit_slope,
    format_slope,
    format_time,
    load_log_timers,
    require_rank_counts,
)
from swiftsim_cli.modes.analyse.scaling.plots import scaling_plot
from swiftsim_cli.utilities.io import create_output_path, slugify
from swiftsim_cli.utilities.messages import plural, warn
from swiftsim_cli.utilities.tables import create_ascii_table

OUT_SUBDIR = "scaling_timers_analysis"

# Step counts differing by more than this fraction trigger a warning.
STEP_COUNT_TOLERANCE = 0.01

_VIEW = {
    ("sum", "root"): "exclusive time summed over the run, rank 0",
    ("sum", "all"): "exclusive time summed over the run, mean over the "
    "ranks with timers",
    ("mean", "root"): "exclusive time per call, rank 0",
    ("mean", "all"): "exclusive time per call, all ranks with timers",
}


def select_timers(
    logs: list[LogTimers], threshold: float, source: RankSource
) -> list[str]:
    """Timers above ``threshold`` % of the step time in at least one log."""
    keys = {k for log in logs for k in log.series("sum", source)}
    best = {k: max(log.percent(k, source) for log in logs) for k in keys}
    return sorted(
        (k for k in keys if best[k] >= threshold),
        key=lambda k: (-best[k], k),
    )


def _column_labels(logs: list[LogTimers]) -> list[str]:
    counts = [log.n_ranks for log in logs]
    seen: dict[int, int] = {}
    labels = []
    for n in counts:
        seen[n] = seen.get(n, 0) + 1
        labels.append(f"{n}r" + (f"#{seen[n]}" if counts.count(n) > 1 else ""))
    return labels


def logs_table(logs: list[LogTimers]) -> str:
    """One row per log: ranks, steps, step time, timed share, unmatched."""
    rows = [
        [
            log.n_ranks,
            f"{len(log.exclusive)}",
            log.n_steps,
            format_time(log.step_time),
            f"{100 * log.total('sum', 'root') / log.step_time:.1f}%"
            if log.step_time
            else "-",
            f"{log.unmatched:.1f}%",
            log.label,
        ]
        for log in logs
    ]
    return create_ascii_table(
        [
            "Ranks",
            "Ranks w/ timers",
            "Steps",
            "Step time",
            "Timed (rank 0)",
            "Unmatched lines",
            "Log",
        ],
        rows,
        "SCALING INPUTS",
    )


def scaling_table(
    logs: list[LogTimers],
    timers: list[str],
    threshold: float,
    statistic: Statistic,
    source: RankSource,
) -> str:
    """Values, slope and qualitative scaling per timer (plus the total)."""
    ranks = [log.n_ranks for log in logs]
    rows = []
    if statistic == "sum":
        totals = [log.total(statistic, source) for log in logs]
        slope = fit_slope(ranks, totals)
        timed = max(
            100 * t / log.step_time if log.step_time else 0.0
            for t, log in zip(totals, logs)
        )
        rows.append(
            [
                "Total (timed)",
                f"{timed:.1f}%",
                format_slope(slope),
                describe_slope(slope),
                *map(format_time, totals),
            ]
        )
    series = [log.series(statistic, source) for log in logs]
    for key in timers:
        values = [s.get(key, 0.0) for s in series]
        slope = fit_slope(ranks, values)
        rows.append(
            [
                key if len(key) <= 64 else key[:61] + "...",
                f"{max(log.percent(key, source) for log in logs):.1f}%",
                format_slope(slope),
                describe_slope(slope),
                *map(format_time, values),
            ]
        )
    return create_ascii_table(
        ["Timer", "Max % of step", "Slope", "Scaling", *_column_labels(logs)],
        rows,
        f"TIMER SCALING ({_VIEW[statistic, source]}; timers above "
        f"{threshold:g}% of the step time in at least one log)",
    )


def scaling_notes(
    logs: list[LogTimers],
    timers: list[str],
    threshold: float,
    statistic: Statistic,
    source: RankSource,
) -> str:
    """A short text summary of one view."""
    ranks = [log.n_ranks for log in logs]
    series = [log.series(statistic, source) for log in logs]
    slopes = [
        (key, slope)
        for key in timers
        if (slope := fit_slope(ranks, [s.get(key, 0.0) for s in series]))
        is not None
    ]
    slopes.sort(key=lambda item: item[1])
    lines = [
        f"Scaling ({_VIEW[statistic, source]}):",
        f"- {plural(len(timers), 'timer')} above {threshold:g}% of the step "
        "time in at least one log.",
    ]
    if statistic == "sum":
        totals = [log.total(statistic, source) for log in logs]
        lines.append(
            f"- Total timed time goes from {format_time(totals[0])} at "
            f"{plural(ranks[0], 'rank')} to {format_time(totals[-1])} at "
            f"{plural(ranks[-1], 'rank')} (slope "
            f"{format_slope(fit_slope(ranks, totals))}; -1 is perfect "
            "strong scaling)."
        )
    else:
        lines.append(
            "- No total in the per-call view: timers have different call "
            "counts."
        )
    if slopes:
        lines.append(
            "- Best scaling: "
            + ", ".join(f"{k} ({format_slope(s)})" for k, s in slopes[:3])
            + "."
        )
        worst = [item for item in slopes[::-1] if item not in slopes[:3]]
        if worst:
            lines.append(
                "- Worst scaling: "
                + ", ".join(f"{k} ({format_slope(s)})" for k, s in worst[:3])
                + "."
            )
    return "\n".join(lines)


def imbalance_table(logs: list[LogTimers], timers: list[str]) -> str | None:
    """Spread of each timer over the ranks of the largest -v 2 run.

    Ranks that never ran a timer count as zero.
    """
    multi = [log for log in logs if len(log.exclusive) > 1]
    if not multi:
        return None
    log = max(multi, key=lambda item: item.n_ranks)
    rows = []
    for key in timers:
        values = [t.get(key, 0.0) for t in log.exclusive.values()]
        mean = float(np.mean(values))
        if mean <= 0:
            continue
        rows.append(
            [
                key if len(key) <= 64 else key[:61] + "...",
                format_time(mean),
                format_time(max(values)),
                format_time(min(values)),
                f"{max(values) / mean:.2f}",
            ]
        )
    if not rows:
        return None
    return create_ascii_table(
        ["Timer", "Mean/rank", "Max/rank", "Min/rank", "Max/mean"],
        rows,
        f"PER-RANK IMBALANCE ({log.label}: "
        f"{plural(len(log.exclusive), 'rank')} with timers of "
        f"{log.n_ranks}, {plural(log.n_steps, 'step')})",
    )


def _check_step_counts(logs: list[LogTimers]) -> None:
    counts = [log.n_steps for log in logs]
    if max(counts) > (1 + STEP_COUNT_TOLERANCE) * min(counts):
        warn(
            f"the logs cover different numbers of steps ({min(counts)} to "
            f"{max(counts)}); summed times are not comparable. Use --steps "
            "to pick the same steps from each log."
        )


def analyse_swift_scaling(
    log_files: list[str],
    output_path: str | Path | None = None,
    prefix: str | None = None,
    show_plot: bool = False,
    step_range: tuple[int, int] | None = None,
    min_percent_threshold: float = 5.0,
    rank_source: RankSource = "root",
    swift_src: Path | None = None,
    match_revision: bool = True,
    regenerate_db: bool = False,
) -> None:
    """Analyse how the timers scale across logs of different rank counts.

    Writes ``scaling_summary.txt``, the scaling plots and, for ``-v 2``
    logs, per-rank plots and ``rank_imbalance_summary.txt`` into
    ``<output_path>/scaling_timers_analysis/``.
    """
    if len(log_files) < 2:
        raise ValueError("Scaling analysis requires at least two log files")
    if step_range is not None and step_range[0] > step_range[1]:
        raise ValueError("Step range must be ordered as START END")
    if min_percent_threshold <= 0:
        raise ValueError("--min-percent-threshold must be positive")
    if rank_source not in ("root", "all"):
        raise ValueError("rank_source must be one of 'root' or 'all'")

    print(
        f"Analysing timer scaling across {plural(len(log_files), 'log file')}"
    )
    logs = []
    for path in log_files:
        print(f"\nProcessing scaling input: {path}")
        logs.append(
            load_log_timers(
                path, step_range, swift_src, match_revision, regenerate_db
            )
        )
    logs.sort(key=lambda log: (log.n_ranks, log.label))
    require_rank_counts([log.n_ranks for log in logs])
    _check_step_counts(logs)
    if rank_source == "all":
        partial = [
            f"{log.label} ({len(log.exclusive)}/{log.n_ranks})"
            for log in logs
            if len(log.exclusive) < log.n_ranks
        ]
        if partial:
            warn(
                "--rank-source all averages over the ranks that printed "
                "timers, but some logs have timers from fewer ranks than "
                "they ran with (-v 1 prints rank 0's only): "
                + ", ".join(partial)
            )
    timers = select_timers(logs, min_percent_threshold, rank_source)
    if not timers:
        raise ValueError(
            "No timers satisfied the selected minimum percentage threshold"
        )

    def out(name: str) -> Path:
        return create_output_path(output_path, prefix, name, OUT_SUBDIR)

    summary = out("scaling_summary.txt")
    clear_outputs(
        summary.parent,
        prefix,
        "timer_scaling_*.png",
        "timer_rank_scaling_*.png",
        "rank_imbalance_summary.txt",
    )
    ranks = [log.n_ranks for log in logs]
    sections = [logs_table(logs)]
    created = []
    for statistic in ("sum", "mean"):
        series = [log.series(statistic, rank_source) for log in logs]
        lines = {k: [s.get(k, 0.0) for s in series] for k in timers}
        total = None
        if statistic == "sum":
            total = "Total (timed)"
            lines = {
                total: [log.total("sum", rank_source) for log in logs],
                **lines,
            }
        plot = scaling_plot(
            ranks,
            lines,
            out(f"timer_scaling_{statistic}_{rank_source}.png"),
            f"Timer scaling: {_VIEW[statistic, rank_source]}",
            "Exclusive time per call (ms)"
            if statistic == "mean"
            else "Exclusive time over the run (ms)",
            total=total,
            show=show_plot,
        )
        if plot is not None:
            created.append(plot)
        sections.append(
            scaling_table(
                logs, timers, min_percent_threshold, statistic, rank_source
            )
            + "\n\n"
            + scaling_notes(
                logs, timers, min_percent_threshold, statistic, rank_source
            )
        )
    text = "\n\n".join(sections)
    print("\n" + text)
    summary.write_text(text + "\n", encoding="utf-8")
    created.append(summary)

    # -v 2 logs: the same view for each rank present in several logs, and
    # each timer with one line per rank.
    if any(len(log.exclusive) > 1 for log in logs):
        common = sorted(
            r
            for r in {r for log in logs for r in log.exclusive}
            if sum(r in log.exclusive for log in logs) >= 2
        )
        for rank in common:
            lines = {
                k: [log.exclusive.get(rank, {}).get(k, 0.0) for log in logs]
                for k in timers
            }
            plot = scaling_plot(
                ranks,
                lines,
                out(f"timer_scaling_rank_{rank:04d}.png"),
                f"Timer scaling, rank {rank}",
                "Exclusive time over the run (ms)",
                show=show_plot,
            )
            if plot is not None:
                created.append(plot)
        if len(common) > 1:
            for key in timers:
                plot = scaling_plot(
                    ranks,
                    {
                        f"Rank {r}": [
                            log.exclusive.get(r, {}).get(key, 0.0)
                            for log in logs
                        ]
                        for r in common
                    },
                    out(
                        "timer_rank_scaling_"
                        + slugify(key, fallback="timer", lowercase=True)
                        + ".png"
                    ),
                    f"{key} by rank",
                    "Exclusive time over the run (ms)",
                    show=show_plot,
                )
                if plot is not None:
                    created.append(plot)
        imbalance = imbalance_table(logs, timers)
        if imbalance is not None:
            print("\n" + imbalance)
            imbalance_path = out("rank_imbalance_summary.txt")
            imbalance_path.write_text(imbalance + "\n", encoding="utf-8")
            created.append(imbalance_path)

    print("\nCreated outputs:")
    for created_path in created:
        print(f"  - {created_path}")
