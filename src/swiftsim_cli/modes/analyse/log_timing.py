"""``analyse log``: where the time goes in a SWIFT run.

Timer lines in the log are matched against the timers parsed from the
SWIFT source (by default at the revision the log was written by), nested
into a runtime call tree per step and MPI rank, and aggregated by call
path. All percentages are of the step-table wall-clock time, so the
top-level breakdown (including the untimed remainder) sums to 100%.

Run SWIFT with ``-v 1`` (rank 0 timers) or ``-v 2`` (every rank).
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from swiftsim_cli.modes.common_args import (
    add_output_arguments,
    add_rank_argument,
    add_steps_argument,
    existing_dir,
    existing_file,
    positive_int,
)
from swiftsim_cli.timing.log_scan import (
    LogScan,
    Step,
    read_log_revision,
    scan_log,
)
from swiftsim_cli.timing.timer_db import resolve_timer_db
from swiftsim_cli.timing.timing_hierarchy import (
    COVERAGE_HIGH,
    StepTree,
    aggregate,
    between_steps_table,
    build_step_trees,
    coverage_table,
    flat_table,
    function_table,
    rank_imbalance_table,
    samples_by_key,
    select_steps,
    top_level_table,
    tree_table,
    write_coverage_csv,
)
from swiftsim_cli.timing.timing_plots import make_plots
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.messages import plural, reading, warn
from swiftsim_cli.utilities.tables import create_ascii_table

# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "log_analysis"

# Unmatched timer lines above this share (percent) trigger a warning.
UNMATCHED_WARN_PERCENT = 5.0

# Timers whose merged subtrees are shown when --hierarchy-functions is unset
DEFAULT_HIERARCHY_FUNCTIONS = (
    "engine_prepare",
    "engine_rebuild",
    "space_rebuild",
    "engine_maketasks",
)


def add_log_arguments(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add the ``log`` subcommand and its arguments."""
    parser = subparsers.add_parser(
        "log",
        help="Analyse the timers in a SWIFT log file (run SWIFT with -v 1 "
        "for rank 0 timers, -v 2 for every rank).",
        description="Analyse where the step time goes from the timers in a "
        "SWIFT log, matched against the timers in the SWIFT source. "
        "Requires SWIFT run with -v 1 (rank 0 timers) or -v 2 (every "
        "rank).",
    )
    parser.set_defaults(func=run_swift_log_timing)
    parser.add_argument(
        "log_file", type=existing_file, metavar="FILE", help="SWIFT log file."
    )
    add_output_arguments(parser, OUT_SUBDIR)
    add_steps_argument(parser)
    parser.add_argument(
        "--top-n",
        type=positive_int,
        default=20,
        metavar="N",
        help="Rows per table, children per node in call trees and bars per "
        "chart (default: 20).",
    )
    parser.add_argument(
        "--hierarchy-functions",
        nargs="+",
        default=None,
        metavar="FUNC",
        help="Timers (function names or timer keys) whose subtrees are "
        "shown merged over all their callers, with a breakdown by caller "
        f"(default: {', '.join(DEFAULT_HIERARCHY_FUNCTIONS)}).",
    )
    add_rank_argument(
        parser,
        default=0,
        help="MPI rank to analyse; 'all' gives one analysis per rank plus "
        "an imbalance table (ranks are never summed).",
    )
    add_timer_db_arguments(parser)


def add_timer_db_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options choosing the source the timers are read from."""
    parser.add_argument(
        "--swift-src",
        type=existing_dir,
        default=None,
        metavar="DIR",
        help="SWIFT source tree to read the timers from (default: the "
        "active profile's checkout, at each log's revision).",
    )
    parser.add_argument(
        "--no-match-revision",
        dest="match_revision",
        action="store_false",
        help="Use the checkout as it is instead of the revision printed "
        "in the log.",
    )
    parser.add_argument(
        "--regenerate-db",
        action="store_true",
        help="Rebuild the cached timer database even if it is current.",
    )


def run_swift_log_timing(args: argparse.Namespace) -> None:
    """Entry point for ``analyse log``."""
    analyse_swift_log_timings(
        log_file=str(args.log_file),
        output_path=args.out,
        prefix=args.prefix,
        show_plot=args.show,
        top_n=args.top_n,
        hierarchy_functions=args.hierarchy_functions,
        rank=args.rank,
        step_range=tuple(args.steps) if args.steps else None,
        swift_src=args.swift_src,
        match_revision=args.match_revision,
        regenerate_db=args.regenerate_db,
    )


def _seconds(ms: float, total: float) -> str:
    share = f" ({100 * ms / total:.1f}%)" if total > 0 else ""
    return f"{ms / 1000:.1f} s{share}"


def _run_summary(scan: LogScan, steps: list[Step]) -> list[str]:
    """How the run's elapsed time splits into init, steps and the rest."""
    total = scan.last_elapsed_s * 1000
    lines = []
    if total <= 0:
        pass  # no rank-0 elapsed stamps to split
    elif scan.init_end_s is not None:
        init = scan.init_end_s * 1000
        all_steps = sum(s.wallclock_ms or 0 for s in scan.complete_steps)
        tail = total - init - all_steps
        lines.append(
            f"Run time {total / 1000:.1f} s = initialisation "
            f"{_seconds(init, total)} + {len(scan.complete_steps)} steps "
            f"{_seconds(all_steps, total)} + after the last step row "
            f"{_seconds(tail, total)}"
            + (" [restarted run]" if scan.restart else "")
        )
    else:
        lines.append(f"Run time {total / 1000:.1f} s (no step table found)")
    if scan.n_ranks > 1:
        lines.append(
            f"MPI ranks: {scan.n_ranks}; ranks that printed timers: "
            f"{sorted(scan.timer_ranks)}"
        )
    if steps:
        step_time = sum(s.wallclock_ms or 0 for s in steps)
        lines.append(
            f"Steps analysed: {len(steps)} (steps {steps[0].number}-"
            f"{steps[-1].number}), step time {step_time / 1000:.1f} s"
        )
    else:
        lines.append("No complete steps to analyse.")
    return lines


def _coverage_line(trees: list[StepTree]) -> str:
    covs = [t.coverage for t in trees if t.coverage is not None]
    if not covs:
        return "Per-step timer coverage: n/a"
    return (
        f"Per-step timer coverage (rank {trees[0].rank}): min "
        f"{min(covs):.1f}%, median {statistics.median(covs):.1f}%, max "
        f"{max(covs):.1f}%"
    )


def unmatched_percent(scan: LogScan) -> float:
    """Share of the timer lines no known timer matched (percent)."""
    unmatched = sum(scan.unmatched.values())
    total = scan.matched + unmatched
    return 100 * unmatched / total if total else 0.0


def check_scan(scan: LogScan, name: str) -> None:
    """Warn about unmatched timers and damaged lines in a scanned log.

    Args:
        scan: The scanned log.
        name: How to refer to the log in the warnings.
    """
    percent = unmatched_percent(scan)
    if percent > UNMATCHED_WARN_PERCENT:
        warn(
            f"{percent:.1f}% of the timer lines in {name} match no timer in "
            "the SWIFT source; the source revision probably differs from "
            "the one that wrote the log (check --swift-src / "
            "--no-match-revision). Unmatched timers may be misplaced in the "
            "call tree."
        )
    if scan.cut_lines:
        warn(
            f"{plural(scan.cut_lines, 'line')} of {name} had another line "
            "appended (interleaved output); recovered "
            f"{plural(scan.recovered, 'timer line')} and "
            f"{plural(scan.recovered_rows, 'step row')}, dropped "
            f"{plural(scan.lost, 'unreadable fragment')}."
        )
    if scan.step_gaps:
        first, nxt = scan.step_gaps[0]
        warn(
            f"{plural(len(scan.step_gaps), 'step row')} of {name} out of "
            f"sequence (first: step {nxt} after {first}), probably damaged "
            "by interleaved output; the steps closed by them span several "
            "steps and are left out of the per-step analysis."
        )


def _match_lines(scan: LogScan, db_status: str) -> list[str]:
    unmatched = sum(scan.unmatched.values())
    lines = [
        f"Timer lines: {scan.matched} matched, {unmatched} unmatched "
        f"({unmatched_percent(scan):.1f}%)",
        f"Timer database: {db_status}",
    ]
    if unmatched:
        top = ", ".join(
            f"{key} ({n})" for key, n in scan.unmatched.most_common(5)
        )
        lines.insert(1, f"  most frequent unmatched: {top}")
    return lines


def _task_tables(steps: list[Step], rank: int, top_n: int) -> list[str]:
    """Task-category CPU time and task-count summaries for one rank."""
    out = []
    categories: dict[str, list[float]] = defaultdict(list)
    for step in steps:
        for name, ms in step.task_categories.get(rank, {}).items():
            categories[name].append(ms)
    if categories:
        total = sum(sum(v) for v in categories.values())
        rows = [
            [
                name,
                f"{sum(v):.1f}",
                f"{100 * sum(v) / total if total else 0:.1f}%",
                f"{sum(v) / len(v):.1f}",
                len(v),
            ]
            for name, v in sorted(categories.items(), key=lambda c: -sum(c[1]))
        ]
        out.append(
            create_ascii_table(
                [
                    "Task category",
                    "CPU time (ms)",
                    "% of total",
                    "Avg/step",
                    "Steps",
                ],
                rows,
                f"TASK CATEGORY CPU TIME (rank {rank})",
            )
        )
    counts: dict[str, list[int]] = defaultdict(list)
    for step in steps:
        last = {s.rank: s for s in step.task_counts}.get(rank)
        if last is not None:
            for name, n in last.counts.items():
                counts[name].append(n)
    rows = [
        [name, f"{sum(v) / len(v):.1f}", min(v), max(v), len(v)]
        for name, v in sorted(counts.items(), key=lambda c: -sum(c[1]))
        if sum(v) > 0
    ]
    if rows:
        out.append(
            create_ascii_table(
                ["Task type", "Mean/step", "Min", "Max", "Steps"],
                rows[:top_n],
                f"TASK COUNTS (rank {rank}, last block per step)",
            )
        )
    return out


def analyse_swift_log_timings(
    log_file: str,
    output_path: str | Path | None = None,
    prefix: str | None = None,
    show_plot: bool = False,
    top_n: int = 20,
    hierarchy_functions: list[str] | None = None,
    rank: int | str = 0,
    step_range: tuple[int, int] | None = None,
    swift_src: Path | None = None,
    match_revision: bool = True,
    regenerate_db: bool = False,
) -> None:
    """Analyse the timers of a SWIFT log.

    Writes ``analysis_tables.txt`` and the charts into
    ``<output_path>/log_analysis/`` and prints a short summary.

    Args:
        log_file: The SWIFT log.
        output_path: Directory receiving the ``log_analysis`` folder.
        prefix: Prefix for each output filename.
        show_plot: Whether to show the charts interactively.
        top_n: Rows per table / children per node / bars per chart.
        hierarchy_functions: Timers to show merged over their callers.
        rank: MPI rank to analyse, or "all".
        step_range: Inclusive (first, last) step to analyse.
        swift_src: Source tree to read the timers from.
        match_revision: Read the timers at the log's revision.
        regenerate_db: Rebuild the cached timer database.
    """
    if top_n < 1:
        raise ValueError("--top-n must be at least 1")
    if step_range is not None and step_range[0] > step_range[1]:
        raise ValueError("--steps: START must not exceed END")
    print(f"Analysing SWIFT log: {log_file}")
    revision, dirty = read_log_revision(log_file)
    db, db_status = resolve_timer_db(
        revision, swift_src, match_revision, regenerate_db, dirty
    )
    with reading(log_file):
        scan = scan_log(log_file, db)
    if not scan.matched:
        if not scan.unmatched:
            raise ValueError(
                f"no timer lines in '{log_file}'; run SWIFT with -v 1 (rank "
                "0 timers) or -v 2 (every rank)"
            )
        raise ValueError(
            f"none of the {sum(scan.unmatched.values())} timer lines in "
            f"'{log_file}' match the timers in the SWIFT source "
            f"({db_status}); check --swift-src"
        )

    if rank == "all":
        ranks = sorted(scan.timer_ranks) or [0]
    else:
        assert isinstance(rank, int)
        if scan.timer_ranks and rank not in scan.timer_ranks:
            raise ValueError(
                f"Rank {rank} printed no timers; ranks with timers: "
                f"{sorted(scan.timer_ranks)}"
            )
        ranks = [rank]
        if len(scan.timer_ranks) > 1:
            warn(
                f"{plural(len(scan.timer_ranks), 'rank')} printed timers; "
                f"showing rank {rank} only (use --rank N or --rank all)."
            )

    check_scan(scan, log_file)
    steps = select_steps(scan.steps, step_range)
    if step_range is not None and not steps:
        available = select_steps(scan.steps, None)
        raise ValueError(
            f"no complete steps in --steps {step_range[0]} {step_range[1]}; "
            + (
                f"the log has steps {available[0].number}-"
                f"{available[-1].number}"
                if available
                else "the log has no complete steps"
            )
        )
    total = sum(s.wallclock_ms or 0.0 for s in steps)
    trees = {r: build_step_trees(steps, r, db) for r in ranks}
    hierarchies = {r: aggregate(trees[r], r, total) for r in ranks}
    primary = ranks[0]
    h = hierarchies[primary]

    names = list(hierarchy_functions or DEFAULT_HIERARCHY_FUNCTIONS)
    unknown = [n for n in names if not h.known(n)]
    if hierarchy_functions and unknown:
        warn(f"no timers named {', '.join(unknown)} in the analysed steps.")
    names = [n for n in names if n not in unknown]

    summary = _run_summary(scan, steps)
    summary.append(_coverage_line(trees[primary]))
    high = [t for t in trees[primary] if (t.coverage or 0.0) > COVERAGE_HIGH]
    if high:
        worst = max(high, key=lambda t: t.coverage or 0.0)
        warn(
            f"timers add up to more than {COVERAGE_HIGH:.0f}% of the "
            f"wall-clock time in {plural(len(high), 'step')} (max "
            f"{worst.coverage:.1f}% in step {worst.step.number}): timers may "
            "be double counted (unmatched timers, damaged lines). See the "
            "coverage outliers in analysis_tables.txt."
        )
    if h.between:
        between = sum(s.inclusive for s in h.between.values())
        summary.append(
            f"Timers between steps (not in the step time): "
            f"{between / 1000:.1f} s in "
            f"{plural(sum(s.calls for s in h.between.values()), 'call')} ("
            + ", ".join(
                sorted(h.between, key=lambda k: -h.between[k].inclusive)[:3]
            )
            + ")"
        )
    summary.append(
        top_level_table(
            h,
            top_n,
            f"WHERE THE STEP TIME GOES (rank {primary}, top-level timers)",
        )
    )
    summary.extend(_match_lines(scan, db_status))
    if h.negative:
        summary.append(
            f"Note: {h.negative} timers report less time than the timers "
            "nested in them (by >1%)."
        )

    sections = ["\n".join(summary)]
    for r in ranks:
        hr = hierarchies[r]
        sections.append(
            tree_table(hr, top_n, f"CALL TREE (rank {r}, all paths)")
        )
        sections.append(
            flat_table(hr, top_n, f"TIMERS BY EXCLUSIVE TIME (rank {r})")
        )
        if r != primary:
            sections.append(
                top_level_table(hr, top_n, f"TOP-LEVEL TIMERS (rank {r})")
            )
        sections.extend(function_table(hr, n, top_n) for n in names)
        if hr.between:
            sections.append(between_steps_table(hr, top_n))
        if trees[r]:
            sections.append(coverage_table(trees[r], top_n))
    if len(ranks) > 1:
        sections.append(rank_imbalance_table(hierarchies, top_n))

    init = scan.init_step
    if init is not None and scan.init_end_s:
        init_h = aggregate(
            build_step_trees([init], primary, db),
            primary,
            scan.init_end_s * 1000,
        )
        sections.append(
            top_level_table(
                init_h,
                top_n,
                f"INITIALISATION (rank {primary}, % of time before the "
                "first step row)",
            ).replace("untimed in step", "untimed")
        )
    sections.extend(_task_tables(steps, primary, top_n))
    if scan.unmatched:
        sections.append(
            create_ascii_table(
                ["Unmatched timer (function: label)", "Lines"],
                [[k, n] for k, n in scan.unmatched.most_common(top_n)],
                "TIMER LINES NOT FOUND IN THE SOURCE",
            )
        )

    report = create_output_path(
        output_path, prefix, "analysis_tables.txt", OUT_SUBDIR
    )
    report.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    coverage_csv = create_output_path(
        output_path, prefix, "per_step_coverage.csv", OUT_SUBDIR
    )
    coverage_csv.unlink(missing_ok=True)
    if trees[primary]:
        write_coverage_csv(trees[primary], coverage_csv)

    categories: dict[str, float] = defaultdict(float)
    for step in steps:
        for name, ms in step.task_categories.get(primary, {}).items():
            categories[name] += ms
    plots = make_plots(
        h,
        trees[primary],
        samples_by_key(trees[primary]),
        categories,
        top_n,
        lambda name: create_output_path(output_path, prefix, name, OUT_SUBDIR),
        show_plot,
    )

    print("\n" + "\n".join(summary))
    print(f"\nSaved analysis tables to {report}")
    print(f"Created {len(plots)} plots in {report.parent}")
