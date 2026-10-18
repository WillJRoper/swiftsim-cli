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
    add_steps_argument,
)
from swiftsim_cli.timing.log_scan import (
    LogScan,
    Step,
    read_log_revision,
    scan_log,
)
from swiftsim_cli.timing.timer_db import resolve_timer_db
from swiftsim_cli.timing.timing_hierarchy import (
    StepTree,
    aggregate,
    build_step_trees,
    coverage_table,
    flat_table,
    function_table,
    rank_imbalance_table,
    samples_by_key,
    select_steps,
    top_level_table,
    tree_table,
)
from swiftsim_cli.timing.timing_plots import make_plots
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.tables import create_ascii_table

# Subdirectory of --out that the outputs are written to
OUT_SUBDIR = "log_analysis"

# Timers whose merged subtrees are shown when --hierarchy-functions is unset
DEFAULT_HIERARCHY_FUNCTIONS = (
    "engine_prepare",
    "engine_rebuild",
    "space_rebuild",
    "engine_maketasks",
)


def _rank_arg(value: str) -> int | str:
    """Parse ``--rank``: a rank number or "all"."""
    if value == "all":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a rank number or 'all', got {value!r}"
        ) from None


def add_log_arguments(subparsers) -> None:
    """Add the ``log`` subcommand and its arguments."""
    parser = subparsers.add_parser(
        "log",
        help="Analyse the timers in a SWIFT log file (run SWIFT with -v 1 "
        "for rank 0 timers, -v 2 for every rank).",
    )
    parser.add_argument("log_file", type=Path, help="SWIFT log file.")
    add_output_arguments(parser, OUT_SUBDIR)
    add_steps_argument(parser)
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Rows per table, children per node in call trees and bars per "
        "chart (default: 20).",
    )
    parser.add_argument(
        "--hierarchy-functions",
        nargs="+",
        default=None,
        help="Timers (function names or timer keys) whose subtrees are "
        "shown merged over all their callers, with a breakdown by caller "
        f"(default: {', '.join(DEFAULT_HIERARCHY_FUNCTIONS)}).",
    )
    parser.add_argument(
        "--rank",
        type=_rank_arg,
        default=0,
        help="MPI rank to analyse, or 'all' for one analysis per rank plus "
        "an imbalance table (default: 0). Ranks are never summed.",
    )
    parser.add_argument(
        "--swift-src",
        type=Path,
        default=None,
        help="SWIFT source tree to read the timers from (default: the "
        "active profile's checkout, at the log's revision).",
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
    if scan.init_end_s is not None:
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


def _match_lines(scan: LogScan, db_status: str) -> list[str]:
    unmatched = sum(scan.unmatched.values())
    total = scan.matched + unmatched
    lines = [
        f"Timer lines: {scan.matched} matched, {unmatched} unmatched "
        f"({100 * unmatched / total if total else 0:.1f}%)",
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
        [name, sum(v), max(v), f"{sum(v) / len(v):.1f}"]
        for name, v in sorted(counts.items(), key=lambda c: -sum(c[1]))
        if sum(v) > 0
    ]
    if rows:
        out.append(
            create_ascii_table(
                ["Task type", "Total", "Max", "Avg/step"],
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
    print(f"Analysing SWIFT log: {log_file}")
    db, db_status = resolve_timer_db(
        read_log_revision(log_file), swift_src, match_revision, regenerate_db
    )
    scan = scan_log(log_file, db)

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
            print(
                f"Warning: {len(scan.timer_ranks)} ranks printed timers; "
                f"showing rank {rank} only (use --rank N or --rank all)."
            )

    steps = select_steps(scan.steps, step_range)
    total = sum(s.wallclock_ms or 0.0 for s in steps)
    trees = {r: build_step_trees(steps, r, db) for r in ranks}
    hierarchies = {r: aggregate(trees[r], r, total) for r in ranks}
    primary = ranks[0]
    h = hierarchies[primary]

    names = list(hierarchy_functions or DEFAULT_HIERARCHY_FUNCTIONS)
    unknown = [n for n in names if not h.known(n)]
    if hierarchy_functions and unknown:
        print(
            "Warning: no timers named "
            + ", ".join(unknown)
            + " in the analysed steps."
        )
    names = [n for n in names if n not in unknown]

    summary = _run_summary(scan, steps)
    summary.append(_coverage_line(trees[primary]))
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
        if trees[r]:
            sections.append(coverage_table(trees[r]))
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
