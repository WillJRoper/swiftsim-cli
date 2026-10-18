"""``analyse scaling runtime``: whole-run strong scaling."""

from __future__ import annotations

from pathlib import Path

from swiftsim_cli.modes.analyse.scaling.data import (
    clear_outputs,
    fit_slope,
    format_slope,
    format_time,
    load_log_runtime,
    require_rank_counts,
)
from swiftsim_cli.modes.analyse.scaling.plots import scaling_plot
from swiftsim_cli.utilities.io import create_output_path
from swiftsim_cli.utilities.messages import plural
from swiftsim_cli.utilities.tables import create_ascii_table

OUT_SUBDIR = "scaling_runtime_analysis"


def analyse_swift_runtime_scaling(
    log_files: list[str],
    output_path: str | Path | None = None,
    prefix: str | None = None,
    show_plot: bool = False,
    step_range: tuple[int, int] | None = None,
) -> None:
    """Analyse whole-run scaling from the step-table wall-clock times.

    The runtime of a log is the summed wall-clock time of its steps
    (initialisation and a restart's 0.000 row excluded), the same step
    time ``analyse log`` and ``scaling timers`` use.
    """
    if len(log_files) < 2:
        raise ValueError("Runtime scaling requires at least two log files")
    if step_range is not None and step_range[0] > step_range[1]:
        raise ValueError("Step range must be ordered as START END")

    print(
        "Analysing runtime scaling across "
        f"{plural(len(log_files), 'log file')}"
    )
    logs = []
    for path in log_files:
        print(f"\nProcessing runtime scaling input: {path}")
        logs.append(load_log_runtime(path, step_range))
    logs.sort(key=lambda log: (log.n_ranks, log.label))
    ranks = [log.n_ranks for log in logs]
    require_rank_counts(ranks)
    runtimes = [log.runtime for log in logs]

    summary = create_output_path(
        output_path, prefix, "runtime_scaling_summary.txt", OUT_SUBDIR
    )
    clear_outputs(summary.parent, prefix, "runtime_scaling.png")
    plot = scaling_plot(
        ranks,
        {"Total runtime": runtimes},
        create_output_path(
            output_path, prefix, "runtime_scaling.png", OUT_SUBDIR
        ),
        "Runtime scaling (summed step wall-clock time)",
        "Runtime (ms)",
        total="Total runtime",
        show=show_plot,
    )
    slope = fit_slope(ranks, runtimes)
    table = create_ascii_table(
        ["Ranks", "Runtime", "Speedup", "Steps", "Log"],
        [
            [
                log.n_ranks,
                format_time(log.runtime),
                f"{runtimes[0] / log.runtime:.2f}x",
                log.n_steps,
                log.label,
            ]
            for log in logs
        ],
        "RUNTIME SCALING SUMMARY",
    )
    lines = [
        "Runtime scaling analysis:",
        f"- Runtime goes from {format_time(runtimes[0])} at "
        f"{plural(ranks[0], 'rank')} to {format_time(runtimes[-1])} at "
        f"{plural(ranks[-1], 'rank')}.",
        f"- Overall strong-scaling slope: {format_slope(slope)} (-1 is "
        "perfect; the plot's dashed line is perfect 1/N scaling from the "
        "smallest run).",
    ]
    if step_range is not None:
        lines.append(
            f"- Only steps {step_range[0]}-{step_range[1]} were included."
        )
    text = table + "\n\n" + "\n".join(lines)
    print("\n" + text)
    summary.write_text(text + "\n", encoding="utf-8")
    print("\nCreated outputs:")
    if plot is not None:
        print(f"  - {plot}")
    print(f"  - {summary}")
