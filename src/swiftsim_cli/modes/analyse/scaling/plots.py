"""Strong-scaling plots: time against MPI rank count plus a speedup panel."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

from swiftsim_cli.utilities.plotting import save_figure, should_use_log_scale

# Inches: main panel height, speedup panel height, figure width.
_MAIN_HEIGHT = 8.0
_SPEEDUP_HEIGHT = 3.0
_WIDTH = 10.5


def _wrap(label: str, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def scaling_plot(
    rank_counts: list[int],
    lines: dict[str, list[float]],
    path: Path,
    title: str,
    ylabel: str,
    total: str | None = None,
    show: bool = False,
) -> Path | None:
    """Plot time against rank count, with speedups below.

    Args:
        rank_counts: Rank count of each log (x values).
        lines: Label -> value per log; values <= 0 are skipped and series
            with fewer than two points are left out.
        path: Where to save the figure.
        title: Figure title.
        ylabel: Label of the time axis.
        total: Label of the line drawn as the black total, with a dashed
            perfect (1/N) scaling reference anchored at its first point.
        show: Whether to show the figure.

    Returns:
        The saved path, or None if nothing could be plotted.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    series = {}
    for label, values in lines.items():
        points = [(n, v) for n, v in zip(rank_counts, values) if v > 0]
        if len({n for n, _ in points}) >= 2:
            series[label] = points
    if not series:
        return None
    if (
        ylabel.endswith("(ms)")
        and max(v for points in series.values() for _, v in points) >= 1000
    ):
        series = {
            k: [(n, v / 1000) for n, v in points]
            for k, points in series.items()
        }
        ylabel = ylabel[: -len("(ms)")] + "(s)"

    n_labels = len(series) + (total in series)
    columns = min(3, n_labels)
    legend_height = 0.9 + 0.65 * (math.ceil(n_labels / columns) - 1)
    fig = plt.figure(
        figsize=(_WIDTH, _MAIN_HEIGHT + _SPEEDUP_HEIGHT + legend_height)
    )
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[_MAIN_HEIGHT, _SPEEDUP_HEIGHT, legend_height],
        hspace=0.05,
    )
    ax = fig.add_subplot(grid[0])
    speedup_ax = fig.add_subplot(grid[1], sharex=ax)
    legend_ax = fig.add_subplot(grid[2])
    legend_ax.axis("off")

    plotted: list[float] = []
    for label, points in series.items():
        x = [n for n, _ in points]
        y = [v for _, v in points]
        is_total = label == total
        (line,) = ax.plot(
            x,
            y,
            marker="o",
            label=_wrap(label),
            linewidth=3 if is_total else 2,
            color="black" if is_total else None,
            zorder=4 if is_total else 2,
        )
        color = line.get_color()
        speedup_ax.plot(
            x,
            [y[0] / v for v in y],
            marker="o",
            color=color,
            linewidth=2.5 if is_total else 1.8,
        )
        plotted += y
        if is_total:
            perfect = [y[0] * x[0] / n for n in x]
            ax.plot(
                x,
                perfect,
                linestyle="--",
                linewidth=2,
                color="0.35",
                label="Perfect scaling (1/N)",
                zorder=3,
            )
            plotted += perfect

    ticks = sorted(set(rank_counts))
    speedup_ax.plot(
        ticks,
        [n / ticks[0] for n in ticks],
        linestyle="--",
        color="black",
        alpha=0.5,
        linewidth=1.2,
    )
    for axis in (ax, speedup_ax):
        axis.set_xscale("log", base=2)
        axis.grid(True, alpha=0.3, linestyle="--")
    speedup_ax.xaxis.set_major_locator(FixedLocator(ticks))
    speedup_ax.xaxis.set_minor_locator(NullLocator())
    speedup_ax.set_xticklabels([str(n) for n in ticks])
    plt.setp(ax.get_xticklabels(), visible=False)
    speedup_ax.set_xlabel("Number of MPI ranks")
    speedup_ax.set_ylabel("Speedup (dashed: ideal from the smallest run)")
    speedup_ax.axhline(1.0, color="0.5", linewidth=1.0, alpha=0.6)
    if should_use_log_scale(plotted):
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=columns,
        fontsize=9,
        frameon=False,
    )
    return save_figure(fig, path, show=show)
