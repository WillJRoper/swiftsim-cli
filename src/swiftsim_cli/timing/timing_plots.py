"""Charts of the aggregated timer statistics of ``analyse log``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from swiftsim_cli.timing.timing_hierarchy import UNTIMED, Hierarchy, StepTree
from swiftsim_cli.utilities.plotting import save_figure


def _ms_label(ms: float) -> str:
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def _barh(
    names: list[str],
    values: list[float],
    labels: list[str],
    xlabel: str,
    title: str,
    path: Path,
    show: bool,
    cmap: str = "Blues_r",
) -> Path | None:
    """Horizontal bar chart, largest at the top."""
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.45 * len(names) + 1.5)))
    colors = plt.get_cmap(cmap)(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")
    return save_figure(fig, path, show=show)


def exclusive_time_chart(
    h: Hierarchy, top_n: int, path: Path, show: bool
) -> Path | None:
    """Timers with the most exclusive (self) time."""
    rows = [r for r in h.flat()[:top_n] if r[1].exclusive > 0]
    return _barh(
        [k for k, _, _ in rows],
        [s.exclusive for _, s, _ in rows],
        [
            f"{_ms_label(s.exclusive)} ({h.percent(s.exclusive):.1f}%)"
            for _, s, _ in rows
        ],
        "Exclusive time (ms)",
        "Timers by Exclusive Time (share of step time)",
        path,
        show,
    )


def step_breakdown_pie(
    h: Hierarchy, path: Path, show: bool, top: int = 6
) -> Path | None:
    """Top-level timers and the untimed remainder as shares of step time."""
    kids = h.children(())
    names = [p[-1] for p in kids[:top]]
    values = [h.paths[p].inclusive for p in kids[:top]]
    other = sum(h.paths[p].inclusive for p in kids[top:])
    if other > 0:
        names.append("other timers")
        values.append(other)
    if h.untimed > 0:
        names.append(UNTIMED)
        values.append(h.untimed)
    if not values or sum(values) <= 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(
        values,
        labels=[n[:30] for n in names],
        autopct="%1.1f%%",
        startangle=90,
        colors=list(plt.get_cmap("Set2")(np.linspace(0, 1, len(names)))),
        pctdistance=0.8,
    )
    ax.set_title(
        "Step Time Breakdown (top-level timers)",
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(fig, path, show=show)


def mean_time_chart(
    h: Hierarchy, top_n: int, path: Path, show: bool
) -> Path | None:
    """Timers with the largest mean (inclusive) time per call."""
    rows = sorted(h.flat(), key=lambda r: -r[1].mean)[:top_n]
    return _barh(
        [k for k, _, _ in rows],
        [s.mean for _, s, _ in rows],
        [f"{s.mean:.2f}ms ({s.calls} calls)" for _, s, _ in rows],
        "Mean inclusive time per call (ms)",
        "Mean Time per Call",
        path,
        show,
        cmap="RdYlBu_r",
    )


def timer_total_histogram(h: Hierarchy, path: Path, show: bool) -> Path | None:
    """Distribution of the inclusive totals of all timers."""
    totals = [outer for _, _, outer in h.flat() if outer > 0]
    if len(totals) < 2:
        return None
    fig, ax = plt.subplots(figsize=(12, 7))
    bins = list(np.logspace(np.log10(min(totals)), np.log10(max(totals)), 30))
    ax.hist(totals, bins=bins, color="steelblue", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Inclusive total per timer (ms)")
    ax.set_ylabel("Number of timers")
    ax.set_title(
        "Distribution of Timer Totals", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    return save_figure(fig, path, show=show)


def task_category_chart(
    categories: dict[str, float], path: Path, show: bool
) -> Path | None:
    """CPU time per task category."""
    total = sum(categories.values())
    items = sorted(
        ((c, t) for c, t in categories.items() if t > 0.001 * total),
        key=lambda item: -item[1],
    )
    return _barh(
        [c for c, _ in items],
        [t for _, t in items],
        [_ms_label(t) for _, t in items],
        "CPU time (ms)",
        "Task Category CPU Time",
        path,
        show,
        cmap="Dark2",
    )


def calls_scatter(
    h: Hierarchy, path: Path, show: bool, top: int = 5
) -> Path | None:
    """Calls vs mean time per call, sized by total time."""
    rows = [r for r in h.flat() if r[1].calls and r[2] > 1]
    if not rows:
        return None
    calls = np.array([s.calls for _, s, _ in rows])
    means = np.array([s.mean for _, s, _ in rows])
    totals = np.array([outer for _, _, outer in rows])
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        calls,
        means,
        s=totals / totals.max() * 300 + 20,
        c=totals,
        cmap="viridis",
        alpha=0.6,
    )
    for i in np.argsort(-totals)[:top]:
        ax.annotate(
            rows[i][0][:30],
            (calls[i], means[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of calls")
    ax.set_ylabel("Mean inclusive time per call (ms)")
    ax.set_title(
        "Timer Call Patterns (size = inclusive total)",
        fontsize=14,
        fontweight="bold",
    )
    fig.colorbar(scatter, ax=ax, label="Inclusive total (ms)")
    return save_figure(fig, path, show=show)


def step_time_series(
    trees: list[StepTree], h: Hierarchy, path: Path, show: bool, top: int = 8
) -> Path | None:
    """Per-step time of the largest top-level timers and the wall-clock."""
    if len(trees) < 2:
        return None
    keys = [p[-1] for p in h.children(())[:top]]
    steps = [t.step.number for t in trees]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        steps,
        [
            np.nan if t.step.wallclock_ms is None else t.step.wallclock_ms
            for t in trees
        ],
        "k-",
        lw=1.5,
        label="step wall-clock",
    )
    for key in keys:
        series = [
            sum(r.inclusive for r in t.roots if r.record.key == key)
            for t in trees
        ]
        ax.plot(steps, series, "o", ms=2, alpha=0.8, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Top-level Timers per Step", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    return save_figure(fig, path, show=show)


def variability_chart(
    samples: dict[str, list[float]], top_n: int, path: Path, show: bool
) -> Path | None:
    """Coefficient of variation of the per-call time of each timer."""
    rows = sorted(
        (
            (key, 100 * float(np.std(v) / np.mean(v)))
            for key, v in samples.items()
            if len(v) > 2 and np.mean(v) > 0
        ),
        key=lambda r: -r[1],
    )[:top_n]
    return _barh(
        [k for k, _ in rows],
        [cv for _, cv in rows],
        [f"{cv:.1f}%" for _, cv in rows],
        "Coefficient of variation of per-call time (%)",
        "Timer Variability",
        path,
        show,
        cmap="plasma",
    )


def make_plots(
    h: Hierarchy,
    trees: list[StepTree],
    samples: dict[str, list[float]],
    categories: dict[str, float],
    top_n: int,
    out: Callable[[str], Path],
    show: bool,
) -> list[Path]:
    """Draw every chart; returns the paths written.

    Args:
        h: Aggregated statistics of the analysed steps.
        trees: The per-step trees they were built from.
        samples: Per-call inclusive times per timer key.
        categories: Task-category CPU time (ms).
        top_n: Maximum number of timers per chart.
        out: Maps a file name to its output path.
        show: Whether to show each chart interactively.
    """
    plots = [
        exclusive_time_chart(h, top_n, out("01_exclusive_time.png"), show),
        step_breakdown_pie(h, out("02_step_breakdown.png"), show),
        mean_time_chart(h, top_n, out("03_mean_time_per_call.png"), show),
        timer_total_histogram(h, out("04_timer_distribution.png"), show),
        task_category_chart(categories, out("05_task_categories.png"), show),
        calls_scatter(h, out("06_call_patterns.png"), show),
        step_time_series(trees, h, out("07_time_series.png"), show),
        variability_chart(
            samples, top_n, out("08_timing_variability.png"), show
        ),
    ]
    return [p for p in plots if p is not None]
