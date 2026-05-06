"""Log timing analysis module for SWIFT simulations.

This module provides comprehensive timing analysis capabilities for SWIFT
simulation logs, featuring hierarchical timing breakdowns, performance
visualization, and detailed statistical analysis.

Key Features:
* Automatically generates timer nesting database from SWIFT source code
* Parses log files to extract timer instances and step information
* Creates hierarchical timing tables showing function call trees
* Implements function timer = max(explicit_timer, sum_of_nested) rule
* Shows "UNACCOUNTED TIME" for missing instrumentation
* Produces standard plots: top timers, distributions, task counts, etc.
* Estimates per-step untimed overhead when step totals are available

Author: SWIFT Development Team
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
from tqdm.auto import tqdm

from swiftsim_cli.profile import load_swift_profile
from swiftsim_cli.src_parser import (
    compile_site_patterns,
    generate_timer_nesting_database,
    load_timer_db,
    scan_log_instances_by_step,
)
from swiftsim_cli.utilities import create_ascii_table, create_output_path

# ============================================================================
# CLI ARGUMENT SETUP
# ============================================================================


def add_log_arguments(subparsers) -> None:
    """Add CLI arguments for log timing analysis.

    Args:
        subparsers: The argparse subparsers object to add the log parser to.
    """
    log_parser = subparsers.add_parser(
        "log",
        help="Analyse timing information from SWIFT log files."
        " To get the most from this mode SWIFT should be run with "
        "-v 1 for verbose output.",
    )

    log_parser.add_argument(
        "log_file",
        help="SWIFT log file to analyse.",
        type=Path,
    )

    log_parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        help="Where to save analysis (default: current directory).",
        default=None,
    )

    log_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="A prefix to add to the analysis files (default: '').",
        default=None,
    )

    log_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
        default=False,
    )

    log_parser.add_argument(
        "--top-n",
        type=int,
        help="Number of top functions to show in detailed plots "
        "(default: 20).",
        default=20,
    )

    log_parser.add_argument(
        "--hierarchy-functions",
        nargs="*",
        help="Functions to show hierarchical timing for. If not specified, "
        "shows default important functions: engine_prepare, "
        "engine_rebuild, space_rebuild, space_split, and "
        "engine_maketasks. Use 'all' to show all functions with "
        "hierarchy data, or specify individual function names.",
        default=None,
    )


def run_swift_log_timing(args: argparse.Namespace) -> None:
    """Execute the SWIFT log timing analysis.

    Args:
        args: Parsed command line arguments containing analysis parameters.
    """
    analyse_swift_log_timings(
        log_file=str(args.log_file),
        output_path=args.output_path,
        prefix=args.prefix,
        show_plot=args.show,
        top_n=args.top_n,
        hierarchy_functions=args.hierarchy_functions,
    )


# ============================================================================
# TIMER CLASSIFICATION AND UTILITIES
# ============================================================================


def classify_timers_by_max_time(
    instances_by_step: Dict, timer_db: Dict, nesting_db: Dict
) -> set:
    """Classify timers using max time per function.

    For each function, the timer with the most total time is treated as the
    function timer. All others are operation timers. No synthetic timers.

    Args:
        instances_by_step: Dictionary mapping step numbers to timer instances
        timer_db: Dictionary of timer definitions by timer ID
        nesting_db: Dictionary of nesting relationships by function name

    Returns:
        Set of timer IDs that are function timers
    """
    function_timer_ids = set()

    # Calculate total time per timer
    timer_totals: dict[int, float] = defaultdict(float)
    for inst_list in instances_by_step.values():
        for inst in inst_list:
            timer_totals[inst.timer_id] += inst.time_ms

    # Group timers by function
    timers_by_function = defaultdict(list)
    for tid, total_time in timer_totals.items():
        if tid in timer_db:
            func_name = timer_db[tid].function
            timers_by_function[func_name].append((tid, total_time))

    # For each function, select the timer with max time as function timer
    for func_name, timer_list in timers_by_function.items():
        if timer_list:
            # Sort by total time and pick the highest
            timer_list.sort(key=lambda x: x[1], reverse=True)
            max_timer_id = timer_list[0][0]
            function_timer_ids.add(max_timer_id)

    # Update timer type classification in timer_db for analysis
    for tid, timer_def in timer_db.items():
        if tid in function_timer_ids:
            timer_def.timer_type = "function"
        else:
            timer_def.timer_type = "operation"

    return function_timer_ids


def load_timer_nesting(
    auto_generate: bool = True, force_regenerate: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Load timer nesting relationships.

    This function loads or generates the timer nesting database required for
    hierarchical timing analysis. It implements a fail-fast approach: if the
    database cannot be generated due to missing configuration, it raises an
    error rather than silently degrading functionality.

    Args:
        auto_generate: If True, automatically generate nesting DB if it
            doesn't exist
        force_regenerate: If True, regenerate even if file exists

    Returns:
        Dictionary mapping function names to their nesting information:
        {
            "function_name": {
                "function_timer": "pattern",
                "file": "source.c",
                "nested_operations": ["op1", "op2"],
                "nested_functions": ["func1", "func2"]
            }
        }

    Raises:
        ValueError: If swiftsim_dir is not set in profile and generation
            is required. This is intentionally not caught to ensure users
            are aware of configuration issues.
    """
    nesting_file = Path.home() / ".swiftsim-utils" / "timer_nesting.yaml"

    # Check if we need to generate the database
    should_generate = force_regenerate or (
        auto_generate and not nesting_file.exists()
    )

    if should_generate:
        print("Auto-generating timer nesting database from source code...")

        # Get SWIFT source directory from profile
        profile = load_swift_profile()
        swift_src = profile.swiftsim_dir

        # Fail fast if configuration is missing. Hierarchical timer analysis
        # is a core feature that requires source code - silently degrading
        # would hide configuration issues from users.
        if swift_src is None:
            raise ValueError(
                "Cannot generate timer nesting database: 'swiftsim_dir' "
                "not set in profile. Run 'swift-cli profile --init' to "
                "configure your SWIFT installation path."
            )

        # Generate the nesting database
        nesting_data = generate_timer_nesting_database(str(swift_src))

        # Save to file
        yaml_writer = YAML()
        yaml_writer.default_flow_style = False
        nesting_file.parent.mkdir(exist_ok=True)
        with open(nesting_file, "w") as f:
            yaml_writer.dump(nesting_data, f)

        print(
            "Generated nesting database with "
            f"{len(nesting_data.get('nesting', {}))} functions"
        )
        return nesting_data.get("nesting", {})

    # Load existing file
    if not nesting_file.exists():
        return {}

    yaml_safe = YAML(typ="safe")
    with open(nesting_file, "r", encoding="utf-8") as f:
        data = yaml_safe.load(f) or {}

    return data.get("nesting", {})


def display_name(tid: str, timer_db: Dict) -> str:
    """Generate a readable display name for a timer.

    Args:
        tid: Timer ID
        timer_db: Dictionary of timer definitions

    Returns:
        Formatted display name for the timer
    """
    if tid.startswith("SYNTHETIC:"):
        func_name = tid[10:]  # Remove "SYNTHETIC:" prefix
        return f"{func_name} [SYNTHETIC:sum_of_operations]"
    td = timer_db[tid]
    return f"{td.function} [{tid}]"


def build_stats(values: List[float]) -> Dict[str, float | int]:
    """Build statistical summary for a list of timing values.

    Args:
        values: List of timing values in milliseconds

    Returns:
        Dictionary containing statistical measures
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return dict(
            total_time=0.0,
            mean_time=0.0,
            median_time=0.0,
            std_time=0.0,
            max_time=0.0,
            min_time=0.0,
            call_count=0,
        )
    return dict(
        total_time=float(np.sum(arr)),
        mean_time=float(np.mean(arr)),
        median_time=float(np.median(arr)),
        std_time=float(np.std(arr)),
        max_time=float(np.max(arr)),
        min_time=float(np.min(arr)),
        call_count=int(arr.size),
    )


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def create_function_timing_chart(
    sorted_functions, timer_db, output_path, prefix, show_plot, out_dir
):
    """Create a clean function timing bar chart.

    Args:
        sorted_functions: List of (timer_id, stats) tuples sorted by time
        timer_db: Dictionary of timer definitions
        output_path: Base output path for saving
        prefix: Filename prefix
        show_plot: Whether to display the plot
        out_dir: Output directory name

    Returns:
        Path to saved plot
    """
    if not sorted_functions:
        return None

    print("Creating function timing chart...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Take top 15 functions for better readability
    top_funcs = sorted_functions[:15]
    names = []
    times = []

    for tid, stats in top_funcs:
        # Clean up function names for better display
        if tid in timer_db:
            func_name = timer_db[tid].function
        elif tid.startswith("SYNTHETIC:"):
            func_name = tid.replace("SYNTHETIC:", "")
        else:
            func_name = "Unknown"
        # Remove SYNTHETIC prefix and cleanup
        clean_name = (
            func_name.replace("SYNTHETIC:", "").replace("_", " ").title()
        )
        names.append(clean_name)
        times.append(stats["total_time"])

    # Use a professional color gradient
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.9, len(names)))

    # Create horizontal bar chart
    bars = ax.barh(
        range(len(names)),
        times,
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    # Clean formatting
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Execution Time (ms)", fontsize=12)
    ax.set_title(
        "Top Function Execution Times",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, times)):
        # Format time nicely
        if time >= 1000:
            label = f"{time / 1000:.1f}s"
        else:
            label = f"{time:.0f}ms"

        ax.text(
            bar.get_width() * 1.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")

    plt.tight_layout()
    p = create_output_path(
        output_path, prefix, "01_function_timing.png", out_dir
    )
    plt.savefig(p, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    return p


def create_function_distribution_chart(
    sorted_functions, timer_db, output_path, prefix, show_plot, out_dir
):
    """Create an improved pie chart showing function distribution.

    Args:
        sorted_functions: List of (timer_id, stats) tuples sorted by time
        timer_db: Dictionary of timer definitions
        output_path: Base output path for saving
        prefix: Filename prefix
        show_plot: Whether to display the plot
        out_dir: Output directory name

    Returns:
        Path to saved plot
    """
    if not sorted_functions:
        return None

    print("Creating function distribution chart...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Take top 6 + others for better readability
    top_items = sorted_functions[:6]
    other_time = sum(st["total_time"] for _, st in sorted_functions[6:])

    names = []
    times = []

    for tid, stats in top_items:
        # Handle both regular and synthetic timer IDs
        if tid in timer_db:
            func_name = timer_db[tid].function
        elif tid.startswith("SYNTHETIC:"):
            func_name = tid.replace("SYNTHETIC:", "")
        else:
            func_name = "Unknown"
        clean_name = (
            func_name.replace("SYNTHETIC:", "").replace("_", " ").title()
        )
        names.append(clean_name[:20])  # Limit length
        times.append(stats["total_time"])

    if other_time > 0:
        names.append("Other Functions")
        times.append(other_time)

    # Use professional colors
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # Create pie chart with clear separation
    wedges, texts, autotexts = ax.pie(
        times,
        labels=names,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 11, "fontweight": "bold"},
        pctdistance=0.85,
        labeldistance=1.1,
    )

    # Style the percentage labels
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)

    ax.set_title(
        "Function Execution Time Distribution",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    p = create_output_path(
        output_path, prefix, "02_function_distribution.png", out_dir
    )
    plt.savefig(p, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    return p


def create_function_efficiency_chart(
    sorted_functions, timer_db, output_path, prefix, show_plot, out_dir
):
    """Create function efficiency chart showing time per call.

    Args:
        sorted_functions: List of (timer_id, stats) tuples sorted by time
        timer_db: Dictionary of timer definitions
        output_path: Base output path for saving
        prefix: Filename prefix
        show_plot: Whether to display the plot
        out_dir: Output directory name

    Returns:
        Path to saved plot
    """
    if not sorted_functions:
        return None

    print("Creating function efficiency chart...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate efficiency (time per call)
    efficiency_data = []
    for tid, stats in sorted_functions[:20]:
        if stats["call_count"] > 0:
            # Handle both regular and synthetic timer IDs
            if tid in timer_db:
                func_name = timer_db[tid].function
            elif tid.startswith("SYNTHETIC:"):
                func_name = tid.replace("SYNTHETIC:", "")
            else:
                func_name = "Unknown"
            clean_name = (
                func_name.replace("SYNTHETIC:", "").replace("_", " ").title()
            )
            efficiency = stats["total_time"] / stats["call_count"]
            efficiency_data.append(
                (clean_name, efficiency, stats["call_count"])
            )

    # Sort by efficiency
    efficiency_data.sort(key=lambda x: x[1], reverse=True)
    efficiency_data = efficiency_data[:15]  # Top 15

    names = [item[0] for item in efficiency_data]
    efficiencies = [item[1] for item in efficiency_data]
    call_counts = [item[2] for item in efficiency_data]

    # Color by call count
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(names)))

    bars = ax.barh(
        range(len(names)),
        efficiencies,
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Average Time per Call (ms)", fontsize=12)
    ax.set_title(
        "Function Efficiency Analysis",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add efficiency labels
    for i, (bar, eff, calls) in enumerate(
        zip(bars, efficiencies, call_counts)
    ):
        if eff >= 1:
            label = f"{eff:.1f}ms ({calls} calls)"
        else:
            label = f"{eff:.2f}ms ({calls} calls)"
        ax.text(
            bar.get_width() * 1.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")

    plt.tight_layout()
    p = create_output_path(
        output_path, prefix, "03_function_efficiency.png", out_dir
    )
    plt.savefig(p, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    return p


def create_timer_distribution_histogram(
    all_stats, output_path, prefix, show_plot, out_dir
):
    """Create timer distribution histogram.

    Args:
        all_stats: Dictionary of all timer statistics
        output_path: Base output path for saving
        prefix: Filename prefix
        show_plot: Whether to display the plot
        out_dir: Output directory name

    Returns:
        Path to saved plot
    """
    print("Creating timer distribution analysis...")

    all_times = []
    for stats in all_stats.values():
        if stats["total_time"] > 0:
            all_times.append(stats["total_time"])

    if not all_times:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create histogram with better binning
    bins = np.logspace(np.log10(min(all_times)), np.log10(max(all_times)), 30)
    counts, bins, patches = ax.hist(
        all_times,
        bins=bins,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
    )

    # Color gradient for histogram
    colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    ax.set_xlabel("Timer Duration (ms)", fontsize=12)
    ax.set_ylabel("Number of Timers", fontsize=12)
    ax.set_title(
        "Distribution of Timer Execution Times",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xscale("log")

    # Add statistics text
    mean_time = np.mean(all_times)
    median_time = np.median(all_times)
    stats_text = (
        f"Mean: {mean_time:.1f}ms\n"
        f"Median: {median_time:.1f}ms\n"
        f"Total Timers: {len(all_times)}"
    )
    ax.text(
        0.7,
        0.8,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = create_output_path(
        output_path, prefix, "04_timer_distribution.png", out_dir
    )
    plt.savefig(p, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    return p


def create_task_category_overview(
    task_category_times, output_path, prefix, show_plot, out_dir
):
    """Create task category overview chart.

    Args:
        task_category_times: Dictionary of task category timing data
        output_path: Base output path for saving
        prefix: Filename prefix
        show_plot: Whether to display the plot
        out_dir: Output directory name

    Returns:
        Path to saved plot
    """
    if not task_category_times:
        return None

    print("Creating task category overview...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate totals and get top categories
    cat_totals = {
        cat: sum(times) for cat, times in task_category_times.items() if times
    }

    if not cat_totals:
        return None

    # Sort and take meaningful categories
    items = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
    # Filter out very small categories
    total_time = sum(cat_totals.values())
    items = [(cat, time) for cat, time in items if time > total_time * 0.001][
        :8
    ]

    categories = [item[0].replace("_", " ").title() for item in items]
    times = [item[1] for item in items]

    # Professional color scheme
    colors = plt.cm.Dark2(np.linspace(0, 1, len(categories)))

    bars = ax.bar(
        range(len(categories)),
        times,
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=1,
    )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Total Time (ms)", fontsize=12)
    ax.set_title(
        "Task Category Performance Overview",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        if time >= 1000:
            label = f"{time / 1000:.1f}s"
        else:
            label = f"{time:.0f}ms"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + height * 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    p = create_output_path(
        output_path, prefix, "05_task_categories.png", out_dir
    )
    plt.savefig(p, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    return p


def create_performance_summary_scatter(
    function_stats, timer_db, output_path, prefix, show_plot, out_dir
):
    """Create performance summary scatter plot.

    Args:
        function_stats: Dictionary of function timer statistics
        timer_db: Dictionary of timer definitions
        output_path: Base output path for saving
        prefix: Filename prefix
        show_plot: Whether to display the plot
        out_dir: Output directory name

    Returns:
        Path to saved plot
    """
    if not function_stats:
        return None

    print("Creating performance summary...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for scatter plot
    scatter_data = []
    for tid, stats in function_stats.items():
        if (
            stats["call_count"] > 0 and stats["total_time"] > 1
        ):  # Filter very small timers
            # Handle both regular and synthetic timer IDs
            if tid in timer_db:
                func_name = timer_db[tid].function
            elif tid.startswith("SYNTHETIC:"):
                func_name = tid.replace("SYNTHETIC:", "")
            else:
                func_name = "Unknown"
            avg_duration = stats["total_time"] / stats["call_count"]
            scatter_data.append(
                {
                    "name": func_name.replace("SYNTHETIC:", "")
                    .replace("_", " ")
                    .title(),
                    "calls": stats["call_count"],
                    "avg_duration": avg_duration,
                    "total_time": stats["total_time"],
                }
            )

    if not scatter_data:
        return None

    calls = [item["calls"] for item in scatter_data]
    avg_durations = [item["avg_duration"] for item in scatter_data]
    total_times = [item["total_time"] for item in scatter_data]

    # Create scatter plot with size proportional to total time
    sizes = np.array(total_times)
    sizes = (sizes / sizes.max()) * 300 + 20  # Scale for visibility

    scatter = ax.scatter(
        calls,
        avg_durations,
        s=sizes,
        alpha=0.6,
        c=total_times,
        cmap="viridis",
        edgecolors="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Number of Calls", fontsize=12)
    ax.set_ylabel("Average Duration per Call (ms)", fontsize=12)
    ax.set_title(
        "Function Call Patterns (Size = Total Time)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(
        "Total Execution Time (ms)",
        rotation=270,
        labelpad=20,
        fontsize=11,
    )

    # Add annotations for outliers
    for item in scatter_data[:5]:  # Top 5 by total time
        ax.annotate(
            item["name"][:15],
            (item["calls"], item["avg_duration"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = create_output_path(
        output_path, prefix, "06_performance_summary.png", out_dir
    )
    plt.savefig(p, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    return p


def create_time_series_plot(
    instances_by_step,
    function_stats,
    timer_db,
    output_path,
    prefix,
    show_plot,
    out_dir,
):
    """Create time series plot showing function evolution over steps.

    Args:
        instances_by_step: Dictionary mapping steps to timer instances
        function_stats: Dictionary of function timer statistics
        timer_db: Dictionary of timer definitions
        output_path: Base output path for saving
        prefix: Filename prefix
        show_plot: Whether to display the plot
        out_dir: Output directory name

    Returns:
        Path to saved plot
    """
    print("Creating time series analysis...")

    # Collect time series data for top functions
    steps_with_data = sorted(
        [k for k in instances_by_step.keys() if k is not None]
    )
    if len(steps_with_data) <= 1 or not function_stats:
        return None

    # Get top 8 functions by total time for time series
    top_functions = sorted(
        function_stats.items(),
        key=lambda x: x[1]["total_time"],
        reverse=True,
    )[:8]

    panel_count = len(top_functions)
    if panel_count == 0:
        return None

    fig_height = max(3.0 * panel_count + 1.5, 6.0)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(14, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes_list = [axis for axis in axes.flat]

    colors = plt.cm.Set1(np.linspace(0, 1, len(top_functions)))

    for i, ((tid, stats), ax) in enumerate(
        zip(top_functions, axes_list, strict=False)
    ):
        # Collect execution times per step for this function
        step_times = []
        step_numbers = []

        for step in steps_with_data:
            # Sum all instances of this timer in this step
            step_total = sum(
                inst.time_ms
                for inst in instances_by_step[step]
                if inst.timer_id == tid
            )
            step_times.append(step_total)
            step_numbers.append(step)

        # Get function name for display
        if tid in timer_db:
            func_name = timer_db[tid].function
            clean_name = func_name.replace("_", " ").title()
        else:
            clean_name = "Unknown Function"

        # Plot one function per panel so the step evolution stays readable.
        ax.plot(
            step_numbers,
            step_times,
            "o-",
            color=colors[i],
            linewidth=2,
            markersize=4,
            alpha=0.8,
        )
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title(clean_name, fontsize=12, loc="left", pad=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_yscale(
            "log"
        )  # Log scale for better visibility of different magnitudes

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Function Execution Times Over Simulation Steps",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    axes_list[-1].set_xlabel("Simulation Step", fontsize=12)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    p = create_output_path(output_path, prefix, "07_time_series.png", out_dir)
    plt.savefig(p, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    return p


def create_timing_variability_chart(
    instances_by_step,
    function_stats,
    timer_db,
    output_path,
    prefix,
    show_plot,
    out_dir,
):
    """Create function timing variability analysis chart.

    Args:
        instances_by_step: Dictionary mapping steps to timer instances
        function_stats: Dictionary of function timer statistics
        timer_db: Dictionary of timer definitions
        output_path: Base output path for saving
        prefix: Filename prefix
        show_plot: Whether to display the plot
        out_dir: Output directory name

    Returns:
        Path to saved plot
    """
    print("Creating timing variability analysis...")

    steps_with_data = sorted(
        [k for k in instances_by_step.keys() if k is not None]
    )

    if len(steps_with_data) <= 2 or not function_stats:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate coefficient of variation for each function
    variability_data = []
    for tid, stats in function_stats.items():
        if (
            stats["call_count"] > 2
        ):  # Need multiple calls to calculate variability
            # Get all execution times for this function
            all_times = []
            for step in steps_with_data:
                step_times = [
                    inst.time_ms
                    for inst in instances_by_step[step]
                    if inst.timer_id == tid
                ]
                all_times.extend(step_times)

            if len(all_times) > 2 and np.mean(all_times) > 0:
                cv = (
                    np.std(all_times) / np.mean(all_times) * 100
                )  # Coefficient of variation as %
                func_name = (
                    timer_db[tid].function if tid in timer_db else "Unknown"
                )
                clean_name = func_name.replace("_", " ").title()
                variability_data.append((clean_name, cv, stats["total_time"]))

    if not variability_data:
        return None

    # Sort by coefficient of variation
    variability_data.sort(key=lambda x: x[1], reverse=True)
    variability_data = variability_data[:15]  # Top 15 most variable

    names = [item[0] for item in variability_data]
    cvs = [item[1] for item in variability_data]
    total_times = [item[2] for item in variability_data]

    # Color by total execution time
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(names)))

    bars = ax.barh(
        range(len(names)),
        cvs,
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Coefficient of Variation (%)", fontsize=12)
    ax.set_title(
        "Function Timing Variability (Higher = More Variable)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add CV% labels on bars
    for i, (bar, cv, total_time) in enumerate(zip(bars, cvs, total_times)):
        label = f"{cv:.1f}%"
        ax.text(
            bar.get_width() * 1.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")

    plt.tight_layout()
    p = create_output_path(
        output_path, prefix, "08_timing_variability.png", out_dir
    )
    plt.savefig(p, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    return p


# ============================================================================
# HIERARCHICAL ANALYSIS FUNCTIONS
# ============================================================================


def get_nested_timers_for_function(
    func_name,
    all_stats_dict,
    timer_db_dict,
    nesting_db_dict,
    visited=None,
):
    """Get function timer.

    This function recursively collects all nested timers to any depth.

    Args:
        func_name (str): The name of the function to get timers for.
        all_stats_dict (dict): Dictionary of all timer stats.
        timer_db_dict (dict): Dictionary of all timer metadata.
        nesting_db_dict (dict): Dictionary of nesting relationships.
        visited (set): Set of already visited function names to prevent
           cycles.

    Returns:
        list: List of (timer_id, stats) tuples for the function and its
              nested timers.
    """
    if visited is None:
        visited = set()

    # Prevent infinite recursion
    if func_name in visited:
        return []

    visited.add(func_name)
    nested_timers = []

    # Find the function timer for this function
    function_timer = None
    for tid, stats in all_stats_dict.items():
        if tid.startswith("SYNTHETIC:"):
            # Handle synthetic timer IDs
            if tid[10:] == func_name:  # SYNTHETIC:func_name
                function_timer = (tid, stats)
                break
        elif (
            timer_db_dict[tid].function == func_name
            and timer_db_dict[tid].timer_type == "function"
        ):
            function_timer = (tid, stats)
            break

    if function_timer:
        nested_timers.append(function_timer)

    # Find operation timers that belong to this function according to
    # nesting DB
    if func_name in nesting_db_dict:
        nested_operations = nesting_db_dict[func_name].get(
            "nested_operations", []
        )

        for tid, stats in all_stats_dict.items():
            if tid.startswith("SYNTHETIC:"):
                continue  # Skip synthetic timers in operation lookup
            if (
                timer_db_dict[tid].function == func_name
                and timer_db_dict[tid].timer_type == "operation"
            ):
                # Check if this operation timer's description matches any
                # in the nesting DB
                timer_desc = timer_db_dict[tid].label_text
                for nested_op_pattern in nested_operations:
                    # Remove the format specifiers to match descriptions
                    expected_desc = (
                        nested_op_pattern.replace("%.3f %s.", "")
                        .replace("%.3f %s", "")
                        .strip()
                    )
                    if expected_desc in timer_desc or timer_desc.startswith(
                        expected_desc
                    ):
                        nested_timers.append((tid, stats))
                        break

        # Find nested function timers that belong to this function
        # according to nesting DB
        nested_functions = nesting_db_dict[func_name].get(
            "nested_functions", []
        )

        for nested_func_name in nested_functions:
            # Recursively get all timers for the nested function
            recursive_timers = get_nested_timers_for_function(
                nested_func_name,
                all_stats_dict,
                timer_db_dict,
                nesting_db_dict,
                visited.copy(),
            )
            nested_timers.extend(recursive_timers)

    return nested_timers


def build_function_hierarchy(
    func_name, all_stats_dict, timer_db_dict, nesting_db_dict, visited=None
):
    """Build a hierarchical structure for a function.

    Args:
        func_name: Name of the function to build hierarchy for
        all_stats_dict: Dictionary of all timer statistics
        timer_db_dict: Dictionary of timer definitions
        nesting_db_dict: Dictionary of nesting relationships
        visited: Set of visited functions to prevent cycles

    Returns:
        Dictionary containing hierarchical structure
    """
    if visited is None:
        visited = set()

    if func_name in visited:
        return {"function": None, "operations": [], "nested_functions": {}}

    visited.add(func_name)

    hierarchy = {
        "function": None,
        "operations": [],
        "nested_functions": {},
    }

    # Find the function timer (including synthetic ones)
    for tid, stats in all_stats_dict.items():
        if tid.startswith("SYNTHETIC:"):
            # Handle synthetic timer IDs
            if tid[10:] == func_name:  # SYNTHETIC:func_name
                hierarchy["function"] = (tid, stats)
                break
        elif (
            timer_db_dict[tid].function == func_name
            and timer_db_dict[tid].timer_type == "function"
        ):
            hierarchy["function"] = (tid, stats)
            break

    # Find operation timers belonging DIRECTLY to this function
    # (not recursively)
    if func_name in nesting_db_dict:
        nested_operations = nesting_db_dict[func_name].get(
            "nested_operations", []
        )

        for tid, stats in all_stats_dict.items():
            if tid.startswith("SYNTHETIC:"):
                continue  # Skip synthetic timers in operation lookup
            if (
                timer_db_dict[tid].function == func_name
                and timer_db_dict[tid].timer_type == "operation"
            ):
                timer_desc = timer_db_dict[tid].label_text
                for nested_op_pattern in nested_operations:
                    expected_desc = (
                        nested_op_pattern.replace("%.3f %s.", "")
                        .replace("%.3f %s", "")
                        .strip()
                    )
                    if expected_desc in timer_desc or timer_desc.startswith(
                        expected_desc
                    ):
                        hierarchy["operations"].append((tid, stats))
                        break

        # Build nested function hierarchies (using shared visited set to
        # prevent duplication)
        nested_functions = nesting_db_dict[func_name].get(
            "nested_functions", []
        )
        for nested_func_name in nested_functions:
            nested_hierarchy = build_function_hierarchy(
                nested_func_name,
                all_stats_dict,
                timer_db_dict,
                nesting_db_dict,
                visited,
            )
            if (
                nested_hierarchy["function"]
                or nested_hierarchy["operations"]
                or nested_hierarchy["nested_functions"]
            ):
                hierarchy["nested_functions"][nested_func_name] = (
                    nested_hierarchy
                )

    return hierarchy


def add_hierarchical_rows(
    rows,
    hierarchy,
    function_timer_time,
    timer_db_dict,
    indent,
    all_stats_dict,
):
    """Add rows to the table in hierarchical order.

    Args:
        rows: List to append table rows to
        hierarchy: Hierarchical structure for the function
        function_timer_time: Total function execution time
        timer_db_dict: Dictionary of timer definitions
        indent: Current indentation level
        all_stats_dict: Dictionary of all timer statistics
    """
    # Combine operations and nested functions, then sort by time
    all_items = []

    # Add operations from this level
    for tid, stats in hierarchy["operations"]:
        total_time = stats["total_time"]
        calls = stats["call_count"]
        pct_of_func = (
            (100 * total_time / function_timer_time)
            if function_timer_time > 0
            else 0.0
        )

        description = timer_db_dict[tid].label_text
        if " took " in description:
            description = description.split(" took ")[0]
        description = f"{indent}└─ {description}"

        row = [
            description[:60] + ("..." if len(description) > 60 else ""),
            f"{total_time:.1f}",
            f"{pct_of_func:.1f}%",
            f"{calls}",
            f"{(total_time / calls if calls > 0 else 0):.2f}",
        ]
        all_items.append((total_time, "operation", row, None, None))

    # Add nested functions
    for nested_func_name, nested_hierarchy in hierarchy[
        "nested_functions"
    ].items():
        if nested_hierarchy["function"]:
            _, func_stats = nested_hierarchy["function"]
            total_time = func_stats["total_time"]
            calls = func_stats["call_count"]
            pct_of_func = (
                (100 * total_time / function_timer_time)
                if function_timer_time > 0
                else 0.0
            )

            description = f"{indent}└─ {nested_func_name} (nested function)"

            row = [
                description[:60] + ("..." if len(description) > 60 else ""),
                f"{total_time:.1f}",
                f"{pct_of_func:.1f}%",
                f"{calls}",
                f"{(total_time / calls if calls > 0 else 0):.2f}",
            ]
            all_items.append(
                (
                    total_time,
                    "function",
                    row,
                    nested_func_name,
                    nested_hierarchy,
                )
            )
        else:
            # If no function timer, but we have operations, create a
            # synthetic function entry
            if nested_hierarchy["operations"]:
                total_ops_time = sum(
                    stats["total_time"]
                    for _, stats in nested_hierarchy["operations"]
                )
                # Use the call count from the first operation as an
                # approximation
                calls = (
                    nested_hierarchy["operations"][0][1]["call_count"]
                    if nested_hierarchy["operations"]
                    else 1
                )
                pct_of_func = (
                    (100 * total_ops_time / function_timer_time)
                    if function_timer_time > 0
                    else 0.0
                )

                description = (
                    f"{indent}└─ {nested_func_name} (nested function)"
                )

                row = [
                    description[:60]
                    + ("..." if len(description) > 60 else ""),
                    f"{total_ops_time:.1f}",
                    f"{pct_of_func:.1f}%",
                    f"{calls}",
                    f"{(total_ops_time / calls if calls > 0 else 0):.2f}",
                ]
                all_items.append(
                    (
                        total_ops_time,
                        "function",
                        row,
                        nested_func_name,
                        nested_hierarchy,
                    )
                )
            else:
                # If no function timer and no operations, skip completely
                continue

    # Calculate unaccounted time (only for top level, when indent is "")
    if indent == "":
        # Sum up only direct children (level 1) - operations from this
        # function plus nested functions
        # This avoids double-counting since nested functions already
        # include their own operations
        accounted_time = 0.0

        # Add direct operations from this function level
        for tid, stats in hierarchy["operations"]:
            accounted_time += stats["total_time"]

        # Add direct nested functions (they already include their own
        # nested content)
        for nested_func_name, nested_hierarchy in hierarchy[
            "nested_functions"
        ].items():
            if nested_hierarchy["function"]:
                _, func_stats = nested_hierarchy["function"]
                accounted_time += func_stats["total_time"]

        # Calculate unaccounted time as the difference between total
        # function time and the sum of direct children
        unaccounted_time = function_timer_time - accounted_time
        if unaccounted_time > 0.1:  # Only show if significant (> 0.1 ms)
            pct_unaccounted = (
                (100 * unaccounted_time / function_timer_time)
                if function_timer_time > 0
                else 0.0
            )
            unaccounted_row = [
                f"{indent}└─ UNACCOUNTED TIME",
                f"{unaccounted_time:.1f}",
                f"{pct_unaccounted:.1f}%",
                "-",
                "-",
            ]
            # Add unaccounted time to the items list so it gets sorted
            # properly by time
            all_items.append(
                (
                    unaccounted_time,
                    "unaccounted",
                    unaccounted_row,
                    None,
                    None,
                )
            )

    # Sort all items by time descending
    all_items.sort(key=lambda x: x[0], reverse=True)

    # Add items to rows in sorted order
    for (
        total_time,
        item_type,
        row,
        nested_func_name,
        nested_hierarchy,
    ) in all_items:
        if row is not None:
            rows.append(row)

        # If this is a nested function, recursively add its content
        if item_type == "function" and nested_hierarchy is not None:
            add_hierarchical_rows(
                rows,
                nested_hierarchy,
                function_timer_time,
                timer_db_dict,
                indent + "   ",
                all_stats_dict,
            )


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================


def analyse_swift_log_timings(
    log_file: str,
    output_path: str | None = None,
    prefix: str | None = None,
    show_plot: bool = True,
    top_n: int = 20,
    hierarchy_functions: list[str] | None = None,
) -> None:
    """Analyse SWIFT timing logs and emit comprehensive timing reports.

    This function provides advanced timing analysis of SWIFT simulation logs,
    featuring hierarchical timing breakdowns that avoid double-counting and
    provide clear visibility into function performance.

    Key Features:
      * Automatically generates timer nesting database from SWIFT source code
      * Parses log files to extract timer instances and step information
      * Creates hierarchical timing tables showing function call trees
      * Implements function timer = max(explicit_timer, sum_of_nested) rule
      * Shows "UNACCOUNTED TIME" for missing instrumentation
      * Produces standard plots: top timers, distributions, task counts, etc.
      * Estimates per-step untimed overhead when step totals are available

    Hierarchical Analysis:
      * Functions are analyzed using nesting relationships from source code
      * Direct operations and nested function calls are properly accounted
      * Percentages are calculated against corrected function timer totals
      * Tables show proper indentation and avoid double-counting issues

    Notes:
      * Nesting database is regenerated automatically from SWIFT source code
      * Display names use `function [file:line]` format for traceability
      * Function timers use the maximum of explicit timers and nested sums

    Args:
      log_file: Path to the SWIFT log file to analyse.
      output_path: Directory where figures are saved. If None, saves to CWD.
      prefix: Optional filename prefix and output subdirectory prefix.
      show_plot: Whether to display plots interactively.
      top_n: Number of top timers shown in multi-item plots/tables.
      hierarchy_functions: Functions to show hierarchical timing for. If None,
        shows default important functions. Use ["all"] to show all functions.

    Returns:
      None. Figures are written to disk; a textual summary is printed to
            stdout.
    """
    print(f"Analyzing SWIFT log:  {log_file}")

    # Output directory naming identical to your original approach
    out_dir = (
        "runtime_analysis" if prefix is None else f"{prefix}_runtime_analysis"
    )

    print("Loading timer database and compiling patterns...")
    # Load DB + compile patterns
    timer_db = load_timer_db()
    compiled = compile_site_patterns(timer_db)
    nesting_db = load_timer_nesting(auto_generate=True, force_regenerate=True)
    print(
        f"Loaded {len(timer_db)} timer definitions and compiled "
        f"{len(compiled)} patterns"
    )
    print(f"Loaded nesting relationships for {len(nesting_db)} functions")

    # Scan log to collect per-step timer instances (encounter order)
    print("Scanning log file for timer instances...")
    instances_by_step, _ = scan_log_instances_by_step(
        log_file, compiled, timer_db
    )

    total_instances = sum(
        len(instances) for instances in instances_by_step.values()
    )
    print(
        f"Found {total_instances} timer instances across "
        f"{len(instances_by_step)} steps"
    )

    # Classify timers: timer with max time per function becomes function timer
    print(
        "Classifying timers: max time per function becomes function timer..."
    )
    function_timer_ids = classify_timers_by_max_time(
        instances_by_step, timer_db, nesting_db
    )
    print(f"Identified {len(function_timer_ids)} function timers")

    # Update timer instances with the new classification
    for inst_list in instances_by_step.values():
        for inst in inst_list:
            if inst.timer_id in function_timer_ids:
                inst.timer_type = "function"
            else:
                inst.timer_type = "operation"

    # Aggregate timing statistics by timer site and timer type
    print("Aggregating timing statistics by timer site and type...")
    function_times_by_tid: Dict[str, List[float]] = defaultdict(list)
    operation_times_by_tid: Dict[str, List[float]] = defaultdict(list)
    call_counts_by_tid: Counter = Counter()

    for inst_list in instances_by_step.values():
        for inst in inst_list:
            if inst.timer_type == "function":
                function_times_by_tid[inst.timer_id].append(inst.time_ms)
            else:  # operation
                operation_times_by_tid[inst.timer_id].append(inst.time_ms)
            call_counts_by_tid[inst.timer_id] += 1

    # Build stats for function and operation timers
    function_stats = {
        tid: build_stats(vals) for tid, vals in function_times_by_tid.items()
    }

    # Note: No synthetic timers - only use actual function timers from the log
    operation_stats = {
        tid: build_stats(vals) for tid, vals in operation_times_by_tid.items()
    }

    # Sort by total time for each timer type
    sorted_functions = sorted(
        function_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
    )

    # For legacy compatibility, combine all timers for some plots
    all_stats = {**function_stats, **operation_stats}
    sorted_all = sorted(
        all_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
    )
    sorted_by_calls = sorted(
        all_stats.items(),
        key=lambda x: x[1]["call_count"],
        reverse=True,
    )

    print("Parsing task categories, step info, and additional log data...")
    step_info: List[Dict[str, float | int]] = []
    task_category_times: Dict[str, List[float]] = defaultdict(list)
    task_counts: Dict[str, List[int]] = defaultdict(list)
    step_totals: Dict[
        int, float
    ] = {}  # wall-clock ms per step (last float on step line)
    step_seen_order: List[int] = []

    # Regex mirrors your original, plus helpers for categories & counts
    step_line_re = re.compile(
        r"^\s*(\d+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)"
    )
    task_line_re = re.compile(
        r"\*\*\*\s+([^:]+):\s+([\d.]+)\s+ms\s+\(([\d.]+)\s%?\)"
    )
    counts_re = re.compile(r"task counts are \[(.*)\]")
    last_float_re = re.compile(r"([-\d.]+)(?:\s*)$")

    # Read the log file and count lines for progress
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    print(
        f"Processing {len(lines)} log lines for task categories and "
        "step info..."
    )

    for raw in tqdm(lines, desc="Parsing log lines", unit="line"):
        line = raw.strip()

        # Step info rows (capture first 5 numeric columns as in your original)
        if sm := step_line_re.match(line):
            try:
                step_num = int(sm.group(1))
                step_seen_order.append(step_num)
                step_info.append(
                    dict(
                        step=step_num,
                        time_1=float(sm.group(2)),
                        time_2=float(sm.group(3)),
                        time_3=float(sm.group(4)),
                        time_4=float(sm.group(5)),
                    )
                )
                # Also capture step total wall-clock ms as last float on
                # the line
                if wm := last_float_re.search(line):
                    try:
                        step_totals[step_num] = float(wm.group(1))
                    except ValueError:
                        pass
            except ValueError:
                # Skip malformed lines (keep behavior similar to your
                # warning pattern)
                pass
            continue

        # Task category summary rows
        if tm := task_line_re.search(line):
            category = tm.group(1).strip()
            if category.lower() != "total":
                try:
                    time_ms = float(tm.group(2))
                    task_category_times[category].append(time_ms)
                except ValueError:
                    pass
            continue

        # Task counts rows
        if cm := counts_re.search(line):
            parts = cm.group(1).split()
            for token in parts:
                if "=" in token:
                    k, v = token.split("=", 1)
                    try:
                        task_counts[k].append(int(v))
                    except ValueError:
                        pass
            continue

    print("Generating analysis plots...")
    plots_created: List[str] = []

    # Create all the plots
    plot_functions: list[tuple[Callable, tuple]] = [
        (
            create_function_timing_chart,
            (
                sorted_functions,
                timer_db,
                output_path,
                prefix,
                show_plot,
                out_dir,
            ),
        ),
        (
            create_function_distribution_chart,
            (
                sorted_functions,
                timer_db,
                output_path,
                prefix,
                show_plot,
                out_dir,
            ),
        ),
        (
            create_function_efficiency_chart,
            (
                sorted_functions,
                timer_db,
                output_path,
                prefix,
                show_plot,
                out_dir,
            ),
        ),
        (
            create_timer_distribution_histogram,
            (all_stats, output_path, prefix, show_plot, out_dir),
        ),
        (
            create_task_category_overview,
            (task_category_times, output_path, prefix, show_plot, out_dir),
        ),
        (
            create_performance_summary_scatter,
            (
                function_stats,
                timer_db,
                output_path,
                prefix,
                show_plot,
                out_dir,
            ),
        ),
        (
            create_time_series_plot,
            (
                instances_by_step,
                function_stats,
                timer_db,
                output_path,
                prefix,
                show_plot,
                out_dir,
            ),
        ),
        (
            create_timing_variability_chart,
            (
                instances_by_step,
                function_stats,
                timer_db,
                output_path,
                prefix,
                show_plot,
                out_dir,
            ),
        ),
    ]

    for plot_func, args in plot_functions:
        try:
            plot_path = plot_func(*args)
            if plot_path:
                plots_created.append(plot_path)
        except Exception as e:
            print(
                f"Warning: Failed to create plot with {plot_func.__name__}: "
                f"{e}"
            )

    print(f"\nCreated {len(plots_created)} plots:")
    for p in plots_created:
        print(f"  - {p}")

    # Print detailed analysis tables and summaries
    _print_analysis_tables(
        sorted_functions,
        sorted_all,
        sorted_by_calls,
        all_stats,
        function_stats,
        operation_stats,
        timer_db,
        nesting_db,
        task_category_times,
        task_counts,
        top_n,
        hierarchy_functions,
    )


def _print_analysis_tables(
    sorted_functions,
    sorted_all,
    sorted_by_calls,
    all_stats,
    function_stats,
    operation_stats,
    timer_db,
    nesting_db,
    task_category_times,
    task_counts,
    top_n,
    hierarchy_functions,
):
    """Print detailed analysis tables and summaries.

    This is a helper function to keep the main analysis function more readable.
    """
    # Totals
    total_function = sum(v["total_time"] for v in function_stats.values())
    total_operation = sum(v["total_time"] for v in operation_stats.values())
    total_all = total_function + total_operation

    # A) Top function timers table with operation breakdown
    headers = [
        "Function Timer",
        "Total Time (ms)",
        "% of All Functions",
        "Calls",
        "Avg/call (ms)",
        "Operations Sum (ms)",
        "% Accounted",
    ]
    rows = []
    for tid, s in sorted_functions[: min(top_n, 25)]:
        total_time = s["total_time"]
        calls = s["call_count"]
        nm = display_name(tid, timer_db)
        if tid.startswith("SYNTHETIC:"):
            function_name = tid[10:]  # Remove "SYNTHETIC:" prefix
        else:
            function_name = timer_db[tid].function

        # Sum operation timers for this function
        operation_sum = 0.0
        for op_tid, op_s in operation_stats.items():
            if timer_db[op_tid].function == function_name:
                operation_sum += op_s["total_time"]

        # Calculate what percentage of function time is accounted for by
        # operations
        accounted_pct = (
            (100 * operation_sum / total_time) if total_time > 0 else 0.0
        )

        rows.append(
            [
                (nm if len(nm) <= 60 else nm[:60] + "..."),
                f"{total_time:.1f}",
                f"{(100 * total_time / total_function):.1f}%",
                f"{calls}",
                f"{(total_time / calls if calls > 0 else 0):.2f}",
                f"{operation_sum:.1f}",
                f"{accounted_pct:.1f}%",
            ]
        )
    print(
        "\n"
        + create_ascii_table(
            headers, rows, "TOP FUNCTION TIMERS (with operation breakdown)"
        )
    )

    # B) Timers grouped by function using nesting relationships
    _print_hierarchical_analysis(
        all_stats, timer_db, nesting_db, hierarchy_functions, top_n
    )

    # C) Top timers by **call count** (table)
    headers = [
        "Timer (function [file:line])",
        "Call Count",
        "Total Time (ms)",
        "Type",
        "Avg/call (ms)",
    ]
    rows = []
    for tid, _ in sorted_by_calls[: min(top_n, 25)]:
        calls = all_stats.get(tid, {}).get("call_count", 0)
        total_time = all_stats.get(tid, {}).get("total_time", 0.0)
        timer_type = timer_db[tid].timer_type if tid in timer_db else "unknown"
        nm = display_name(tid, timer_db)
        rows.append(
            [
                (nm if len(nm) <= 60 else nm[:60] + "..."),
                f"{calls}",
                f"{total_time:.1f}",
                timer_type,
                f"{(total_time / calls if calls > 0 else 0):.2f}",
            ]
        )
    print("\n" + create_ascii_table(headers, rows, "TOP TIMERS BY CALL COUNT"))

    # Task category and count summaries
    _print_task_summaries(task_category_times, task_counts)

    # Overall summary
    _print_overall_summary(
        total_function,
        total_operation,
        total_all,
        function_stats,
        operation_stats,
        all_stats,
        sorted_all,
    )


def _print_hierarchical_analysis(
    all_stats, timer_db, nesting_db, hierarchy_functions, top_n
):
    """Print hierarchical timing analysis tables."""
    print("\n" + "=" * 100)
    print("TIMERS BY FUNCTION (using nesting relationships)")
    print("=" * 100)

    # Calculate total time per function including only nested timers
    func_totals = {}
    for func_name in nesting_db.keys():
        nested_timers = get_nested_timers_for_function(
            func_name, all_stats, timer_db, nesting_db
        )
        if nested_timers:
            func_totals[func_name] = sum(
                s["total_time"] for _, s in nested_timers
            )

    # Also include functions that have timers but no nesting info
    for tid, s in all_stats.items():
        if tid.startswith("SYNTHETIC:"):
            continue  # Skip synthetic timers in hierarchical analysis
        func_name = timer_db[tid].function
        if func_name not in func_totals:
            if func_name not in nesting_db:  # Only if not in nesting DB
                if func_name not in func_totals:
                    func_totals[func_name] = 0
                func_totals[func_name] += s["total_time"]

    sorted_funcs = sorted(
        func_totals.items(), key=lambda x: x[1], reverse=True
    )

    # Determine which functions to show hierarchically
    if hierarchy_functions is None:
        # Default to showing a few key functions that are typically important
        default_functions = [
            "engine_prepare",
            "engine_rebuild",
            "space_rebuild",
            "space_split",
            "engine_maketasks",
        ]
        functions_to_show = [
            (func_name, func_totals.get(func_name, 0))
            for func_name in default_functions
            if func_name in func_totals
        ]
        # Sort by total time
        functions_to_show.sort(key=lambda x: x[1], reverse=True)
    elif len(hierarchy_functions) == 1 and hierarchy_functions[0] == "all":
        # Show all functions with hierarchy data (up to top_n limit)
        functions_to_show = sorted_funcs[: min(top_n, 10)]
    else:
        # Show only the specified functions
        functions_to_show = [
            (func_name, func_totals.get(func_name, 0))
            for func_name in hierarchy_functions
            if func_name in func_totals
        ]
        # Sort by total time
        functions_to_show.sort(key=lambda x: x[1], reverse=True)

    for func_name, func_total in functions_to_show:
        # Build hierarchical timer structure for this function (non-recursive
        # for display)
        if func_name in nesting_db:
            hierarchy = build_function_hierarchy(
                func_name, all_stats, timer_db, nesting_db
            )
        else:
            # Fall back to all timers in this function if no nesting info
            timers = [
                (tid, s)
                for tid, s in all_stats.items()
                if timer_db[tid].function == func_name
            ]
            hierarchy = {
                "function": None,
                "operations": [],
                "nested_functions": {},
            }
            for tid, s in timers:
                if timer_db[tid].timer_type == "function":
                    hierarchy["function"] = (tid, s)
                else:
                    hierarchy["operations"].append((tid, s))

        # Skip if no timers found for this function at all
        if (
            not hierarchy["function"]
            and not hierarchy["operations"]
            and not hierarchy["nested_functions"]
        ):
            continue

        # Calculate function timer time using the rule:
        # function_timer = max(explicit_function_timer, sum_of_nested_timers)
        explicit_function_timer_time = 0.0
        if hierarchy["function"]:
            _, func_stats = hierarchy["function"]
            explicit_function_timer_time = func_stats["total_time"]

        # Calculate sum of all nested timers (operations + nested functions)
        nested_sum = 0.0
        for _, stats in hierarchy["operations"]:
            nested_sum += stats["total_time"]

        for nested_func_hierarchy in hierarchy["nested_functions"].values():
            if nested_func_hierarchy["function"]:
                _, nested_func_stats = nested_func_hierarchy["function"]
                nested_sum += nested_func_stats["total_time"]

        # Apply the rule: function timer = max(explicit, sum_of_nested)
        function_timer_time = max(explicit_function_timer_time, nested_sum)

        if function_timer_time == 0.0:
            continue  # Skip if no timer data found

        print(
            f"\n{func_name}: {function_timer_time:.1f} ms "
            "(function execution time)"
        )
        print("-" * (len(func_name) + 40))

        headers = [
            "Timer Description",
            "Time (ms)",
            "% of Function Time",
            "Calls",
            "Avg/call (ms)",
        ]
        rows = []

        # First add the function timer itself (always 100%)
        if hierarchy["function"]:
            tid, func_stats = hierarchy["function"]
            function_label = "Function execution time"
            if tid.startswith("SYNTHETIC:"):
                function_label = "Function execution time (sum of operations)"
            # Use function_timer_time (the corrected max value) instead of
            # func_stats total_time
            pcent = function_timer_time / func_stats["call_count"]
            rows.append(
                [
                    function_label,
                    f"{function_timer_time:.1f}",
                    "100.0%",
                    f"{func_stats['call_count']}",
                    f"{pcent:.2f}",
                ]
            )

        # Add hierarchical rows with proper ordering
        add_hierarchical_rows(
            rows, hierarchy, function_timer_time, timer_db, "", all_stats
        )

        if len(rows) > 1:  # Only show if there are nested items
            print(create_ascii_table(headers, rows, ""))


def _print_task_summaries(task_category_times, task_counts):
    """Print task category and count summaries."""
    # Task category summary (if present)
    if task_category_times:
        headers = [
            "Task Category",
            "Total Time (ms)",
            "% of Category Time",
            "Avg/Step (ms)",
            "Steps",
        ]
        rows = []
        cat_total_time = sum(
            sum(vals) for vals in task_category_times.values()
        )
        for cat, times in sorted(
            task_category_times.items(), key=lambda x: sum(x[1]), reverse=True
        ):
            if times:
                total_cat = sum(times)
                avg = total_cat / len(times)
                pct = (
                    (100 * total_cat / cat_total_time)
                    if cat_total_time > 0
                    else 0.0
                )
                rows.append(
                    [
                        (cat if len(cat) <= 30 else cat[:30] + "..."),
                        f"{total_cat:.1f}",
                        f"{pct:.1f}%",
                        f"{avg:.1f}",
                        f"{len(times)}",
                    ]
                )
        print(
            "\n" + create_ascii_table(headers, rows, "TASK CATEGORY SUMMARY")
        )

    # Task count summary (if present)
    if task_counts:
        headers = ["Task Type", "Total Count", "Max Count", "Avg/Step"]
        rows = []
        for tname, counts in sorted(
            task_counts.items(), key=lambda x: sum(x[1]), reverse=True
        ):
            if counts:
                tot = sum(counts)
                mx = max(counts)
                avg = tot / len(counts)
                if tot > 0:
                    rows.append(
                        [
                            (
                                tname
                                if len(tname) <= 25
                                else tname[:25] + "..."
                            ),
                            f"{tot}",
                            f"{mx}",
                            f"{avg:.1f}",
                        ]
                    )
        print("\n" + create_ascii_table(headers, rows, "TASK COUNT SUMMARY"))


def _print_overall_summary(
    total_function,
    total_operation,
    total_all,
    function_stats,
    operation_stats,
    all_stats,
    sorted_all,
):
    """Print overall performance summary."""
    print("\nPERFORMANCE SUMMARY")
    print("-" * 100)
    print(f"Total function timer time:              {total_function:.1f} ms")
    print(f"Total operation timer time:             {total_operation:.1f} ms")
    print(f"Total all timer time:                   {total_all:.1f} ms")
    print(f"Function timers:                        {len(function_stats)}")
    print(f"Operation timers:                       {len(operation_stats)}")
    print(f"Total unique timers:                    {len(all_stats)}")
    print(
        "Total timer call instances:             "
        f"{sum(s['call_count'] for s in all_stats.values())}"
    )

    # Top-k coverage
    if len(sorted_all) >= 5:
        top_5 = sum(s["total_time"] for _, s in sorted_all[:5])
        pct = (100 * top_5 / total_all) if total_all > 0 else 0.0
        print(
            f"Top 5 timers account for:              {pct:.1f}% of total time"
        )
    if len(sorted_all) >= 10:
        top_10 = sum(s["total_time"] for _, s in sorted_all[:10])
        pct = (100 * top_10 / total_all) if total_all > 0 else 0.0
        print(
            f"Top 10 timers account for:             {pct:.1f}% of total time"
        )

    # Type breakdown
    if total_all > 0:
        print(
            f"Function timer share:                   "
            f"{100 * total_function / total_all:.1f}%"
        )
        print(
            f"Operation timer share:                  "
            f"{100 * total_operation / total_all:.1f}%"
        )
