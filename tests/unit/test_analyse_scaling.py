"""Unit tests for scaling analysis helpers."""

from pathlib import Path

import pytest

from swiftsim_cli.modes.analyse.scaling import (
    ScalingLogData,
    _aggregate_timer_totals,
    _build_legend_labels,
    _build_main_series_dict,
    _build_perfect_scaling_series,
    _build_perfect_speedup_series,
    _build_scaling_analysis,
    _build_total_series,
    _clean_timer_label_text,
    _collect_comparable_ranks,
    _compute_speedup_series,
    _extract_rank_count_from_log,
    _fit_scaling_slope,
    _legend_columns,
    _main_timer_value,
    _main_total_value,
    _plot_would_use_log_scale,
    _rank_source_description,
    _rank_source_title,
    _rank_total_value,
    _run_statistic_description,
    _run_statistic_title,
    _select_timer_ids,
    _should_use_log_scale,
    _wrap_legend_label,
)
from swiftsim_cli.src_parser import TimerDef, TimerInstance


def test_extract_rank_count_from_log(tmp_path: Path):
    """MPI rank count is extracted from the log header."""
    log_file = tmp_path / "swift.log"
    log_file.write_text(
        "[0000] main: MPI is up and running with 16 node(s).\n",
        encoding="utf-8",
    )

    assert _extract_rank_count_from_log(str(log_file)) == 16


def test_aggregate_timer_totals_filters_by_step():
    """Only instances within the requested step range are accumulated."""
    instances_by_step = {
        1: [
            TimerInstance(
                timer_id="timer_a",
                function="func",
                step=1,
                time_ms=10.0,
                line_index=0,
                timer_type="timer",
                rank=0,
            )
        ],
        2: [
            TimerInstance(
                timer_id="timer_a",
                function="func",
                step=2,
                time_ms=5.0,
                line_index=1,
                timer_type="timer",
                rank=1,
            )
        ],
    }

    timer_totals, timer_call_counts, per_rank_totals, per_rank_call_counts = (
        _aggregate_timer_totals(instances_by_step, (2, 2))
    )

    assert timer_totals == {"timer_a": 5.0}
    assert timer_call_counts == {"timer_a": 1}
    assert per_rank_totals == {1: {"timer_a": 5.0}}
    assert per_rank_call_counts == {1: {"timer_a": 1}}


def test_select_timer_ids_uses_max_percentage_threshold():
    """A timer is kept if it crosses the threshold in any one log."""
    scaling_data = [
        ScalingLogData(
            log_file="a.log",
            label="a.log",
            rank_count=16,
            timer_totals={"timer_a": 50.0, "timer_b": 5.0},
            timer_percentages={"timer_a": 50.0, "timer_b": 5.0},
            total_time=100.0,
            per_rank_totals={0: {"timer_a": 50.0, "timer_b": 5.0}},
            timer_call_counts={"timer_a": 5, "timer_b": 1},
            total_call_count=6,
            per_rank_call_counts={0: {"timer_a": 5, "timer_b": 1}},
            emitted_rank_count=1,
        ),
        ScalingLogData(
            log_file="b.log",
            label="b.log",
            rank_count=32,
            timer_totals={"timer_a": 4.0, "timer_b": 12.0},
            timer_percentages={"timer_a": 4.0, "timer_b": 12.0},
            total_time=100.0,
            per_rank_totals={0: {"timer_a": 4.0, "timer_b": 12.0}},
            timer_call_counts={"timer_a": 1, "timer_b": 3},
            total_call_count=4,
            per_rank_call_counts={0: {"timer_a": 1, "timer_b": 3}},
            emitted_rank_count=1,
        ),
    ]

    assert _select_timer_ids(scaling_data, 10.0) == ["timer_a", "timer_b"]


def test_fit_scaling_slope_matches_inverse_scaling():
    """Ideal strong scaling produces a slope of about -1."""
    slope = _fit_scaling_slope([16, 32, 64], [64.0, 32.0, 16.0])

    assert slope is not None
    assert slope == pytest.approx(-1.0)


def test_should_use_log_scale_requires_large_dynamic_range():
    """Log scale is only enabled for sufficiently wide ranges."""
    assert _should_use_log_scale([1.0, 100.0])
    assert not _should_use_log_scale([10.0, 20.0, 30.0])


def test_build_total_series_uses_total_time_by_default():
    """The default main series uses rank 0 only."""
    scaling_data = [
        ScalingLogData(
            log_file="a.log",
            label="a.log",
            rank_count=2,
            timer_totals={"timer_a": 30.0},
            timer_percentages={"timer_a": 30.0},
            total_time=100.0,
            per_rank_totals={0: {"timer_a": 30.0}},
            timer_call_counts={"timer_a": 3},
            total_call_count=3,
            per_rank_call_counts={0: {"timer_a": 3}},
            emitted_rank_count=1,
        ),
        ScalingLogData(
            log_file="b.log",
            label="b.log",
            rank_count=4,
            timer_totals={"timer_a": 10.0},
            timer_percentages={"timer_a": 10.0},
            total_time=60.0,
            per_rank_totals={0: {"timer_a": 10.0}},
            timer_call_counts={"timer_a": 2},
            total_call_count=2,
            per_rank_call_counts={0: {"timer_a": 2}},
            emitted_rank_count=1,
        ),
    ]

    assert _build_total_series(scaling_data) == ([2, 4], [30.0, 10.0])
    assert _build_total_series(scaling_data, run_statistic="mean") == (
        [2, 4],
        [10.0, 5.0],
    )
    assert _build_total_series(scaling_data, rank_source="all") == (
        [2, 4],
        [100.0, 60.0],
    )


def test_build_total_series_uses_per_rank_totals_when_requested():
    """Per-rank totals are summed only for the selected rank."""
    scaling_data = [
        ScalingLogData(
            log_file="a.log",
            label="a.log",
            rank_count=2,
            timer_totals={"timer_a": 30.0},
            timer_percentages={"timer_a": 30.0},
            total_time=100.0,
            per_rank_totals={0: {"timer_a": 10.0}, 1: {"timer_a": 15.0}},
            timer_call_counts={"timer_a": 5},
            total_call_count=5,
            per_rank_call_counts={0: {"timer_a": 2}, 1: {"timer_a": 3}},
            emitted_rank_count=2,
        ),
        ScalingLogData(
            log_file="b.log",
            label="b.log",
            rank_count=4,
            timer_totals={"timer_a": 10.0},
            timer_percentages={"timer_a": 10.0},
            total_time=60.0,
            per_rank_totals={1: {"timer_a": 8.0, "timer_b": 2.0}},
            timer_call_counts={"timer_a": 4, "timer_b": 1},
            total_call_count=5,
            per_rank_call_counts={1: {"timer_a": 4, "timer_b": 1}},
            emitted_rank_count=1,
        ),
    ]

    assert _build_total_series(scaling_data, per_rank=1) == (
        [2, 4],
        [15.0, 10.0],
    )
    assert _build_total_series(
        scaling_data, per_rank=1, run_statistic="mean"
    ) == (
        [2, 4],
        [5.0, 2.0],
    )


def test_build_perfect_scaling_series_is_anchored_to_first_point():
    """Perfect scaling follows a 1/N trend from the smallest-rank run."""
    assert _build_perfect_scaling_series([2, 4, 8], [100.0, 0.0, 0.0]) == [
        100.0,
        50.0,
        25.0,
    ]


def test_speedup_helpers_match_relative_improvement():
    """Speedup helpers normalize both measured and perfect curves correctly."""
    assert _compute_speedup_series([100.0, 50.0, 25.0]) == [1.0, 2.0, 4.0]
    assert _build_perfect_speedup_series([2, 4, 8]) == [1.0, 2.0, 4.0]


def test_legend_columns_prefers_more_rows_over_excessive_width():
    """Legend columns are fixed to three where possible."""
    assert _legend_columns(["only one"]) == 1
    assert _legend_columns(["a", "b"]) == 2
    assert _legend_columns(["a", "b", "c"]) == 3
    assert _legend_columns(["short"] * 30) == 3


def test_wrap_legend_label_breaks_long_labels_on_words():
    """Long legend labels are wrapped to avoid horizontal overlap."""
    wrapped = _wrap_legend_label("engine prepare rebuild reweight", width=12)

    assert "\n" in wrapped
    assert "rebuild" in wrapped


def test_build_legend_labels_adds_total_and_reference_entries():
    """Legend labels include the derived helper series first."""
    timer_db = {
        "timer_a": TimerDef(
            timer_id="timer_a",
            function="engine_rebuild",
            log_pattern="pattern",
            start_line=1,
            end_line=2,
            label_text="took %.3f %s.",
            timer_type="timer",
        )
    }

    labels = _build_legend_labels(["timer_a"], timer_db)

    assert labels[0] == "Total"
    assert labels[1] == "Perfect scaling"
    assert labels[2] == "engine_rebuild"

    labels_without_total = _build_legend_labels(
        ["timer_a"], timer_db, include_total_reference=False
    )
    assert labels_without_total == ["engine_rebuild"]


def test_clean_timer_label_text_strips_timing_boilerplate():
    """Display label cleanup removes timing-only wrappers and format tokens."""
    assert _clean_timer_label_text("(%s)") == ""
    assert (
        _clean_timer_label_text(
            "took %.3f %s (including unskip, rebuild and reweight)."
        )
        == "including unskip, rebuild and reweight"
    )


def test_main_series_supports_root_vs_all_and_sum_vs_mean():
    """Main series values separate run statistic from rank source."""
    log_data = ScalingLogData(
        log_file="a.log",
        label="a.log",
        rank_count=4,
        timer_totals={"timer_a": 12.0, "timer_b": 6.0},
        timer_percentages={"timer_a": 66.7, "timer_b": 33.3},
        total_time=18.0,
        per_rank_totals={
            0: {"timer_a": 4.0},
            1: {"timer_a": 8.0, "timer_b": 6.0},
        },
        timer_call_counts={"timer_a": 3, "timer_b": 2},
        total_call_count=5,
        per_rank_call_counts={
            0: {"timer_a": 1},
            1: {"timer_a": 2, "timer_b": 2},
        },
        emitted_rank_count=2,
    )

    assert _build_main_series_dict(log_data, "sum", "root") == {"timer_a": 4.0}
    assert _build_main_series_dict(log_data, "sum", "all") == {
        "timer_a": 12.0,
        "timer_b": 6.0,
    }
    assert _build_main_series_dict(log_data, "mean", "root") == {
        "timer_a": 4.0
    }
    assert _build_main_series_dict(log_data, "mean", "all") == {
        "timer_a": 4.0,
        "timer_b": 3.0,
    }
    assert _main_timer_value(log_data, "timer_a", "sum", "all") == 12.0
    assert _main_timer_value(log_data, "timer_a", "mean", "all") == 4.0
    assert _main_total_value(log_data, "sum", "all") == 18.0
    assert _main_total_value(log_data, "mean", "all") == pytest.approx(3.6)
    assert _main_total_value(log_data, "sum", "root") == 4.0
    assert _rank_total_value(log_data, 1, "sum") == 14.0
    assert _rank_total_value(log_data, 1, "mean") == pytest.approx(14.0 / 4.0)


def test_statistic_and_rank_source_labels_are_human_readable():
    """Caption helpers return readable text for both dimensions."""
    assert _run_statistic_title("sum") == "Summed Over Run"
    assert _run_statistic_title("mean") == "Mean Per Call"
    assert (
        _run_statistic_description("sum") == "summed timer time over the run"
    )
    assert (
        _run_statistic_description("mean")
        == "average timer time per call over the run"
    )
    assert _rank_source_title("root") == "Rank 0 Only"
    assert _rank_source_title("all") == "All Emitting Ranks"
    assert _rank_source_description("root") == "rank 0 only"
    assert (
        _rank_source_description("all")
        == "all emitting ranks present in the log"
    )


def test_collect_comparable_ranks_requires_at_least_two_runs():
    """Timer-rank comparison plots only use ranks seen in multiple logs."""
    scaling_data = [
        ScalingLogData(
            log_file="a.log",
            label="a.log",
            rank_count=4,
            timer_totals={"timer_a": 12.0},
            timer_percentages={"timer_a": 100.0},
            total_time=12.0,
            per_rank_totals={0: {"timer_a": 4.0}, 1: {"timer_a": 8.0}},
            timer_call_counts={"timer_a": 3},
            total_call_count=3,
            per_rank_call_counts={0: {"timer_a": 1}, 1: {"timer_a": 2}},
            emitted_rank_count=2,
        ),
        ScalingLogData(
            log_file="b.log",
            label="b.log",
            rank_count=8,
            timer_totals={"timer_a": 18.0},
            timer_percentages={"timer_a": 100.0},
            total_time=18.0,
            per_rank_totals={0: {"timer_a": 6.0}, 2: {"timer_a": 12.0}},
            timer_call_counts={"timer_a": 4},
            total_call_count=4,
            per_rank_call_counts={0: {"timer_a": 1}, 2: {"timer_a": 3}},
            emitted_rank_count=2,
        ),
    ]

    assert _collect_comparable_ranks(scaling_data) == [0]


def test_plot_would_use_log_scale_ignores_hidden_mean_total_series():
    """Mean views only consider the series that are actually plotted."""
    scaling_data = [
        ScalingLogData(
            log_file="a.log",
            label="a.log",
            rank_count=2,
            timer_totals={
                "timer_a": 10.0,
                "timer_b": 12.0,
                "timer_c": 1000000.0,
            },
            timer_percentages={
                "timer_a": 0.0,
                "timer_b": 0.0,
                "timer_c": 100.0,
            },
            total_time=1000022.0,
            per_rank_totals={
                0: {"timer_a": 10.0, "timer_b": 12.0, "timer_c": 1000000.0}
            },
            timer_call_counts={"timer_a": 1, "timer_b": 1, "timer_c": 1},
            total_call_count=3,
            per_rank_call_counts={
                0: {"timer_a": 1, "timer_b": 1, "timer_c": 1}
            },
            emitted_rank_count=1,
        ),
        ScalingLogData(
            log_file="b.log",
            label="b.log",
            rank_count=4,
            timer_totals={
                "timer_a": 10.0,
                "timer_b": 12.0,
                "timer_c": 1000000.0,
            },
            timer_percentages={
                "timer_a": 0.0,
                "timer_b": 0.0,
                "timer_c": 100.0,
            },
            total_time=1000022.0,
            per_rank_totals={
                0: {"timer_a": 10.0, "timer_b": 12.0, "timer_c": 1000000.0}
            },
            timer_call_counts={"timer_a": 1, "timer_b": 1, "timer_c": 1},
            total_call_count=3,
            per_rank_call_counts={
                0: {"timer_a": 1, "timer_b": 1, "timer_c": 1}
            },
            emitted_rank_count=1,
        ),
    ]

    assert not _plot_would_use_log_scale(
        scaling_data,
        ["timer_a", "timer_b"],
        run_statistic="mean",
        rank_source="all",
    )


def test_build_scaling_analysis_avoids_duplicate_weakest_entries():
    """Small timer sets should not repeat the same timers in both summaries."""
    scaling_data = [
        ScalingLogData(
            log_file="a.log",
            label="a.log",
            rank_count=2,
            timer_totals={"timer_a": 20.0, "timer_b": 10.0},
            timer_percentages={"timer_a": 66.7, "timer_b": 33.3},
            total_time=30.0,
            per_rank_totals={0: {"timer_a": 20.0, "timer_b": 10.0}},
            timer_call_counts={"timer_a": 2, "timer_b": 2},
            total_call_count=4,
            per_rank_call_counts={0: {"timer_a": 2, "timer_b": 2}},
            emitted_rank_count=1,
        ),
        ScalingLogData(
            log_file="b.log",
            label="b.log",
            rank_count=4,
            timer_totals={"timer_a": 10.0, "timer_b": 12.0},
            timer_percentages={"timer_a": 45.5, "timer_b": 54.5},
            total_time=22.0,
            per_rank_totals={0: {"timer_a": 10.0, "timer_b": 12.0}},
            timer_call_counts={"timer_a": 2, "timer_b": 2},
            total_call_count=4,
            per_rank_call_counts={0: {"timer_a": 2, "timer_b": 2}},
            emitted_rank_count=1,
        ),
    ]
    timer_db = {
        "timer_a": TimerDef(
            timer_id="timer_a",
            function="engine_a",
            log_pattern="pattern",
            start_line=1,
            end_line=2,
            label_text="took %.3f %s.",
            timer_type="timer",
        ),
        "timer_b": TimerDef(
            timer_id="timer_b",
            function="engine_b",
            log_pattern="pattern",
            start_line=1,
            end_line=2,
            label_text="took %.3f %s.",
            timer_type="timer",
        ),
    }

    analysis = _build_scaling_analysis(
        scaling_data,
        ["timer_a", "timer_b"],
        timer_db,
        min_percent_threshold=10.0,
        used_log_scale=False,
        run_statistic="sum",
        rank_source="root",
    )

    assert "- Strongest improving timers:" in analysis
    assert "- Weakest or regressing timers:" not in analysis
