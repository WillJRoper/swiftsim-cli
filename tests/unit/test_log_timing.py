"""Tests for the log timing analysis module."""

from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pytest

from swiftsim_cli.modes.analyse.log_timing import (
    _hierarchy_corrected_total,
    _hierarchy_direct_accounted_time,
    _print_hierarchical_analysis,
    _print_overall_summary,
    add_log_arguments,
    analyse_swift_log_timings,
    build_function_hierarchy,
    build_stats,
    classify_timers_by_max_time,
    create_time_series_plot,
    display_name,
    get_nested_timers_for_function,
    load_timer_nesting,
    run_swift_log_timing,
)


class TestLogTimingCLI:
    """Tests for CLI argument setup."""

    def test_add_log_arguments(self):
        """Test that log arguments are added correctly."""
        subparsers = Mock()
        log_parser = Mock()
        subparsers.add_parser.return_value = log_parser

        add_log_arguments(subparsers)

        # Verify add_parser was called
        subparsers.add_parser.assert_called_once()
        call_args = subparsers.add_parser.call_args
        assert call_args[0][0] == "log"
        assert "help" in call_args[1]

        # Verify arguments were added
        assert log_parser.add_argument.call_count >= 5


class TestBuildStats:
    """Tests for the build_stats function."""

    def test_build_stats_basic(self):
        """Test build_stats with basic values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = build_stats(values)

        assert stats["call_count"] == 5
        assert stats["mean_time"] == pytest.approx(3.0)
        assert stats["median_time"] == pytest.approx(3.0)
        assert stats["min_time"] == pytest.approx(1.0)
        assert stats["max_time"] == pytest.approx(5.0)
        assert stats["std_time"] > 0
        assert stats["total_time"] == pytest.approx(15.0)

    def test_build_stats_single_value(self):
        """Test build_stats with a single value."""
        values = [42.0]
        stats = build_stats(values)

        assert stats["call_count"] == 1
        assert stats["mean_time"] == pytest.approx(42.0)
        assert stats["median_time"] == pytest.approx(42.0)
        assert stats["min_time"] == pytest.approx(42.0)
        assert stats["max_time"] == pytest.approx(42.0)
        assert stats["std_time"] == pytest.approx(0.0)
        assert stats["total_time"] == pytest.approx(42.0)

    def test_build_stats_empty(self):
        """Test build_stats with empty list."""
        values = []
        stats = build_stats(values)

        assert stats["call_count"] == 0
        assert stats["mean_time"] == pytest.approx(0.0)
        assert stats["median_time"] == pytest.approx(0.0)
        assert stats["min_time"] == pytest.approx(0.0)
        assert stats["max_time"] == pytest.approx(0.0)
        assert stats["std_time"] == pytest.approx(0.0)
        assert stats["total_time"] == pytest.approx(0.0)

    def test_build_stats_with_zeros(self):
        """Test build_stats with values including zeros."""
        values = [0.0, 0.0, 1.0, 2.0]
        stats = build_stats(values)

        assert stats["call_count"] == 4
        assert stats["mean_time"] == pytest.approx(0.75)
        assert stats["total_time"] == pytest.approx(3.0)


class TestDisplayName:
    """Tests for the display_name function."""

    def test_display_name_with_db_entry(self):
        """Test display_name when timer is in database."""
        # Create a mock timer definition object
        mock_timer_def = Mock()
        mock_timer_def.function = "my_function"

        timer_db = {"timer_foo": mock_timer_def}

        name = display_name("timer_foo", timer_db)
        assert "my_function" in name
        assert "timer_foo" in name

    def test_display_name_with_runtime_variant(self):
        """Runtime-derived timer variants should keep the variant label."""
        mock_timer_def = Mock()
        mock_timer_def.function = "engine_launch"

        timer_db = {"engine_launch.c:50|(tasks)": mock_timer_def}

        name = display_name("engine_launch.c:50|(tasks)", timer_db)
        assert "engine_launch" in name
        assert "(tasks)" in name

    def test_display_name_synthetic(self):
        """Test display_name with synthetic timer."""
        timer_db = {}
        name = display_name("SYNTHETIC:my_operation", timer_db)
        assert "my_operation" in name
        assert "SYNTHETIC" in name

    def test_display_name_raises_keyerror_for_missing(self):
        """Test display_name raises KeyError for missing timer."""
        timer_db = {}
        with pytest.raises(KeyError):
            display_name("unknown_timer", timer_db)


class TestClassifyTimersByMaxTime:
    """Tests for the classify_timers_by_max_time function."""

    def test_classify_timers_basic(self):
        """Test timer classification with basic data."""
        # Create mock timer instances
        mock_inst_a = Mock()
        mock_inst_a.timer_id = "timer_a"
        mock_inst_a.time_ms = 10.0

        mock_inst_b = Mock()
        mock_inst_b.timer_id = "timer_b"
        mock_inst_b.time_ms = 5.0

        instances_by_step = {
            "step1": [mock_inst_a, mock_inst_b],
        }

        # Create mock timer definitions
        mock_timer_def_a = Mock()
        mock_timer_def_a.function = "function_1"

        mock_timer_def_b = Mock()
        mock_timer_def_b.function = "function_2"

        timer_db = {
            "timer_a": mock_timer_def_a,
            "timer_b": mock_timer_def_b,
        }

        nesting_db = {}

        result = classify_timers_by_max_time(
            instances_by_step, timer_db, nesting_db
        )

        # Should return a set of function timer IDs
        assert isinstance(result, set)
        assert "timer_a" in result
        assert "timer_b" in result

    def test_classify_timers_empty(self):
        """Test timer classification with empty data."""
        instances_by_step = {}
        timer_db = {}
        nesting_db = {}

        result = classify_timers_by_max_time(
            instances_by_step, timer_db, nesting_db
        )

        # Should return an empty set
        assert isinstance(result, set)
        assert len(result) == 0

    def test_classify_timers_same_function(self):
        """Test timer classification with multiple timers per function."""
        # Create mock instances
        mock_inst1 = Mock()
        mock_inst1.timer_id = "timer_main"
        mock_inst1.time_ms = 100.0

        mock_inst2 = Mock()
        mock_inst2.timer_id = "timer_alt"
        mock_inst2.time_ms = 50.0

        instances_by_step = {
            "step1": [mock_inst1, mock_inst2],
        }

        # Both timers belong to same function
        mock_timer_def1 = Mock()
        mock_timer_def1.function = "shared_function"

        mock_timer_def2 = Mock()
        mock_timer_def2.function = "shared_function"

        timer_db = {
            "timer_main": mock_timer_def1,
            "timer_alt": mock_timer_def2,
        }

        nesting_db = {}

        result = classify_timers_by_max_time(
            instances_by_step, timer_db, nesting_db
        )

        # Should select the timer with max time as function timer
        assert isinstance(result, set)
        assert "timer_main" in result
        # timer_alt should not be selected as function timer (lower time)
        assert "timer_alt" not in result


class TestLogTimingWithRealData:
    """Tests using real log file data."""

    @pytest.fixture
    def test_log_path(self):
        """Get path to test log file."""
        # Assuming the test is run from the project root
        log_path = Path(__file__).parent.parent / "data" / "test_log.txt"
        if not log_path.exists():
            pytest.skip(f"Test log file not found at {log_path}")
        return log_path

    def test_run_swift_log_timing_file_exists(self, test_log_path):
        """Test that test log file exists."""
        # Just verify the test data exists
        assert test_log_path.exists()
        assert test_log_path.is_file()

    def test_build_stats_with_mock_timer_data(self):
        """Test build_stats with mock timer timing values."""
        # Simulate timing values from a log (in milliseconds)
        timing_values = [125.5, 130.2, 128.9, 131.1, 127.8]

        stats = build_stats(timing_values)

        assert stats["call_count"] == 5
        assert stats["mean_time"] > 125.0
        assert stats["mean_time"] < 135.0
        assert stats["total_time"] > 600.0


class TestLoadTimerNesting:
    """Tests for load_timer_nesting function."""

    @patch("swiftsim_cli.modes.analyse.log_timing.open", create=True)
    @patch("swiftsim_cli.modes.analyse.log_timing.YAML")
    @patch(
        "swiftsim_cli.modes.analyse.log_timing.generate_timer_nesting_database"
    )
    @patch("swiftsim_cli.modes.analyse.log_timing.load_swift_profile")
    @patch("swiftsim_cli.modes.analyse.log_timing.Path")
    def test_load_timer_nesting_generates_if_missing(
        self,
        mock_path_class,
        mock_load_profile,
        mock_gen_db,
        mock_yaml,
        mock_open,
    ):
        """Test that nesting DB is generated if file doesn't exist."""
        # Mock Path.home()
        mock_home = Mock()
        mock_home_path = Mock()
        mock_home_path.__truediv__ = lambda self, x: mock_home_path  # Chain
        mock_home.return_value = mock_home_path
        mock_path_class.home = mock_home

        # Mock the nesting file not existing
        mock_nesting_file = Mock()
        mock_nesting_file.exists.return_value = False
        mock_nesting_file.parent = Mock()
        mock_home_path.exists.return_value = False

        # Mock profile
        mock_profile = Mock()
        mock_profile.get.return_value = "/fake/swift"
        mock_load_profile.return_value = mock_profile

        # Mock the generation to return proper structure
        mock_gen_db.return_value = {"nesting": {"func1": {}}}

        # Mock YAML writer
        mock_yaml_instance = Mock()
        mock_yaml.return_value = mock_yaml_instance

        # Call with auto_generate and force_regenerate to trigger
        load_timer_nesting(auto_generate=True, force_regenerate=True)

        # Should have tried to generate
        assert mock_gen_db.called

    @patch("swiftsim_cli.modes.analyse.log_timing.Path")
    def test_load_timer_nesting_returns_empty_if_no_auto_gen(
        self, mock_path_class
    ):
        """Test empty dict returned if file missing and auto_generate=False."""
        # Mock Path.home()
        mock_home = Mock()
        mock_home_path = Mock()
        mock_home_path.__truediv__ = lambda self, x: mock_home_path  # Chain
        mock_home.return_value = mock_home_path
        mock_path_class.home = mock_home

        # Mock the nesting file not existing
        mock_home_path.exists.return_value = False

        # Call with auto_generate=False
        result = load_timer_nesting(
            auto_generate=False, force_regenerate=False
        )

        # Should return empty dict
        assert result == {}


class TestGetNestedTimersForFunction:
    """Tests for get_nested_timers_for_function."""

    def test_get_nested_timers_basic(self):
        """Test getting nested timers for a function."""
        # Create mock stats for timers
        all_stats_dict = {
            "timer1": {"total_time": 100.0},
            "timer2": {"total_time": 50.0},
        }

        # Create mock timer database
        mock_timer1 = Mock()
        mock_timer1.function = "parent_func"
        mock_timer1.timer_type = "function"

        mock_timer2 = Mock()
        mock_timer2.function = "child_func"
        mock_timer2.timer_type = "function"

        timer_db = {
            "timer1": mock_timer1,
            "timer2": mock_timer2,
        }

        # Create nesting database
        nesting_db = {
            "parent_func": {
                "nested_functions": ["child_func"],
            },
        }

        result = get_nested_timers_for_function(
            "parent_func", all_stats_dict, timer_db, nesting_db
        )

        # Should return list of timer tuples
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_get_nested_timers_function_not_in_db(self):
        """Test getting nested timers when function not in database."""
        all_stats_dict = {}
        timer_db = {}
        nesting_db = {}

        result = get_nested_timers_for_function(
            "unknown_func", all_stats_dict, timer_db, nesting_db
        )

        # Should return empty list
        assert result == []

    def test_get_nested_timers_prevents_cycles(self):
        """Test that the function prevents infinite recursion."""
        all_stats_dict = {}
        timer_db = {}

        # Create circular dependency
        nesting_db = {
            "func_a": {
                "nested_functions": ["func_b"],
            },
            "func_b": {
                "nested_functions": ["func_a"],  # Circular!
            },
        }

        # Should not crash due to infinite recursion
        result = get_nested_timers_for_function(
            "func_a", all_stats_dict, timer_db, nesting_db
        )

        # Should return empty list (no timers defined)
        assert isinstance(result, list)


class TestBuildFunctionHierarchy:
    """Tests for build_function_hierarchy."""

    def test_build_function_hierarchy_basic(self):
        """Test building function hierarchy."""
        # Create mock stats dict
        all_stats_dict = {
            "timer_0": {"total_time": 100.0, "call_count": 5},
            "timer_1": {"total_time": 50.0, "call_count": 3},
        }

        # Create timer database
        mock_timer_0 = Mock()
        mock_timer_0.function = "func_0"
        mock_timer_0.timer_type = "function"

        mock_timer_1 = Mock()
        mock_timer_1.function = "func_1"
        mock_timer_1.timer_type = "function"

        timer_db = {
            "timer_0": mock_timer_0,
            "timer_1": mock_timer_1,
        }

        # Simple nesting: func_0 calls func_1
        nesting_db = {
            "func_0": {
                "nested_functions": ["func_1"],
            },
            "func_1": {
                "nested_functions": [],
            },
        }

        result = build_function_hierarchy(
            "func_0", all_stats_dict, timer_db, nesting_db
        )

        # Should return hierarchy data
        assert result is not None
        assert isinstance(result, dict)
        assert "function" in result
        assert "operations" in result
        assert "nested_functions" in result

    def test_build_function_hierarchy_no_instances(self):
        """Test building hierarchy when function has no timer instances."""
        all_stats_dict = {}
        timer_db = {}
        nesting_db = {}

        result = build_function_hierarchy(
            "nonexistent_func", all_stats_dict, timer_db, nesting_db
        )

        # Should return dict structure (not None)
        assert result is not None
        assert isinstance(result, dict)
        assert result["function"] is None
        assert result["operations"] == []
        assert result["nested_functions"] == {}

    def test_build_function_hierarchy_prevents_cycles(self):
        """Test that hierarchy building prevents infinite recursion."""
        all_stats_dict = {}
        timer_db = {}

        # Create circular dependency
        nesting_db = {
            "func_a": {
                "nested_functions": ["func_b"],
            },
            "func_b": {
                "nested_functions": ["func_a"],  # Circular!
            },
        }

        # Should not crash due to infinite recursion
        result = build_function_hierarchy(
            "func_a", all_stats_dict, timer_db, nesting_db
        )

        # Should return structure
        assert isinstance(result, dict)

    def test_build_function_hierarchy_allows_repeated_function_in_branches(
        self,
    ):
        """Repeated callees should still appear in separate branches."""
        all_stats_dict = {
            "timer_root": {"total_time": 100.0, "call_count": 1},
            "timer_left": {"total_time": 40.0, "call_count": 1},
            "timer_right": {"total_time": 30.0, "call_count": 1},
            "timer_leaf": {"total_time": 20.0, "call_count": 2},
        }
        timer_db = {
            "timer_root": Mock(function="root", timer_type="function"),
            "timer_left": Mock(function="left", timer_type="function"),
            "timer_right": Mock(function="right", timer_type="function"),
            "timer_leaf": Mock(function="leaf", timer_type="function"),
        }
        nesting_db = {
            "root": {"nested_functions": ["left", "right"]},
            "left": {"nested_functions": ["leaf"]},
            "right": {"nested_functions": ["leaf"]},
            "leaf": {"nested_functions": []},
        }

        result = build_function_hierarchy(
            "root", all_stats_dict, timer_db, nesting_db
        )

        assert "leaf" in result["nested_functions"]["left"]["nested_functions"]
        assert (
            "leaf" in result["nested_functions"]["right"]["nested_functions"]
        )

    def test_hierarchy_corrected_total_uses_recursive_child_totals(self):
        """Recursive child totals should be used for corrected time."""
        hierarchy = {
            "function": (
                "timer_parent",
                {"total_time": 50.0, "call_count": 2},
            ),
            "operations": [
                ("op_parent", {"total_time": 10.0, "call_count": 2})
            ],
            "nested_functions": {
                "child": {
                    "function": (
                        "timer_child",
                        {"total_time": 20.0, "call_count": 2},
                    ),
                    "operations": [
                        ("op_child", {"total_time": 40.0, "call_count": 2})
                    ],
                    "nested_functions": {},
                }
            },
        }

        assert (
            _hierarchy_corrected_total(hierarchy["nested_functions"]["child"])
            == 40.0
        )
        assert _hierarchy_direct_accounted_time(hierarchy) == 50.0
        assert _hierarchy_corrected_total(hierarchy) == 50.0

    def test_hierarchy_unaccounted_shrinks_with_corrected_child_totals(self):
        """Direct accounting should use corrected child totals."""
        hierarchy = {
            "function": (
                "timer_prepare",
                {"total_time": 100.0, "call_count": 1},
            ),
            "operations": [],
            "nested_functions": {
                "engine_rebuild": {
                    "function": (
                        "timer_rebuild",
                        {"total_time": 10.0, "call_count": 1},
                    ),
                    "operations": [
                        ("op_rebuild", {"total_time": 60.0, "call_count": 1})
                    ],
                    "nested_functions": {},
                }
            },
        }

        assert (
            _hierarchy_corrected_total(
                hierarchy["nested_functions"]["engine_rebuild"]
            )
            == 60.0
        )
        assert _hierarchy_direct_accounted_time(hierarchy) == 60.0
        assert _hierarchy_corrected_total(hierarchy) == 100.0

    def test_hierarchy_corrected_total_still_handles_cycles(self):
        """Cycle prevention should still hold with path-based recursion."""
        all_stats_dict = {
            "timer_a": {"total_time": 10.0, "call_count": 1},
            "timer_b": {"total_time": 5.0, "call_count": 1},
        }
        timer_db = {
            "timer_a": Mock(function="func_a", timer_type="function"),
            "timer_b": Mock(function="func_b", timer_type="function"),
        }
        nesting_db = {
            "func_a": {"nested_functions": ["func_b"]},
            "func_b": {"nested_functions": ["func_a"]},
        }

        result = build_function_hierarchy(
            "func_a", all_stats_dict, timer_db, nesting_db
        )

        assert _hierarchy_corrected_total(result) == 10.0

    def test_build_function_hierarchy_skips_phantom_parent_without_timer(self):
        """Descendants alone should not create a displayed parent node."""
        all_stats_dict = {
            "timer_parent": {"total_time": 100.0, "call_count": 1},
            "timer_child": {"total_time": 20.0, "call_count": 1},
        }
        timer_db = {
            "timer_parent": Mock(function="parent", timer_type="function"),
            "timer_child": Mock(function="child", timer_type="function"),
        }
        nesting_db = {
            "parent": {"nested_functions": ["missing_mid"]},
            "missing_mid": {"nested_functions": ["child"]},
            "child": {"nested_functions": []},
        }

        result = build_function_hierarchy(
            "parent", all_stats_dict, timer_db, nesting_db
        )

        assert "missing_mid" not in result["nested_functions"]


class TestRunSwiftLogTiming:
    """Tests for run_swift_log_timing."""

    @patch("swiftsim_cli.modes.analyse.log_timing.analyse_swift_log_timings")
    def test_run_swift_log_timing_calls_analyse(self, mock_analyse, tmp_path):
        """Test that run_swift_log_timing calls analyse_swift_log_timings."""
        # Create a fake log file
        log_file = tmp_path / "test.log"
        log_file.write_text("test log content")

        # Create args
        args = Mock()
        args.log_file = log_file
        args.output_path = tmp_path
        args.prefix = "test"
        args.show = False
        args.top_n = 20
        args.hierarchy_functions = None

        # Call the function
        run_swift_log_timing(args)

        # Verify analyse_swift_log_timings was called
        mock_analyse.assert_called_once()
        call_args = mock_analyse.call_args
        assert str(log_file) in str(call_args)


class TestAnalyseSwiftLogTimingsWithMocks:
    """Tests for analyse_swift_log_timings with comprehensive mocking."""

    @patch("swiftsim_cli.modes.analyse.log_timing.create_output_path")
    @patch("swiftsim_cli.modes.analyse.log_timing.plt")
    @patch("swiftsim_cli.modes.analyse.log_timing.classify_timers_by_max_time")
    @patch("swiftsim_cli.modes.analyse.log_timing.scan_log_instances_by_step")
    @patch("swiftsim_cli.modes.analyse.log_timing.load_timer_nesting")
    @patch("swiftsim_cli.modes.analyse.log_timing.compile_site_patterns")
    @patch("swiftsim_cli.modes.analyse.log_timing.load_timer_db")
    def test_analyse_swift_log_timings_full_flow(
        self,
        mock_load_db,
        mock_compile,
        mock_load_nesting,
        mock_scan_log,
        mock_classify,
        mock_plt,
        mock_create_path,
        tmp_path,
    ):
        """Test full analysis flow with mocked dependencies."""
        # Create a test log file
        log_file = tmp_path / "test.log"
        log_file.write_text("mock log content")

        # Mock output path
        def create_path_side_effect(
            output_path=None,
            prefix=None,
            filename="output.png",
            out_dir=None,
        ):
            base_path = tmp_path
            if out_dir is not None:
                base_path = base_path / out_dir
                base_path.mkdir(parents=True, exist_ok=True)
            return base_path / f"{prefix + '_' if prefix else ''}{filename}"

        mock_create_path.side_effect = create_path_side_effect

        # Mock timer database
        mock_timer_def = Mock()
        mock_timer_def.function = "test_func"
        mock_timer_def.timer_type = "function"
        timer_db = {"timer1": mock_timer_def}
        mock_load_db.return_value = timer_db

        # Mock compiled patterns
        mock_compile.return_value = []

        # Mock nesting database
        nesting_db = {}
        mock_load_nesting.return_value = nesting_db

        # Mock scan results with timer instances
        mock_inst = Mock()
        mock_inst.timer_id = "timer1"
        mock_inst.time_ms = 100.0
        mock_inst.task_count = 1
        instances_by_step = {"step1": [mock_inst]}
        mock_scan_log.return_value = (instances_by_step, {})

        # Mock classification
        mock_classify.return_value = {"timer1"}

        # Call the function
        analyse_swift_log_timings(
            log_file=str(log_file),
            output_path=None,
            prefix="test",
            show_plot=False,
            top_n=10,
            hierarchy_functions=None,
        )

        # Verify key functions were called
        mock_load_db.assert_called_once_with(force_regenerate=True)
        mock_compile.assert_called_once()
        mock_load_nesting.assert_called_once()
        mock_scan_log.assert_called_once()
        mock_classify.assert_called_once()
        report_path = (
            tmp_path / "test_runtime_analysis" / "test_analysis_tables.txt"
        )
        assert report_path.exists()
        assert "TOP FUNCTION TIMERS" in report_path.read_text(encoding="utf-8")

    @patch("swiftsim_cli.modes.analyse.log_timing.create_output_path")
    @patch("swiftsim_cli.modes.analyse.log_timing.plt")
    @patch("swiftsim_cli.modes.analyse.log_timing.classify_timers_by_max_time")
    @patch("swiftsim_cli.modes.analyse.log_timing.scan_log_instances_by_step")
    @patch("swiftsim_cli.modes.analyse.log_timing.load_timer_nesting")
    @patch("swiftsim_cli.modes.analyse.log_timing.compile_site_patterns")
    @patch("swiftsim_cli.modes.analyse.log_timing.load_timer_db")
    def test_analyse_swift_log_timings_with_empty_log(
        self,
        mock_load_db,
        mock_compile,
        mock_load_nesting,
        mock_scan_log,
        mock_classify,
        mock_plt,
        mock_create_path,
        tmp_path,
    ):
        """Test analysis handles empty log gracefully."""
        # Create an empty log file
        log_file = tmp_path / "empty.log"
        log_file.write_text("")

        # Mock output path
        def create_path_side_effect(
            output_path=None,
            prefix=None,
            filename="output.png",
            out_dir=None,
        ):
            base_path = tmp_path
            if out_dir is not None:
                base_path = base_path / out_dir
                base_path.mkdir(parents=True, exist_ok=True)
            return base_path / f"{prefix + '_' if prefix else ''}{filename}"

        mock_create_path.side_effect = create_path_side_effect

        # Mock timer database (empty)
        mock_load_db.return_value = {}

        # Mock compiled patterns (empty)
        mock_compile.return_value = []

        # Mock nesting database (empty)
        mock_load_nesting.return_value = {}

        # Mock scan results (no instances)
        mock_scan_log.return_value = ({}, {})

        # Mock classification (empty set)
        mock_classify.return_value = set()

        # Call the function - should not crash
        analyse_swift_log_timings(
            log_file=str(log_file),
            output_path=None,
            prefix="test",
            show_plot=False,
            top_n=10,
            hierarchy_functions=None,
        )

        # Verify it handled empty data
        mock_load_db.assert_called_once_with(force_regenerate=True)
        mock_scan_log.assert_called_once()

    def test_zoom_space_split_operations_attach_in_hierarchy(self):
        """Zoom split operation timers should attach under space_split."""
        timer_db = {
            "space_split_fn": Mock(
                function="space_split",
                timer_type="function",
                label_text="took %.3f %s.",
            ),
            "space_split_zoom": Mock(
                function="space_split",
                timer_type="operation",
                label_text=(
                    "Zoom cell tree and multipole construction took %.3f %s."
                ),
            ),
            "space_split_bg": Mock(
                function="space_split",
                timer_type="operation",
                label_text=(
                    "Background cell tree and multipole construction took "
                    "%.3f %s."
                ),
            ),
        }
        all_stats = {
            "space_split_fn": {"total_time": 100.0, "call_count": 1},
            "space_split_zoom": {"total_time": 60.0, "call_count": 1},
            "space_split_bg": {"total_time": 40.0, "call_count": 1},
        }
        nesting_db = {
            "space_split": {
                "nested_functions": [],
                "nested_operations": [
                    "Zoom cell tree and multipole construction took %.3f %s.",
                    "Background cell tree and multipole construction took "
                    "%.3f %s.",
                ],
            }
        }

        hierarchy = build_function_hierarchy(
            "space_split", all_stats, timer_db, nesting_db
        )

        operation_ids = [tid for tid, _ in hierarchy["operations"]]
        assert operation_ids == ["space_split_zoom", "space_split_bg"]

    @patch("swiftsim_cli.modes.analyse.log_timing.create_output_path")
    @patch("swiftsim_cli.modes.analyse.log_timing.plt")
    @patch("swiftsim_cli.modes.analyse.log_timing.classify_timers_by_max_time")
    @patch("swiftsim_cli.modes.analyse.log_timing.scan_log_instances_by_step")
    @patch("swiftsim_cli.modes.analyse.log_timing.load_timer_nesting")
    @patch("swiftsim_cli.modes.analyse.log_timing.compile_site_patterns")
    @patch("swiftsim_cli.modes.analyse.log_timing.load_timer_db")
    def test_analysis_refreshes_timer_db_before_nesting_regeneration(
        self,
        mock_load_db,
        mock_compile,
        mock_load_nesting,
        mock_scan_log,
        mock_classify,
        mock_plt,
        mock_create_path,
        tmp_path,
    ):
        """Analysis should refresh timer regex metadata alongside nesting."""
        log_file = tmp_path / "test.log"
        log_file.write_text("mock log content")

        def create_path_side_effect(
            output_path=None,
            prefix=None,
            filename="output.png",
            out_dir=None,
        ):
            base_path = tmp_path
            if out_dir is not None:
                base_path = base_path / out_dir
                base_path.mkdir(parents=True, exist_ok=True)
            return base_path / f"{prefix + '_' if prefix else ''}{filename}"

        mock_create_path.side_effect = create_path_side_effect

        timer_db = {
            "timer1": Mock(function="space_split", timer_type="function")
        }
        mock_load_db.return_value = timer_db
        mock_compile.return_value = []
        mock_load_nesting.return_value = {}
        mock_scan_log.return_value = ({}, {})
        mock_classify.return_value = set()

        analyse_swift_log_timings(
            log_file=str(log_file),
            output_path=None,
            prefix="test",
            show_plot=False,
            top_n=10,
            hierarchy_functions=None,
        )

        mock_load_db.assert_called_once_with(force_regenerate=True)
        assert mock_compile.call_args_list[0].args[0] is timer_db
        mock_load_nesting.assert_called_once_with(
            auto_generate=True, force_regenerate=True
        )


class TestCreateTimeSeriesPlot:
    """Tests for the time series plotting helper."""

    @patch("swiftsim_cli.modes.analyse.log_timing.create_output_path")
    @patch("swiftsim_cli.modes.analyse.log_timing.plt.savefig")
    @patch("swiftsim_cli.modes.analyse.log_timing.plt.close")
    @patch("swiftsim_cli.modes.analyse.log_timing.plt.show")
    def test_create_time_series_plot_uses_stacked_subplots(
        self,
        mock_show,
        mock_close,
        mock_savefig,
        mock_create_output_path,
        tmp_path,
    ):
        """Top functions are rendered as one shared-x subplot per function."""
        mock_create_output_path.return_value = tmp_path / "07_time_series.png"

        timer_db = {
            "timer_a": Mock(function="engine_prepare"),
            "timer_b": Mock(function="engine_rebuild"),
            "timer_c": Mock(function="engine_launch"),
        }
        function_stats = {
            "timer_a": {"total_time": 30.0},
            "timer_b": {"total_time": 20.0},
            "timer_c": {"total_time": 10.0},
        }

        def make_instance(timer_id: str, time_ms: float) -> Mock:
            inst = Mock()
            inst.timer_id = timer_id
            inst.time_ms = time_ms
            return inst

        instances_by_step = {
            1: [
                make_instance("timer_a", 10.0),
                make_instance("timer_b", 7.0),
                make_instance("timer_c", 3.0),
            ],
            2: [
                make_instance("timer_a", 11.0),
                make_instance("timer_b", 6.0),
                make_instance("timer_c", 4.0),
            ],
            3: [
                make_instance("timer_a", 9.0),
                make_instance("timer_b", 7.0),
                make_instance("timer_c", 3.0),
            ],
        }

        output = create_time_series_plot(
            instances_by_step=instances_by_step,
            function_stats=function_stats,
            timer_db=timer_db,
            output_path=None,
            prefix=None,
            show_plot=False,
            out_dir="analysis",
        )

        assert output == tmp_path / "07_time_series.png"
        mock_savefig.assert_called_once()
        mock_show.assert_not_called()
        mock_close.assert_called_once()

        figure = plt.gcf()
        assert len(figure.axes) == 3
        assert figure.get_size_inches()[1] == pytest.approx(19.5)


class TestHierarchicalAnalysisDefaults:
    """Tests for default hierarchical function selection."""

    def test_default_hierarchy_includes_engine_prepare(self, capsys):
        """Default hierarchical output should include engine_prepare."""
        timer_db = {
            "timer_prepare": Mock(
                function="engine_prepare", timer_type="function"
            ),
            "timer_rebuild": Mock(
                function="engine_rebuild", timer_type="function"
            ),
            "timer_space": Mock(
                function="space_rebuild", timer_type="function"
            ),
            "timer_split": Mock(function="space_split", timer_type="function"),
            "timer_tasks": Mock(
                function="engine_maketasks", timer_type="function"
            ),
        }
        all_stats = {
            "timer_prepare": {
                "total_time": 500.0,
                "call_count": 5,
                "mean_time": 100.0,
            },
            "timer_rebuild": {
                "total_time": 400.0,
                "call_count": 4,
                "mean_time": 100.0,
            },
            "timer_space": {
                "total_time": 300.0,
                "call_count": 3,
                "mean_time": 100.0,
            },
            "timer_split": {
                "total_time": 200.0,
                "call_count": 2,
                "mean_time": 100.0,
            },
            "timer_tasks": {
                "total_time": 100.0,
                "call_count": 1,
                "mean_time": 100.0,
            },
        }
        nesting_db = {
            "engine_prepare": {
                "nested_functions": [],
                "nested_operations": [],
            },
            "engine_rebuild": {
                "nested_functions": [],
                "nested_operations": [],
            },
            "space_rebuild": {"nested_functions": [], "nested_operations": []},
            "space_split": {"nested_functions": [], "nested_operations": []},
            "engine_maketasks": {
                "nested_functions": [],
                "nested_operations": [],
            },
        }

        _print_hierarchical_analysis(
            all_stats=all_stats,
            timer_db=timer_db,
            nesting_db=nesting_db,
            hierarchy_functions=None,
            top_n=20,
        )

        output = capsys.readouterr().out
        assert "engine_prepare:" in output
        assert (
            "% (function execution time / percentage of total run time)"
            in output
        )

    def test_hierarchical_analysis_prints_context_note(self, capsys):
        """Hierarchical output should document whole-run attribution."""
        timer_db = {
            "timer_parent": Mock(
                function="engine_prepare",
                timer_type="function",
                label_text="took %.3f %s.",
            )
        }
        all_stats = {"timer_parent": {"total_time": 120.0, "call_count": 1}}
        nesting_db = {
            "engine_prepare": {
                "nested_functions": [],
                "nested_operations": [],
            }
        }

        _print_hierarchical_analysis(
            all_stats=all_stats,
            timer_db=timer_db,
            nesting_db=nesting_db,
            hierarchy_functions=["engine_prepare"],
            top_n=10,
        )

        output = capsys.readouterr().out
        assert "whole-run timer totals" in output

    def test_hierarchical_analysis_title_includes_total_runtime_percentage(
        self, capsys
    ):
        """Hierarchical titles should report percentage of total runtime."""
        timer_db = {
            "timer_parent": Mock(
                function="space_rebuild",
                timer_type="function",
                label_text="took %.3f %s.",
            ),
            "timer_other": Mock(
                function="other_function",
                timer_type="function",
                label_text="took %.3f %s.",
            ),
        }
        all_stats = {
            "timer_parent": {"total_time": 120.0, "call_count": 2},
            "timer_other": {"total_time": 80.0, "call_count": 1},
        }
        nesting_db = {
            "space_rebuild": {
                "nested_functions": [],
                "nested_operations": [],
            }
        }

        _print_hierarchical_analysis(
            all_stats=all_stats,
            timer_db=timer_db,
            nesting_db=nesting_db,
            hierarchy_functions=["space_rebuild"],
            top_n=10,
        )

        output = capsys.readouterr().out
        assert (
            "space_rebuild: 120.0 ms / 60.0% "
            "(function execution time / percentage of total run time)"
            in output
        )

    def test_hierarchical_analysis_prints_vertical_tree_guides(self, capsys):
        """Nested hierarchy rows should keep faint vertical guide lines."""
        timer_db = {
            "parent_timer": Mock(
                function="engine_prepare",
                timer_type="function",
                label_text="took %.3f %s.",
            ),
            "child_timer": Mock(
                function="child_func",
                timer_type="function",
                label_text="took %.3f %s.",
            ),
            "child_op": Mock(
                function="child_func",
                timer_type="operation",
                label_text="child op took %.3f %s.",
            ),
            "sibling_op": Mock(
                function="engine_prepare",
                timer_type="operation",
                label_text="sibling op took %.3f %s.",
            ),
        }
        all_stats = {
            "parent_timer": {"total_time": 100.0, "call_count": 1},
            "child_timer": {"total_time": 60.0, "call_count": 1},
            "child_op": {"total_time": 25.0, "call_count": 1},
            "sibling_op": {"total_time": 20.0, "call_count": 1},
        }
        nesting_db = {
            "engine_prepare": {
                "nested_functions": ["child_func"],
                "nested_operations": ["sibling op took %.3f %s."],
            },
            "child_func": {
                "nested_functions": [],
                "nested_operations": ["child op took %.3f %s."],
            },
        }

        _print_hierarchical_analysis(
            all_stats=all_stats,
            timer_db=timer_db,
            nesting_db=nesting_db,
            hierarchy_functions=["engine_prepare"],
            top_n=10,
        )

        output = capsys.readouterr().out
        assert "├─ child_func (nested function)" in output
        assert "│  └─ child op" in output


class TestOverallSummary:
    """Tests for the bottom-of-report performance summary."""

    def test_overall_summary_includes_step_table_sanity_check(self, capsys):
        """Summary should compare function timers against step totals."""
        _print_overall_summary(
            total_function=70.0,
            total_operation=20.0,
            total_all=90.0,
            function_stats={"func": {"call_count": 2}},
            operation_stats={"op": {"call_count": 3}},
            all_stats={
                "func": {"call_count": 2, "total_time": 70.0},
                "op": {"call_count": 3, "total_time": 20.0},
            },
            sorted_all=[
                ("func", {"total_time": 70.0}),
                ("op", {"total_time": 20.0}),
            ],
            step_totals={0: 60.0, 1: 40.0},
        )

        output = capsys.readouterr().out
        assert "Step-table wallclock total:            100.0 ms" in output
        assert "Function timer coverage of steps:      70.0%" in output
        assert (
            "Time outside function timers:          30.0 ms (30.0%)" in output
        )
