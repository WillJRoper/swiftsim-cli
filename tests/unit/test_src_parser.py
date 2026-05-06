"""Unit tests for the src_parser module."""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from swiftsim_cli.src_parser import (
    TimerDef,
    TimerInstance,
    TimerNestingGenerator,
    TimerSite,
    _build_log_pattern,
    _classify_timer_type,
    _extract_function_name_from_header,
    _printf_to_regex,
    _scan_balanced_call,
    _unescape_minimal,
    compile_site_patterns,
    load_timer_db,
    scan_log_instances_by_step,
)


class TestUtilityFunctions:
    """Test utility functions in src_parser module."""

    def test_unescape_minimal(self):
        """Test minimal string unescaping."""
        assert _unescape_minimal(r"Hello \"World\"") == 'Hello "World"'
        assert (
            _unescape_minimal(r"Path\\to\\file") == "Path\\to\\file"
        )  # \\ -> \
        assert _unescape_minimal(r"Line\nbreak") == "Line\nbreak"
        assert _unescape_minimal(r"Tab\there") == "Tab\there"

    def test_printf_to_regex(self):
        """Test printf format string to regex conversion."""
        label, regex = _printf_to_regex("took %.3f %s")
        assert label == "took %.3f %s"
        assert ".*?" in regex

        label, regex = _printf_to_regex("Operation took %f ms")
        assert label == "Operation took %f ms"
        assert ".*?" in regex

    def test_classify_timer_type(self):
        """Test timer type classification."""
        # All timers should return 'timer' for initial classification
        assert _classify_timer_type("took %.3f %s") == "timer"
        assert _classify_timer_type("Operation took %.3f %s") == "timer"
        assert _classify_timer_type("Complex operation took %f ms") == "timer"

    def test_build_log_pattern(self):
        """Test log pattern building."""
        label, pattern = _build_log_pattern("test_function", "took %.3f %s")

        assert "test_function" in pattern
        assert label == "took %.3f %s"
        # Check for the actual pattern components
        assert "took" in pattern
        assert "test_function" in pattern
        assert "([\\d.]+)" in pattern  # Captures decimal numbers

    def test_build_log_pattern_accepts_variable_width_rank_prefix(self):
        """Generated log regex should accept non-zero-padded ranks."""
        _, pattern = _build_log_pattern("space_split", "took %.3f %s")

        assert re.compile(pattern).search("[7] 12.3 space_split: took 4.5 ms.")

    def test_extract_function_name_from_header(self):
        """Test function name extraction from C headers."""
        # Valid function headers
        assert (
            _extract_function_name_from_header("void test_function(int x) {")
            == "test_function"
        )
        assert (
            _extract_function_name_from_header(
                "static int calculate_sum(float a, float b) {"
            )
            == "calculate_sum"
        )
        assert (
            _extract_function_name_from_header(
                "INLINE double space_split(struct cell *c) {"
            )
            == "space_split"
        )

        # Should filter out keywords
        assert _extract_function_name_from_header("if (condition) {") is None
        assert (
            _extract_function_name_from_header(
                "for (int i = 0; i < 10; i++) {"
            )
            is None
        )

        # No parentheses
        assert _extract_function_name_from_header("int variable = 5;") is None

    def test_scan_balanced_call(self):
        """Test balanced parentheses scanning."""
        lines = [
            'message("Simple message")',
            'message("Multi-line"',
            '        " message")',
            "other_code();",
        ]

        # Simple case
        result, end_line = _scan_balanced_call(lines, 0, 7)  # After "message"
        assert '"Simple message"' in result
        assert end_line == 0

        # Multi-line case
        result, end_line = _scan_balanced_call(lines, 1, 7)  # After "message"
        assert '"Multi-line"' in result
        assert '" message"' in result
        assert end_line == 2


class TestTimerDataStructures:
    """Test timer data structure classes."""

    def test_timer_site_creation(self):
        """Test TimerSite creation."""
        site = TimerSite(
            timer_id="test.c:100",
            file="test.c",
            function="test_func",
            start_line=95,
            end_line=100,
            tic_var="tic",
            label_text="Operation took time",
            log_pattern="test pattern",
            timer_type="operation",
        )

        assert site.timer_id == "test.c:100"
        assert site.function == "test_func"
        assert site.timer_type == "operation"

    def test_timer_def_creation(self):
        """Test TimerDef creation."""
        timer_def = TimerDef(
            timer_id="test.c:100",
            function="test_func",
            log_pattern="test pattern",
            start_line=95,
            end_line=100,
            label_text="Operation took time",
            timer_type="operation",
        )

        assert timer_def.timer_id == "test.c:100"
        assert timer_def.function == "test_func"
        assert timer_def.timer_type == "operation"

    def test_timer_instance_creation(self):
        """Test TimerInstance creation."""
        instance = TimerInstance(
            timer_id="test.c:100",
            function="test_func",
            step=5,
            time_ms=123.45,
            line_index=200,
            timer_type="operation",
        )

        assert instance.timer_id == "test.c:100"
        assert instance.time_ms == 123.45
        assert instance.step == 5


class TestPatternCompilation:
    """Test pattern compilation functions."""

    def test_compile_site_patterns(self, sample_timer_db):
        """Test timer pattern compilation."""
        compiled = compile_site_patterns(sample_timer_db)

        assert len(compiled) == len(sample_timer_db)

        for timer_id, pattern in compiled:
            assert timer_id in sample_timer_db
            assert hasattr(pattern, "search")  # Should be compiled regex

    def test_compile_site_patterns_bad_regex(self):
        """Test pattern compilation with bad regex raises ValueError."""
        bad_timer_db = {
            "bad.c:1": TimerDef(
                timer_id="bad.c:1",
                function="bad_func",
                log_pattern="[unclosed bracket",  # Invalid regex
                start_line=1,
                end_line=1,
                label_text="bad",
                timer_type="operation",
            )
        }

        # Should raise ValueError for corrupted timer database
        with pytest.raises(ValueError) as exc_info:
            compile_site_patterns(bad_timer_db)

        # Verify error message contains helpful information
        error_msg = str(exc_info.value)
        assert "bad.c:1" in error_msg  # Should include timer ID
        assert "[unclosed bracket" in error_msg  # Should include pattern
        assert "corrupted timer database" in error_msg.lower()
        assert "regenerating" in error_msg.lower()  # Should suggest fix


class TestLogScanning:
    """Test log scanning functionality."""

    def test_scan_log_instances_by_step(
        self, temp_dir, sample_timer_db, sample_log_content
    ):
        """Test log scanning for timer instances."""
        # Create a test log file
        log_file = temp_dir / "test.log"
        log_file.write_text(sample_log_content)

        # Compile patterns
        compiled = compile_site_patterns(sample_timer_db)

        # Scan the log
        instances_by_step, step_lines = scan_log_instances_by_step(
            str(log_file), compiled, sample_timer_db
        )

        # Should find instances
        assert len(instances_by_step) > 0

        # Check that instances are properly created
        for step, instances in instances_by_step.items():
            for instance in instances:
                assert isinstance(instance, TimerInstance)
                assert instance.timer_id in sample_timer_db
                assert instance.time_ms > 0

    def test_scan_log_instances_empty_file(self, temp_dir, sample_timer_db):
        """Test log scanning with empty file."""
        log_file = temp_dir / "empty.log"
        log_file.write_text("")

        compiled = compile_site_patterns(sample_timer_db)

        instances_by_step, step_lines = scan_log_instances_by_step(
            str(log_file), compiled, sample_timer_db
        )

        assert len(instances_by_step) == 0
        assert len(step_lines) == 0

    @patch("builtins.open", side_effect=OSError("File not found"))
    def test_scan_log_instances_file_error(self, mock_open, sample_timer_db):
        """Test log scanning with file error."""
        compiled = compile_site_patterns(sample_timer_db)

        # Should handle file errors gracefully
        with pytest.raises(OSError):
            scan_log_instances_by_step(
                "nonexistent.log", compiled, sample_timer_db
            )

    def test_scan_log_instances_does_not_treat_elapsed_lines_as_steps(
        self, temp_dir
    ):
        """Leading elapsed-time values should not be mistaken for steps."""
        timer_db = {
            "test.c:1": TimerDef(
                timer_id="test.c:1",
                function="space_split",
                log_pattern=r"^space_split:\s+took\s+([\d.]+)\s+ms\.\s*$",
                start_line=1,
                end_line=1,
                label_text="took %.3f %s.",
                timer_type="operation",
            )
        }
        compiled = compile_site_patterns(timer_db)
        log_file = temp_dir / "elapsed_not_step.log"
        log_file.write_text(
            "12.345 unrelated header line\nspace_split: took 4.0 ms.\n",
            encoding="utf-8",
        )

        instances_by_step, step_lines = scan_log_instances_by_step(
            str(log_file), compiled, timer_db
        )

        assert step_lines == []
        assert None in instances_by_step


class TestTimerNestingGeneration:
    """Test timer nesting generation functionality."""

    def test_tree_sitter_availability(self):
        """Test tree-sitter availability check."""
        # Test that tree-sitter can be imported
        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_c  # noqa: F401

            tree_sitter_available = True
        except ImportError:
            tree_sitter_available = False

        assert isinstance(tree_sitter_available, bool)

    @pytest.mark.skipif(
        not pytest.importorskip(
            "tree_sitter", reason="tree-sitter not available"
        ),
        reason="Requires tree-sitter",
    )
    def test_timer_nesting_generator_init(self):
        """Test TimerNestingGenerator initialization."""
        timer_data = [
            {
                "timer_id": "test.c:1",
                "function": "test_func",
                "timer_type": "function",
                "file": "test.c",
            }
        ]

        generator = TimerNestingGenerator("/fake/src", timer_data)
        assert generator.src_dir == Path("/fake/src")
        assert len(generator.timer_data) == 1

    @pytest.mark.skipif(
        not pytest.importorskip(
            "tree_sitter", reason="tree-sitter not available"
        ),
        reason="Requires tree-sitter",
    )
    def test_extract_function_calls_handles_parenthesized_identifier(self):
        """Parenthesized direct callees should still be recognized."""
        import tree_sitter_c
        from tree_sitter import Language, Parser

        timer_data = [
            {
                "timer_id": "test.c:1",
                "function": "caller",
                "timer_type": "function",
                "file": "test.c",
            }
        ]
        generator = TimerNestingGenerator("/fake/src", timer_data)

        parser = Parser(Language(tree_sitter_c.language()))
        source_code = b"void caller(void) { (child)(); other(); }"
        tree = parser.parse(source_code)
        function_node = tree.root_node.children[0]

        calls = generator._extract_function_calls(function_node, source_code)

        assert "child" in calls
        assert "other" in calls


class TestTimerDatabase:
    """Test timer database loading functionality."""

    def test_load_timer_db_force_regenerate(self, temp_dir):
        """Forced reload should regenerate timers.yaml from source."""
        from unittest.mock import Mock, patch

        import yaml

        timer_file = temp_dir / "timers.yaml"
        timer_file.write_text("timers: []\n")

        regenerated = {
            "timers": [
                {
                    "timer_id": "zoom.c:10",
                    "function": "space_split",
                    "log_pattern": r"^.*space_split:.*$",
                    "start_line": 1,
                    "end_line": 10,
                    "label_text": (
                        "Zoom cell tree and multipole construction took "
                        "%.3f %s."
                    ),
                    "timer_type": "operation",
                }
            ]
        }

        def fake_parse_src_timers(_src_dir):
            with open(timer_file, "w", encoding="utf-8") as handle:
                yaml.dump(regenerated, handle)
            return regenerated

        with patch("swiftsim_cli.src_parser.TIMER_FILE", timer_file), patch(
            "swiftsim_cli.src_parser.load_swift_profile",
            return_value=Mock(swiftsim_dir="/fake/swift"),
        ), patch(
            "swiftsim_cli.src_parser.parse_src_timers",
            side_effect=fake_parse_src_timers,
        ) as mock_parse:
            timer_db = load_timer_db(force_regenerate=True)

        mock_parse.assert_called_once_with("/fake/swift")
        assert "zoom.c:10" in timer_db

    def test_load_timer_db_force_regenerate_requires_swiftsim_dir(
        self, temp_dir
    ):
        """Forced regeneration should fail clearly without a SWIFT path."""
        from unittest.mock import Mock, patch

        timer_file = temp_dir / "missing.yaml"

        with patch("swiftsim_cli.src_parser.TIMER_FILE", timer_file), patch(
            "swiftsim_cli.src_parser.load_swift_profile",
            return_value=Mock(swiftsim_dir=None),
        ):
            with pytest.raises(ValueError, match="swiftsim_dir"):
                load_timer_db(force_regenerate=True)

    def test_load_timer_db_with_existing_file(self, temp_dir, sample_timer_db):
        """Test loading timer database when file exists."""
        # Create a mock timer file
        from unittest.mock import patch

        import yaml

        timer_file = temp_dir / "timers.yaml"

        # Convert TimerDef objects to list of dicts for YAML
        timers_list = []
        for timer_id, timer_def in sample_timer_db.items():
            timers_list.append(
                {
                    "timer_id": timer_id,
                    "function": timer_def.function,
                    "log_pattern": timer_def.log_pattern,
                    "start_line": timer_def.start_line,
                    "end_line": timer_def.end_line,
                    "label_text": timer_def.label_text,
                    "timer_type": timer_def.timer_type,
                }
            )

        yaml_data = {"timers": timers_list}

        with open(timer_file, "w") as f:
            yaml.dump(yaml_data, f)

        # Mock TIMER_FILE to point to our test file
        with patch("swiftsim_cli.src_parser.TIMER_FILE", timer_file):
            timer_db = load_timer_db()

            # Should load timers from file
            assert len(timer_db) == len(sample_timer_db)
            for timer_id in sample_timer_db:
                assert timer_id in timer_db
                assert isinstance(timer_db[timer_id], TimerDef)

    def test_load_timer_db_empty_file(self, temp_dir):
        """Test loading timer database from empty file."""
        from unittest.mock import patch

        timer_file = temp_dir / "empty_timers.yaml"
        timer_file.write_text("")

        with patch("swiftsim_cli.src_parser.TIMER_FILE", timer_file):
            timer_db = load_timer_db()

            # Empty file should return empty dict
            assert timer_db == {}

    def test_load_timer_db_file_without_timers_key(self, temp_dir):
        """Test loading timer database from file without 'timers' key."""
        from unittest.mock import patch

        import yaml

        timer_file = temp_dir / "no_timers_key.yaml"

        # YAML file with no 'timers' key
        yaml_data = {"some_other_key": "value"}

        with open(timer_file, "w") as f:
            yaml.dump(yaml_data, f)

        with patch("swiftsim_cli.src_parser.TIMER_FILE", timer_file):
            timer_db = load_timer_db()

            # Should return empty dict when 'timers' key is missing
            assert timer_db == {}
