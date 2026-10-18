"""Unit tests for utility functions."""

from subprocess import CalledProcessError, TimeoutExpired
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pytest

from swiftsim_cli.utilities import (
    create_ascii_table,
    create_output_path,
    run_command_in_dir,
    slugify,
)
from swiftsim_cli.utilities.plotting import save_figure, should_use_log_scale


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_ascii_table_basic(self):
        """Test basic ASCII table creation."""
        headers = ["Name", "Age", "City"]
        rows = [
            ["Alice", "25", "New York"],
            ["Bob", "30", "London"],
            ["Charlie", "35", "Tokyo"],
        ]

        table = create_ascii_table(headers, rows)

        assert isinstance(table, str)
        assert "Name" in table
        assert "Age" in table
        assert "City" in table
        assert "Alice" in table
        assert "Bob" in table
        assert "Charlie" in table
        assert "25" in table
        assert "30" in table
        assert "35" in table

    def test_create_ascii_table_with_title(self):
        """Test ASCII table creation with title."""
        headers = ["Item", "Count"]
        rows = [["Apples", "5"], ["Bananas", "3"]]
        title = "Fruit Inventory"

        table = create_ascii_table(headers, rows, title)

        assert title in table
        assert "Item" in table
        assert "Count" in table

    def test_create_ascii_table_empty_rows(self):
        """Test ASCII table creation with empty rows."""
        headers = ["Name", "Value"]
        rows: list[list[str]] = []

        table = create_ascii_table(headers, rows)

        assert isinstance(table, str)
        assert "Name" in table
        assert "Value" in table

    def test_create_ascii_table_wide_content(self):
        """Test ASCII table with wide content."""
        headers = ["Short", "Very Long Content That Should Be Handled"]
        rows = [
            [
                "A",
                "This is a very long piece of content that might cause "
                "formatting issues",
            ],
            ["B", "Another long piece of text"],
        ]

        table = create_ascii_table(headers, rows)

        assert isinstance(table, str)
        assert "Short" in table
        assert "Very Long Content" in table

    @patch("subprocess.run")
    def test_run_command_in_dir_success(self, mock_run, temp_dir):
        """Test running command successfully."""
        mock_run.return_value = Mock(returncode=0)

        # Should not raise exception
        run_command_in_dir(["ls", "-la"], temp_dir)

        mock_run.assert_called_once()
        args = mock_run.call_args
        assert args[0][0] == ["ls", "-la"]
        assert args[1]["cwd"] == str(temp_dir)
        assert args[1]["check"] is True

    @patch("subprocess.run")
    def test_run_command_in_dir_failure(self, mock_run, temp_dir):
        """Test running command that fails."""
        mock_run.side_effect = CalledProcessError(1, ["fake_command"])

        with pytest.raises(CalledProcessError):
            run_command_in_dir(["fake_command"], temp_dir)

    def test_create_output_path_basic(self, temp_dir):
        """Test basic output path creation."""
        result = create_output_path(
            output_path=str(temp_dir),
            prefix="test",
            filename="output.png",
            out_dir=None,
        )

        expected = temp_dir / "test_output.png"
        assert result == expected

    def test_create_output_path_with_out_dir(self, temp_dir):
        """Test output path creation with output directory."""
        out_dir = temp_dir / "output"

        result = create_output_path(
            output_path=str(temp_dir),
            prefix="test",
            filename="result.png",
            out_dir=str(out_dir),
        )

        expected = out_dir / "test_result.png"
        assert result == expected
        assert out_dir.exists()  # Should create the directory

    def test_create_output_path_no_prefix(self, temp_dir):
        """Test output path creation without prefix."""
        result = create_output_path(
            output_path=str(temp_dir),
            prefix=None,
            filename="output.png",
            out_dir=None,
        )

        expected = temp_dir / "output.png"
        assert result == expected

    def test_create_output_path_complex(self, temp_dir):
        """Test complex output path creation."""
        out_dir = temp_dir / "analysis" / "plots"

        result = create_output_path(
            output_path=str(temp_dir),
            prefix="experiment_01",
            filename="timing_analysis.png",
            out_dir=str(out_dir),
        )

        expected = out_dir / "experiment_01_timing_analysis.png"
        assert result == expected
        assert out_dir.exists()

    def test_create_output_path_existing_out_dir(self, temp_dir):
        """Test output path creation with existing output directory."""
        out_dir = temp_dir / "existing_output"
        out_dir.mkdir(parents=True)

        # Create a file to verify directory already exists
        existing_file = out_dir / "existing.txt"
        existing_file.write_text("test")

        result = create_output_path(
            output_path=str(temp_dir),
            prefix="test",
            filename="new_file.png",
            out_dir=str(out_dir),
        )

        expected = out_dir / "test_new_file.png"
        assert result == expected
        assert existing_file.exists()  # Should preserve existing content


class TestUtilityEdgeCases:
    """Test edge cases for utility functions."""

    def test_create_ascii_table_unicode(self):
        """Test ASCII table with unicode characters."""
        headers = ["Name", "Symbol"]
        rows = [["Alpha", "α"], ["Beta", "β"], ["Gamma", "γ"]]

        table = create_ascii_table(headers, rows)

        assert isinstance(table, str)
        assert "α" in table
        assert "β" in table
        assert "γ" in table

    def test_create_ascii_table_numeric_content(self):
        """Test ASCII table with numeric content."""
        headers = ["Value", "Squared"]
        rows = [[1, 1], [2, 4], [3, 9]]

        # Convert to strings as expected by the function
        string_rows = [[str(cell) for cell in row] for row in rows]

        table = create_ascii_table(headers, string_rows)

        assert isinstance(table, str)
        assert "Value" in table
        assert "Squared" in table

    @patch("subprocess.run")
    def test_run_command_in_dir_timeout(self, mock_run, temp_dir):
        """Test running command that times out."""
        mock_run.side_effect = TimeoutExpired("sleep", 10)

        with pytest.raises(TimeoutExpired):
            run_command_in_dir(["sleep", "10"], temp_dir)

    def test_create_output_path_none(self, temp_dir):
        """Test create_output_path with None output_path."""
        with patch("pathlib.Path.cwd", return_value=temp_dir):
            result = create_output_path(
                output_path=None,
                prefix="test",
                filename="output.png",
                out_dir=None,
            )

            expected = temp_dir / "test_output.png"
            assert result == expected

    def test_create_output_path_file_exists(self, temp_dir):
        """Test create_output_path when path exists as a file."""
        # Create a file
        file_path = temp_dir / "existing_file.txt"
        file_path.write_text("test content")

        # Try to create output path with a file path
        with pytest.raises(ValueError, match="exists but is not a directory"):
            create_output_path(
                output_path=str(file_path),
                prefix="test",
                filename="output.png",
                out_dir=None,
            )


class TestSlugify:
    """Tests for slugify."""

    def test_default_replaces_runs_and_keeps_case(self):
        """Runs of other characters become one underscore."""
        assert slugify("  Engine: launch (tasks) ") == "Engine_launch_tasks"

    def test_lowercase(self):
        """Lowercasing is optional."""
        assert slugify("Engine Launch", lowercase=True) == "engine_launch"

    def test_keep_extra_characters(self):
        """Extra characters can be kept, e.g. for file labels."""
        assert slugify("run-1.v2 final", keep="._-") == "run-1.v2_final"
        assert slugify("run-1.v2 final") == "run_1_v2_final"

    def test_fallback_when_empty(self):
        """An empty slug falls back to the given text."""
        assert slugify("()", fallback="timer") == "timer"
        assert slugify("") == "item"


class TestShouldUseLogScale:
    """Tests for should_use_log_scale."""

    def test_wide_range(self):
        """Two orders of magnitude switch to a log axis."""
        assert should_use_log_scale([1.0, 100.0])

    def test_narrow_range(self):
        """Smaller ranges stay linear."""
        assert not should_use_log_scale([10.0, 20.0, 999.0])

    def test_zeros_and_negatives_are_ignored(self):
        """Only positive values count towards the range."""
        assert not should_use_log_scale([0.0, -500.0, 5.0, 10.0])
        assert should_use_log_scale([0.0, -1.0, 0.5, 50.0])

    @pytest.mark.parametrize("values", [[], [42.0], [0.0, 0.0], [-1.0, 3.0]])
    def test_too_few_positive_values(self, values):
        """Fewer than two positive values never use a log axis."""
        assert not should_use_log_scale(values)

    def test_accepts_generators(self):
        """Any iterable of values is accepted."""
        assert should_use_log_scale(v for v in (1.0, 1000.0))


class TestSaveFigure:
    """Tests for save_figure."""

    def test_saves_and_closes(self, temp_dir):
        """The figure is written, closed and its path returned."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = temp_dir / "plot.png"

        with patch("swiftsim_cli.utilities.plotting.plt.show") as mock_show:
            result = save_figure(fig, path, dpi=50)

        assert result == path
        assert path.exists()
        assert not plt.fignum_exists(fig.number)
        mock_show.assert_not_called()

    def test_show_displays_before_closing(self, temp_dir):
        """With show=True the figure is shown, then closed."""
        fig, _ = plt.subplots()

        def check_open():
            assert plt.fignum_exists(fig.number)

        with patch(
            "swiftsim_cli.utilities.plotting.plt.show",
            side_effect=check_open,
        ) as mock_show:
            save_figure(fig, temp_dir / "plot.png", show=True, dpi=50)

        mock_show.assert_called_once()
        assert not plt.fignum_exists(fig.number)
