"""Unit tests for utility functions."""

from subprocess import CalledProcessError, TimeoutExpired
from unittest.mock import Mock, patch

import pytest

from swiftsim_cli.utilities import (
    create_ascii_table,
    create_output_path,
    make_directory,
    run_command_in_dir,
)


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
        rows = []

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

    def test_make_directory_new(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_directory"

        assert not new_dir.exists()
        make_directory(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_make_directory_existing(self, temp_dir):
        """Test making directory that already exists."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        assert existing_dir.exists()
        # Should not raise exception
        make_directory(existing_dir)
        assert existing_dir.exists()

    def test_make_directory_nested(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"

        assert not nested_dir.exists()
        make_directory(nested_dir)
        assert nested_dir.exists()
        assert nested_dir.is_dir()

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

    def test_make_directory_permission_error(self, temp_dir):
        """Test make_directory with permission error."""
        # Create a scenario that might cause permission error
        restricted_path = temp_dir / "restricted"

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")

            with pytest.raises(PermissionError):
                make_directory(restricted_path)

    @patch("subprocess.run")
    def test_run_command_in_dir_timeout(self, mock_run, temp_dir):
        """Test running command that times out."""
        mock_run.side_effect = TimeoutExpired("sleep", 10)

        with pytest.raises(TimeoutExpired):
            run_command_in_dir(["sleep", "10"], temp_dir)

    def test_make_directory_file_exists(self, temp_dir):
        """Test make_directory when path exists as a file."""
        # Create a file
        file_path = temp_dir / "existing_file.txt"
        file_path.write_text("test content")

        # Try to make a directory with the same path
        with pytest.raises(
            FileExistsError, match="exists and is not a directory"
        ):
            make_directory(file_path)

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
