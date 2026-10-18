"""ASCII table formatting."""

from collections.abc import Sequence


def create_ascii_table(
    headers: Sequence[object],
    rows: Sequence[Sequence[object]],
    title: str | None = None,
) -> str:
    """Create a formatted ASCII table.

    Args:
        headers: List of column headers
        rows: List of row data (each row is a list)
        title: Optional table title

    Returns:
        String containing the formatted table
    """
    # Calculate column widths
    col_widths = [len(str(header)) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Add padding
    col_widths = [w + 2 for w in col_widths]

    # Create separator line
    separator = "+" + "+".join("-" * w for w in col_widths) + "+"

    # Build table
    table_lines = []

    # Title
    if title:
        total_width = len(separator)
        table_lines.append("+" + "=" * (total_width - 2) + "+")
        title_line = f"| {title:^{total_width - 4}} |"
        table_lines.append(title_line)

    table_lines.append(separator)

    # Headers
    header_line = "|"
    for i, header in enumerate(headers):
        header_line += f" {str(header):^{col_widths[i] - 2}} |"
    table_lines.append(header_line)
    table_lines.append(separator)

    # Rows
    for row in rows:
        row_line = "|"
        for i, cell in enumerate(row):
            # Right-align numbers, left-align text
            cell_str = str(cell)
            if i == 0:  # First column (names) - left align
                row_line += f" {cell_str:<{col_widths[i] - 2}} |"
            else:  # Other columns (numbers) - right align
                row_line += f" {cell_str:>{col_widths[i] - 2}} |"
        table_lines.append(row_line)

    table_lines.append(separator)

    return "\n".join(table_lines)
