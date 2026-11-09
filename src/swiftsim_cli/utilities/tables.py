"""ASCII table formatting."""

from collections.abc import Sequence


def _is_number(cell: object) -> bool:
    """Whether a cell reads as a number (e.g. "1.5", "12%", "3e4")."""
    try:
        float(str(cell).strip().rstrip("%").replace(",", ""))
    except ValueError:
        return False
    return True


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

    # Right-align columns of numbers, left-align everything else
    numeric = [
        i > 0 and all(_is_number(row[i]) for row in rows if str(row[i]))
        for i in range(len(headers))
    ]

    # Add padding
    col_widths = [w + 2 for w in col_widths]

    # Widen the last column if the title would not fit
    if title:
        table_width = sum(col_widths) + len(col_widths) + 1
        col_widths[-1] += max(0, len(title) + 4 - table_width)

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
            align = ">" if numeric[i] else "<"
            row_line += f" {str(cell):{align}{col_widths[i] - 2}} |"
        table_lines.append(row_line)

    table_lines.append(separator)

    return "\n".join(table_lines)
