"""The help text quoted in the docs matches the real help."""

import re
import textwrap
from pathlib import Path

import pytest

from swiftsim_cli.multi_mode_args import MultiModeCLIArgs, top_level_help

DOCS = Path(__file__).parents[2] / "docs" / "source"


def _block_after(page: str, marker: str) -> str:
    """Return the first ``code-block:: text`` after ``marker`` in a page."""
    text = (DOCS / page).read_text(encoding="utf-8")
    match = re.search(
        re.escape(marker)
        + r".*?\.\. code-block:: text\n\n((?:   [^\n]*\n|\n)+)",
        text,
        re.DOTALL,
    )
    assert match, f"no text block after {marker!r} in {page}"
    return textwrap.dedent(match.group(1)).strip("\n") + "\n"


def test_docs_top_level_help():
    """The docs' `swift-cli --help` block is the real output."""
    block = _block_after("command_line/index.rst", "``swift-cli --help``")
    assert block == top_level_help() + "\n"


def test_docs_analyse_help(monkeypatch, capsys):
    """The docs' `analyse --help` block is the real output (80 cols).

    Whitespace is ignored: argparse's column layout differs between Python
    patch releases.
    """
    monkeypatch.setenv("COLUMNS", "80")
    with pytest.raises(SystemExit):
        MultiModeCLIArgs(["analyse", "--help"])
    block = _block_after("analysis/index.rst", "``swift-cli analyse --help``")
    assert block.split() == capsys.readouterr().out.split()
