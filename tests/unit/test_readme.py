"""The help text quoted in the README matches the real help."""

import re
from pathlib import Path

import pytest

from swiftsim_cli.multi_mode_args import MultiModeCLIArgs, top_level_help

README = Path(__file__).parents[2] / "README.md"


def _block_after(marker: str) -> str:
    """Return the first fenced code block after ``marker`` in the README."""
    text = README.read_text(encoding="utf-8")
    match = re.search(
        re.escape(marker) + r".*?```text\n(.*?)```", text, re.DOTALL
    )
    assert match, f"no code block after {marker!r} in the README"
    return match.group(1)


def test_readme_top_level_help():
    """The README's `swift-cli --help` block is the real output."""
    assert (
        _block_after("`swift-cli --help` prints:") == top_level_help() + "\n"
    )


def test_readme_analyse_help(monkeypatch, capsys):
    """The README's `analyse --help` block is the real output (80 cols).

    Whitespace is ignored: argparse's column layout differs between Python
    patch releases.
    """
    monkeypatch.setenv("COLUMNS", "80")
    with pytest.raises(SystemExit):
        MultiModeCLIArgs(["analyse", "--help"])
    readme = _block_after("## Analysis (`analyse`)")
    assert readme.split() == capsys.readouterr().out.split()
