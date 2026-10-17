"""Shared utilities for SWIFTSim-CLI.

Submodules:
    io: Subprocess, output-path and filename helpers.
    tables: ASCII table formatting.
    plotting: Matplotlib helpers (imports matplotlib, so not re-exported).
    step_table: Parser for SWIFT's per-step stdout/timesteps table.
"""

from swiftsim_cli.utilities.io import (
    ascii_art,
    create_output_path,
    run_command_in_dir,
    slugify,
)
from swiftsim_cli.utilities.tables import create_ascii_table

__all__ = [
    "ascii_art",
    "create_ascii_table",
    "create_output_path",
    "run_command_in_dir",
    "slugify",
]
