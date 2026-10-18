"""Mode definitions for swift-cli.

`Mode` is the single list of modes. Each mode lives in the module of the
same name (dashes become underscores) and is only imported by `load_mode`
when that mode is parsed or run, so ``swift-cli --help`` and friends never
pay for the heavy imports of the analysis modes.
"""

import importlib
from types import ModuleType
from typing import Literal, get_args

Mode = Literal[
    "profile",
    "config",
    "output-times",
    "update",
    "switch",
    "make",
    "new",
    "analyse",
    "show",
    "utils",
]

AVAILABLE_MODES: tuple[Mode, ...] = get_args(Mode)

# One-line description of each mode, shown by 'swift-cli --help' and at the
# top of 'swift-cli <mode> --help'
MODE_DESCRIPTIONS: dict[Mode, str] = {
    "profile": "Manage SWIFT profiles (one worktree, branch and build each).",
    "config": "Set the current profile's configure flags and reconfigure.",
    "output-times": "Generate a list of snapshot (and snipshot) output times.",
    "update": "Pull and rebuild the current profile.",
    "switch": "Make another profile the current one.",
    "make": "Compile the current profile.",
    "new": "Create a new simulation run directory.",
    "analyse": "Analyse SWIFT logs and diagnostic outputs.",
    "show": "Show the current profile (and optionally its cosmology).",
    "utils": "Run small helper calculations (e.g. softenings).",
}


def load_mode(name: Mode) -> ModuleType:
    """Import and return the module implementing a mode.

    Args:
        name: The mode name.

    Returns:
        The mode's module (with ``add_arguments`` and ``run``).
    """
    return importlib.import_module(f"{__name__}.{name.replace('-', '_')}")
