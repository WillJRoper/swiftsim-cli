"""Mode definitions and constants for swift-cli."""

from typing import Literal

from . import (
    analyse,
    config,
    make,
    new,
    output_times,
    profile,
    show,
    switch,
    update,
    utils,
)

# Available modes for the swift-cli CLI
AVAILABLE_MODES = [
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

# Type hint for mode names
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

# One-line description of each mode, shown by 'swift-cli --help' and at the
# top of 'swift-cli <mode> --help'
MODE_DESCRIPTIONS = {
    "profile": "Manage SWIFT profiles (one git worktree, branch and build "
    "each).",
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

# Mapping of mode names to their modules
MODE_MODULES = {
    "profile": profile,
    "config": config,
    "output-times": output_times,
    "update": update,
    "switch": switch,
    "make": make,
    "new": new,
    "analyse": analyse,
    "show": show,
    "utils": utils,
}
