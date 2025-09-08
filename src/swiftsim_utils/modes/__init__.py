"""Mode definitions and constants for swift-cli."""

from typing import Literal

from . import analyse, config, make, new, output_times, profile, switch, update

# Available modes for swift-utils CLI
AVAILABLE_MODES = [
    "profile",
    "config",
    "output-times",
    "update",
    "switch",
    "make",
    "new",
    "analyse",
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
]

# Mode module imports

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
}
