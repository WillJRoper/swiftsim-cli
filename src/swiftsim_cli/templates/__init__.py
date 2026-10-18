"""Runnable SWIFT parameter templates used by ``swift-cli new``.

Each template is a small parameter file derived from SWIFT's own examples
(named in its header comment), holding only what that kind of run needs.
``new`` fills in the initial conditions, units, start time and
softenings.
"""

from importlib.resources import files
from pathlib import Path

# Template name -> what it is for
TEMPLATES = {
    "cosmo-dmo": "cosmological, dark matter only (gravity)",
    "cosmo-hydro": "cosmological, gravity + hydrodynamics",
    "isolated-gravity": "non-cosmological, non-periodic, gravity",
    "isolated-hydro": "non-cosmological, non-periodic, gravity + hydro",
}


def template_path(name: str) -> Path:
    """Return the path of a bundled template.

    Args:
        name: The template name (a key of TEMPLATES).

    Returns:
        The template file.
    """
    if name not in TEMPLATES:
        raise ValueError(
            f"unknown template '{name}'; choose from {', '.join(TEMPLATES)}"
        )
    return Path(str(files(__name__) / f"{name}.yml"))


def choose_template(cosmological: bool, hydro: bool) -> str:
    """Pick the template for a kind of run.

    Args:
        cosmological: Whether the run is cosmological.
        hydro: Whether the run has gas.

    Returns:
        The template name.
    """
    kind = "cosmo" if cosmological else "isolated"
    physics = "hydro" if hydro else ("dmo" if cosmological else "gravity")
    return f"{kind}-{physics}"
