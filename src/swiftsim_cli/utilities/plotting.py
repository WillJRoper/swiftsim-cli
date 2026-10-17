"""Matplotlib helpers shared by the plotting modes."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_figure(
    fig: Figure, path: Path, show: bool = False, dpi: int = 300
) -> Path:
    """Save a figure, optionally show it, then close it.

    Args:
        fig: The figure to save.
        path: Where to write the image.
        show: Whether to display the figure interactively after saving.
        dpi: Resolution of the saved image.

    Returns:
        The path the figure was saved to.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def should_use_log_scale(values: Iterable[float]) -> bool:
    """Return whether positive values span two orders of magnitude.

    Zero and negative values are ignored since they cannot be shown on a
    log axis.

    Args:
        values: The values that will be plotted.

    Returns:
        True if at least two positive values exist and their max/min ratio
        is at least 100.
    """
    positive_values = [value for value in values if value > 0]
    if len(positive_values) < 2:
        return False
    return max(positive_values) / min(positive_values) >= 100.0
