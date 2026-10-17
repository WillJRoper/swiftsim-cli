"""Tests for the gravity error map analysis mode."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from swiftsim_cli.modes.analyse import gravity_error_maps
from swiftsim_cli.modes.analyse.gravity_error_maps import (
    _create_continuous_error_map,
    analyse_gravity_error_maps,
)

from .test_gravity_checks import write_force_file


def test_distance_panel_autoscales_counts_and_uses_box_centre(tmp_path):
    """The count hexbin is autoscaled and distances use the data centre."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 10.0, size=(500, 3))
    errors = rng.uniform(1e-6, 1e-2, size=500)

    with patch.object(gravity_error_maps.plt, "close"):
        _create_continuous_error_map(
            exact_pos=pos,
            norm_error=errors,
            label="run",
            resolution=10,
            output_path=str(tmp_path),
            prefix=None,
            show_plot=False,
        )
        fig = plt.gcf()

    counts = fig.axes[3].collections[0]
    assert counts.norm.vmin >= 1
    assert counts.norm.vmax > counts.norm.vmin
    # Max distance from the centre of a 10^3 box is 5*sqrt(3) ~ 8.7;
    # measuring from (0.5, 0.5, 0.5) would reach ~16.
    assert counts.get_offsets()[:, 0].max() < 9.5
    plt.close(fig)


def test_analyse_gravity_error_maps_with_shuffled_ids(tmp_path):
    """End to end: misaligned IDs are matched and both maps written."""
    exact = tmp_path / "gravity_checks_exact_step0001.dat"
    swift = tmp_path / "gravity_checks_swift_step0001_order3.dat"
    write_force_file(exact, list(range(20)))
    write_force_file(swift, list(range(25)), accel_scale=1.001)

    analyse_gravity_error_maps(
        [exact],
        ["my run"],
        output_path=tmp_path,
        show_plot=False,
        resolution=5,
    )

    assert (tmp_path / "gravity_error_map_my_run.png").exists()
    assert (tmp_path / "gravity_binary_error_map_my_run.png").exists()
