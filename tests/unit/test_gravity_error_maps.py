"""Tests for the gravity error map analysis mode."""

from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

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

    with patch("matplotlib.pyplot.close"):
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

    counts: Any = fig.axes[3].collections[0]  # a PolyCollection (hexbin)
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

    out_dir = tmp_path / "gravity_error_maps_analysis"
    assert (out_dir / "gravity_error_map_my_run.png").exists()
    assert (out_dir / "gravity_binary_error_map_my_run.png").exists()


def test_error_maps_fail_when_no_input_is_usable(tmp_path, capsys):
    """Every input failing is an error, and no maps are written."""
    lonely = tmp_path / "gravity_checks_exact_step0001.dat"
    write_force_file(lonely, [1, 2, 3])

    with pytest.raises(ValueError, match="none of the gravity check inputs"):
        analyse_gravity_error_maps([lonely], ["run"], output_path=tmp_path)

    assert "warning: skipping" in capsys.readouterr().err
    assert not (tmp_path / "gravity_error_maps_analysis").exists()
