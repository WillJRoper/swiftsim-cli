"""Tests for the gravity error map analysis mode."""

from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from swiftsim_cli.modes.analyse.gravity_error_maps import (
    _create_continuous_error_map,
    _distance_from_centre,
    analyse_gravity_error_maps,
)

from .test_gravity_checks import write_force_file


def test_distance_uses_median_centre_and_periodic_wrap():
    """An outlier does not move the centre; distances wrap with a box."""
    pos = np.array([[1.0, 1, 1], [1.2, 1, 1], [0.8, 1, 1], [9.0, 1, 1]])

    distance = _distance_from_centre(pos, box_size=None)
    # Median x is 1.1: a bounding-box centre would be 4.9
    np.testing.assert_allclose(distance, [0.1, 0.1, 0.3, 7.9])

    wrapped = _distance_from_centre(pos, box_size=10.0)
    np.testing.assert_allclose(wrapped, [0.1, 0.1, 0.3, 2.1])


def _map_figure(tmp_path, n, **kwargs):
    """Draw the continuous map for n random particles; return the figure."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 10.0, size=(n, 3))
    errors = 10 ** rng.uniform(-6, -2, size=n)
    with patch("matplotlib.pyplot.close"):
        _create_continuous_error_map(
            exact_pos=pos,
            norm_error=errors,
            label="run",
            resolution=10,
            output_path=str(tmp_path),
            prefix=None,
            show_plot=False,
            **kwargs,
        )
        fig = plt.gcf()
    return fig, errors


def test_colour_scale_autoscales_from_percentiles(tmp_path):
    """Large samples: median hexbins with a 1st-99th percentile scale."""
    fig, errors = _map_figure(tmp_path, 6000)

    hexes: Any = fig.axes[0].collections[0]
    low, high = np.percentile(errors, [1, 99])
    assert hexes.norm.vmin == pytest.approx(low)
    assert hexes.norm.vmax == pytest.approx(high)
    [radial] = [ax for ax in fig.axes if ax.get_title().startswith("Error vs")]
    counts: Any = radial.collections[0]  # counts, autoscaled
    assert counts.norm.vmin >= 1
    plt.close(fig)


def test_colour_scale_can_be_set(tmp_path):
    """--vmin/--vmax override the percentile scale."""
    fig, _ = _map_figure(tmp_path, 6000, vmin=1e-5, vmax=1e-3)

    hexes: Any = fig.axes[0].collections[0]
    assert (hexes.norm.vmin, hexes.norm.vmax) == (1e-5, 1e-3)
    plt.close(fig)


def test_small_samples_are_scattered(tmp_path):
    """A few hundred particles are drawn individually, not as hexbins."""
    from matplotlib.collections import PathCollection

    fig, _ = _map_figure(tmp_path, 200)

    assert isinstance(fig.axes[0].collections[0], PathCollection)
    plt.close(fig)


def test_vmin_must_be_below_vmax():
    """An inverted colour range is rejected up front."""
    with pytest.raises(ValueError, match="--vmin < --vmax"):
        analyse_gravity_error_maps(["a"], ["a"], vmin=1.0, vmax=0.1)


def test_labels_are_slugified_without_collisions(tmp_path):
    """Labels become safe file names; clashing slugs get a suffix."""
    files = []
    for step in (1, 2):
        exact = tmp_path / f"gravity_checks_exact_step000{step}.dat"
        write_force_file(exact, list(range(10)))
        write_force_file(
            tmp_path / f"gravity_checks_swift_step000{step}_order3.dat",
            list(range(10)),
            accel_scale=1.01,
        )
        files.append(exact)

    analyse_gravity_error_maps(
        files, ["a/b", "a b"], output_path=tmp_path, resolution=5
    )

    out_dir = tmp_path / "gravity_error_maps_analysis"
    assert (out_dir / "gravity_error_map_a_b.png").exists()
    assert (out_dir / "gravity_error_map_a_b_2.png").exists()


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
