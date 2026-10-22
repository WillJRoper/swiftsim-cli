"""Built-in SWIFT task/cell name tables and colour helpers for task plots.

The tables here are only a fallback: :mod:`.task_names` reads the names
from the SWIFT source the dump was written by whenever a checkout is
available. Two snapshots are kept because the zoom branch inserts a
``progeny`` subtype (shifting every later subtype index) and adds cell
type/subtype columns to the dump.
"""

from __future__ import annotations

from typing import Sequence

# taskID_names in src/task.c (identical on master and the zoom branch).
TASK_TYPES = (
    "none",
    "sort",
    "self",
    "pair",
    "init_grav",
    "init_grav_out",
    "ghost_in",
    "ghost",
    "ghost_out",
    "extra_ghost",
    "drift_part",
    "drift_spart",
    "drift_sink",
    "drift_bpart",
    "drift_gpart",
    "drift_gpart_out",
    "hydro_end_force",
    "kick1",
    "kick2",
    "timestep",
    "timestep_limiter",
    "timestep_sync",
    "collect",
    "send",
    "recv",
    "pack",
    "unpack",
    "grav_long_range",
    "grav_mm",
    "grav_down_in",
    "grav_down",
    "grav_end_force",
    "cooling",
    "cooling_in",
    "cooling_out",
    "star_formation",
    "star_formation_in",
    "star_formation_out",
    "star_formation_sink",
    "csds",
    "stars_in",
    "stars_out",
    "stars_ghost_in",
    "stars_density_ghost",
    "stars_ghost_out",
    "stars_prep_ghost1",
    "hydro_prep_ghost1",
    "stars_prep_ghost2",
    "stars_sort",
    "stars_resort",
    "bh_in",
    "bh_out",
    "bh_density_ghost",
    "bh_swallow_ghost1",
    "bh_swallow_ghost2",
    "bh_swallow_ghost3",
    "fof_self",
    "fof_pair",
    "fof_attach_self",
    "fof_attach_pair",
    "neutrino_weight",
    "sink_in",
    "sink_density_ghost",
    "sink_ghost1",
    "sink_ghost2",
    "sink_out",
    "rt_in",
    "rt_out",
    "sink_formation",
    "rt_ghost1",
    "rt_ghost2",
    "rt_transport_out",
    "rt_tchem",
    "rt_advance_cell_time",
    "rt_sorts",
    "rt_collect_times",
)

# subtaskID_names in src/task.c on upstream master.
UPSTREAM_SUBTYPES = (
    "none",
    "density",
    "gradient",
    "force",
    "limiter",
    "grav",
    "fof",
    "external_grav",
    "tend",
    "xv",
    "rho",
    "part_swallow",
    "bpart_merger",
    "gpart",
    "spart_density",
    "part_prep1",
    "spart_prep2",
    "stars_density",
    "stars_prep1",
    "stars_prep2",
    "stars_feedback",
    "sf_counts",
    "grav_counts",
    "bpart_rho",
    "bpart_feedback",
    "bh_density",
    "bh_swallow",
    "do_gas_swallow",
    "do_bh_swallow",
    "bh_feedback",
    "sink_density",
    "sink_do_sink_swallow",
    "sink_swallow",
    "sink_do_gas_swallow",
    "rt_gradient",
    "rt_transport",
)

# The zoom branch adds "progeny" after "grav".
ZOOM_SUBTYPES = (*UPSTREAM_SUBTYPES[:6], "progeny", *UPSTREAM_SUBTYPES[6:])

# enum cell_types / cell_subtypes in the zoom branch's src/cell.h. Older
# zoom dumps had a buffer cell type, so background was 3.
ZOOM_CELL_TYPES = ("Regular", "Zoom", "Background")
LEGACY_ZOOM_CELL_TYPES = ("Regular", "Zoom", "Buffer", "Background")
ZOOM_CELL_SUBTYPES = ("Regular", "Neighbour", "Void")


def distinct_colours(count: int) -> list[str]:
    """Return ``count`` visually distinct colours as hex strings.

    The first 30 are the saturated halves of matplotlib's tab20, tab20b
    and tab20c maps, then their light halves; beyond 60 the hue circle is
    sampled evenly.
    """
    from matplotlib import colormaps
    from matplotlib.colors import hsv_to_rgb, to_hex

    maps = [
        [to_hex(colormaps[name](i)) for i in range(20)]
        for name in ("tab20", "tab20b", "tab20c")
    ]
    palette = [c for m in maps for c in m[::2]] + [
        c for m in maps for c in m[1::2]
    ]
    if count <= len(palette):
        return palette[:count]
    return [
        to_hex(tuple(hsv_to_rgb((i / count, 0.75, 0.6 + 0.3 * (i % 2)))))
        for i in range(count)
    ]


def colour_map(labels: Sequence[str]) -> dict[str, str]:
    """Map labels to distinct colours in the order given.

    Callers pass labels in a fixed order (e.g. by task-type index, or
    sorted), so a label keeps its colour across ranks and plots of one run
    rather than depending on how often it occurs.
    """
    unique = list(dict.fromkeys(labels))
    return dict(zip(unique, distinct_colours(len(unique))))
