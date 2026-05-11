"""Metadata and colour helpers for SWIFT task-debug plots."""

from __future__ import annotations

TASKTYPES = [
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
    "rt_sort",
    "rt_collect_times",
]

SUBTYPES = [
    "none",
    "density",
    "gradient",
    "force",
    "limiter",
    "grav",
    "progeny",
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
]

CELL_TYPES = ["Regular", "Zoom", "Buff", "Bkg"]
CELL_SUBTYPES = ["Regular", "Neighbour", "Void", None]

FULLTYPES = [
    "self/limiter",
    "self/force",
    "self/gradient",
    "self/density",
    "self/grav",
    "pair/limiter",
    "pair/force",
    "pair/gradient",
    "pair/density",
    "pair/grav",
    "recv/xv",
    "send/xv",
    "recv/rho",
    "send/rho",
    "recv/tend_part",
    "send/tend_part",
    "recv/tend_gpart",
    "send/tend_gpart",
    "recv/tend_spart",
    "send/tend_spart",
    "recv/tend_bpart",
    "send/tend_bpart",
    "recv/gpart",
    "send/gpart",
    "recv/spart",
    "send/spart",
    "send/sf_counts",
    "recv/sf_counts",
    "recv/bpart",
    "send/bpart",
    "recv/limiter",
    "send/limiter",
    "pack/limiter",
    "unpack/limiter",
    "self/stars_density",
    "pair/stars_density",
    "self/stars_prep1",
    "pair/stars_prep1",
    "self/stars_prep2",
    "pair/stars_prep2",
    "self/stars_feedback",
    "pair/stars_feedback",
    "self/bh_density",
    "pair/bh_density",
    "self/bh_swallow",
    "pair/bh_swallow",
    "self/do_swallow",
    "pair/do_swallow",
    "self/bh_feedback",
    "pair/bh_feedback",
    "self/rt_gradient",
    "pair/rt_gradient",
    "self/rt_transport",
    "pair/rt_transport",
    "self/sink_density",
    "pair/sink_density",
    "self/sink_swallow",
    "pair/sink_swallow",
    "self/sink_do_swallow",
    "pair/sink_do_swallow",
    "self/sink_do_gas_swallow",
    "pair/sink_do_gas_swallow",
]

COLOURS = [
    "cyan",
    "lightgray",
    "darkblue",
    "yellow",
    "tan",
    "dodgerblue",
    "sienna",
    "aquamarine",
    "bisque",
    "blue",
    "green",
    "lightgreen",
    "brown",
    "purple",
    "moccasin",
    "olivedrab",
    "chartreuse",
    "olive",
    "darkgreen",
    "green",
    "mediumseagreen",
    "mediumaquamarine",
    "darkslategrey",
    "mediumturquoise",
    "black",
    "cadetblue",
    "skyblue",
    "red",
    "slategray",
    "gold",
    "slateblue",
    "blueviolet",
    "mediumorchid",
    "firebrick",
    "magenta",
    "hotpink",
    "pink",
    "orange",
    "lightgreen",
]


def _assign_palette(labels: list[str], start_index: int = 0) -> dict[str, str]:
    """Assign colours to labels, recycling the SWIFT palette when needed."""
    colour_map: dict[str, str] = {}
    palette_size = len(COLOURS)
    for index, label in enumerate(labels):
        colour_map[label] = COLOURS[(start_index + index) % palette_size]
    return colour_map


TASK_COLOURS = _assign_palette(TASKTYPES)

_subtype_colours = _assign_palette(FULLTYPES + SUBTYPES, len(TASKTYPES))
SUB_COLOURS = {
    label: _subtype_colours[label] for label in FULLTYPES + SUBTYPES
}


def task_label(task_type_index: int, subtype_index: int) -> str:
    """Return the human-readable task label for integer task IDs."""
    task_type = TASKTYPES[task_type_index]
    subtype = SUBTYPES[subtype_index]
    if subtype == "none":
        return task_type
    return f"{task_type}/{subtype}"


def task_colour(task_type: str, subtype: str) -> str:
    """Return the colour used by SWIFT's task plotting scripts."""
    if "fof" in task_type:
        return TASK_COLOURS[task_type]

    if any(key in task_type for key in ("self", "pair", "recv", "send")):
        full_type = f"{task_type}/{subtype}"
        if full_type in SUB_COLOURS:
            return SUB_COLOURS[full_type]
        return SUB_COLOURS.get(subtype, TASK_COLOURS[task_type])

    return TASK_COLOURS[task_type]
