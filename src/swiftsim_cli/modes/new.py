"""New mode for creating new SWIFT run directories.

``new`` starts from a small runnable parameter template bundled with
swift-cli (chosen from the ICs, or with ``--template``), or from the
user's own file (``-P/--params`` or the profile's ``parameter_file``),
fills in what the ICs determine (file name, units, start time,
softenings), applies ``--set`` overrides, checks the result against
SWIFT's parameter parser limits and writes ``params.yml``.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any, Tuple

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from swiftsim_cli.modes.common_args import existing_file
from swiftsim_cli.modes.utils import (
    ICHeader,
    compute_softenings,
    ic_unit_parameters,
    read_ic_header,
)
from swiftsim_cli.params import (
    _parse_parameters,
    check_swift_parameter_text,
    round_trip_yaml,
)
from swiftsim_cli.profile import load_swift_profile
from swiftsim_cli.templates import TEMPLATES, choose_template, template_path
from swiftsim_cli.utilities.messages import note, warn

yaml = round_trip_yaml()

# The (unrunnable) SWIFT example that lists every parameter; never used as
# a base even if a profile points at it
SWIFT_PARAMETER_EXAMPLE = "parameter_example.yml"


def _kv_pair(arg: str) -> Tuple[str, str]:
    """Parse a SECTION:KEY=VALUE override.

    Argparse "type" function: split it into a ("SECTION:KEY", "VALUE")
    tuple, or raise an error if the syntax is wrong or the value is empty.
    """
    key, sep, val = arg.partition("=")
    parts = key.split(":")
    if not sep or len(parts) != 2 or not all(p.strip() for p in parts):
        raise argparse.ArgumentTypeError(
            f"invalid parameter override '{arg}'; expected SECTION:KEY=VALUE"
        )
    if not val.strip():
        raise argparse.ArgumentTypeError(
            f"invalid parameter override '{arg}'; the value is empty"
        )
    return key.strip(), val.strip()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'new' mode."""
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory to create the new run in.",
    )
    parser.add_argument(
        "--ics",
        required=True,
        type=existing_file,
        metavar="FILE",
        help="Initial conditions HDF5 file.",
    )
    base = parser.add_mutually_exclusive_group()
    base.add_argument(
        "--template",
        choices=list(TEMPLATES),
        default=None,
        metavar="NAME",
        help="Parameter template to start from: "
        + "; ".join(f"{k} ({v})" for k, v in TEMPLATES.items())
        + " (default: chosen from the ICs).",
    )
    base.add_argument(
        "-P",
        "--params",
        type=Path,
        default=None,
        metavar="FILE",
        help="Start from this SWIFT parameter file instead of a template "
        "(default: the profile's parameter_file, if set).",
    )
    parser.add_argument(
        "--set",
        metavar="SECTION:KEY=VALUE",
        type=_kv_pair,
        action="append",
        help="Set a SWIFT parameter, e.g. --set "
        "TimeIntegration:dt_max=0.01; can be repeated. Numbers are written "
        "as numbers, anything else verbatim.",
        default=[],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing params.yml in the run directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the profile's SWIFT binary with --dry-run on the new "
        "parameter file (in the run directory) and report the result.",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the new mode."""
    make_new_run_dir(
        output_dir=args.out,
        inicond_file=args.ics,
        overide_params=dict(args.set),
        force=args.force,
        template=args.template,
        params_file=args.params,
        check=args.check,
    )


def _profile_swift_dir() -> Path | None:
    """Return the active profile's SWIFT worktree, or None if unusable."""
    from swiftsim_cli.swiftsim_dir import get_swiftsim_dir

    try:
        return get_swiftsim_dir()
    except (ValueError, OSError):
        return None


def base_parameters(
    header: ICHeader,
    params_file: Path | None = None,
    template: str | None = None,
) -> tuple[dict, str]:
    """Load the parameters a new run starts from.

    In order: ``params_file``, the named template, the profile's
    ``parameter_file`` (unless it is SWIFT's parameter_example.yml), the
    template matching the ICs.

    Args:
        header: The IC header (used to choose a template).
        params_file: The user's own parameter file, if any.
        template: A template name, if one was chosen.

    Returns:
        The parameters and a description of where they came from.
    """
    if params_file is None and template is None:
        profile_file = load_swift_profile().parameter_file
        if profile_file and profile_file.name != SWIFT_PARAMETER_EXAMPLE:
            params_file = profile_file
    if params_file is not None:
        params = _parse_parameters(Path(params_file).expanduser())
        return params, str(params_file)
    if template is None:
        template = choose_template(header.is_cosmological, header.count(0) > 0)
    params = _parse_parameters(template_path(template))
    return params, f"template {template}"


def _section(params: dict, name: str) -> dict:
    """Return a section of the parameters, creating it if missing."""
    if not isinstance(params.get(name), dict):
        params[name] = CommentedMap()
    return params[name]


def apply_ic_settings(
    params: dict, header: ICHeader, quiet: bool = False
) -> None:
    """Set what the ICs determine: units, h-factors, start time, gas.

    Args:
        params: The parameters to update in place.
        header: The IC header.
        quiet: Don't print notes (used by ``utils softenings``).
    """
    say = (lambda message: None) if quiet else note
    cosmological = bool(params.get("Cosmology"))
    if cosmological != header.is_cosmological:
        warn(
            f"the ICs look {'' if header.is_cosmological else 'non-'}"
            "cosmological but the parameters "
            f"{'have' if cosmological else 'have no'} Cosmology section; "
            "keeping the parameters' choice (pick another --template if "
            "that is wrong)"
        )
    ics = _section(params, "InitialConditions")

    unit_params = ic_unit_parameters(header)
    if unit_params:
        units = _section(params, "InternalUnitSystem")
        units.update(unit_params)
        # Drop the template's comments ("# 1 Mpc") on replaced units
        if isinstance(units, CommentedMap):
            for key in unit_params:
                units.ca.items.pop(key, None)
    elif cosmological:
        # Gadget/ic_gen ICs have no Units group, use h^-1 units and
        # sqrt(a) velocities
        ics["cleanup_h_factors"] = 1
        ics["cleanup_velocity_factors"] = 1
        say(
            "the ICs have no Units group: assuming Gadget-style ICs in "
            "the parameters' units with h^-1 factors (cleanup_h_factors "
            "and cleanup_velocity_factors set to 1)"
        )
    else:
        say(
            "the ICs have no Units group: assuming they use the "
            "parameters' InternalUnitSystem"
        )

    if cosmological:
        cosmo = params["Cosmology"]
        if header.a_begin is not None:
            cosmo["a_begin"] = _round12(header.a_begin)
        a_begin = float(cosmo.get("a_begin", 0))
        for name in ("Snapshots", "Statistics"):
            if name in params and "time_first" not in params[name]:
                params[name]["scale_factor_first"] = _round12(a_begin)
        if "SPH" in params and header.count(0) == 0:
            ics["generate_gas_in_ics"] = 1
            ics["cleanup_smoothing_lengths"] = 1
            say("no gas in the ICs: SWIFT will generate it from the DM")
    else:
        params.pop("Cosmology", None)
        time_begin = header.time or 0.0
        _section(params, "TimeIntegration")["time_begin"] = time_begin
        for name in ("Snapshots", "Statistics"):
            if name in params and "scale_factor_first" not in params[name]:
                params[name]["time_first"] = time_begin


def _round12(value: float) -> float:
    """Round to 12 significant figures (drops float noise)."""
    return float(f"{value:.12g}")


def _parse_value(text: str) -> Any:
    """Turn a --set value into an int, float, flow list or verbatim string."""
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    if text.startswith("[") and text.endswith("]"):
        seq = CommentedSeq(
            _parse_value(v.strip()) for v in text[1:-1].split(",") if v.strip()
        )
        seq.fa.set_flow_style()
        return seq
    return text


def apply_overrides(params: dict, overide_params: dict | None = None) -> None:
    """Apply --set overrides to the parameters.

    Values that parse as int or float are written as numbers, ``[a, b]``
    as a list and anything else verbatim. Sections and keys missing from
    the parameters are added, with a warning in case of a typo.

    Args:
        params: The parameter dictionary to update.
        overide_params: {"SECTION:KEY": "VALUE"} overrides.
    """
    for key, value in (overide_params or {}).items():
        parent, sep, child = key.partition(":")
        if not sep or not parent or not child or ":" in child:
            raise ValueError(
                f"Invalid parameter key '{key}'; expected SECTION:KEY"
            )
        if parent not in params:
            warn(f"adding new section {parent} (for --set {key})")
            params[parent] = CommentedMap()
        elif child not in params[parent]:
            warn(f"adding new parameter {key}")
        params[parent][child] = _parse_value(value)


def _report_softenings(params: dict) -> None:
    """Print the softenings written to the parameter file."""
    gravity = params.get("Gravity") or {}
    lines = [
        f"  {key}: {value}"
        for key, value in gravity.items()
        if "softening" in key and "adaptive" not in key
    ]
    if lines:
        print("Softenings:\n" + "\n".join(lines))


def _tidy_blank_lines(text: str) -> str:
    """Keep blank lines between sections only.

    ruamel attaches a section's trailing blank line to its last key, so
    keys added later land after it; move such blank lines to the end of
    the section.
    """
    lines = text.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        following = next((x for x in lines[i + 1 :] if x.strip()), "")
        if not line.strip():
            if following[:1].isspace() or (out and not out[-1].strip()):
                continue
        elif not line[0].isspace() and out and out[-1][:1].isspace():
            out.append("")
        out.append(line)
    return "\n".join(out).rstrip("\n") + "\n"


def swift_flags(params: dict, header: ICHeader, zoom: bool) -> list[str]:
    """Return the SWIFT run-time physics flags for these parameters.

    Without --stars SWIFT drops star particles from the ICs, so it is added
    whenever the ICs have some.
    """
    flags = []
    if params.get("Cosmology"):
        flags.append("--cosmology")
    if "Gravity" in params:
        flags.append("--self-gravity")
    if "SPH" in params:
        flags.append("--hydro")
    if header.count(4) > 0:
        flags.append("--stars")
    if zoom:
        flags.append("--zoom")
    return flags


def make_new_parameter_file(
    output_dir: Path,
    inicond_file: Path,
    swift_dir: Path | None = None,
    overide_params: dict | None = None,
    force: bool = False,
    template: str | None = None,
    params_file: Path | None = None,
) -> tuple[dict, list[str]]:
    """Build, check and write params.yml for a new run.

    Args:
        output_dir: The run directory (created if needed).
        inicond_file: The initial conditions file.
        swift_dir: The SWIFT worktree (default: the profile's), used to
            check for zoom support.
        overide_params: {"SECTION:KEY": "VALUE"} overrides.
        force: Overwrite an existing params.yml.
        template: The template to start from (default: chosen from the
            ICs).
        params_file: The user's own parameter file to start from.

    Returns:
        The parameters written and the SWIFT flags they need.

    Raises:
        FileExistsError: If params.yml exists and force is False.
        ValueError: If the ICs are unusable or the result would be rejected
            by SWIFT's parameter parser.
    """
    # SWIFT runs from inside the run directory, so paths must be absolute
    output_dir = output_dir.resolve()
    inicond_file = inicond_file.resolve()
    output_param_file = output_dir / "params.yml"
    if output_param_file.exists() and not force:
        raise FileExistsError(
            f"{output_param_file} already exists, use --force to overwrite."
        )

    header = read_ic_header(inicond_file)
    params, source = base_parameters(header, params_file, template)
    print(f"Starting from {source}")

    _section(params, "InitialConditions")["file_name"] = str(inicond_file)
    _section(params, "Snapshots")["subdir"] = str(output_dir / "snapshots")
    _section(params, "Restarts")["subdir"] = str(output_dir / "restart")
    apply_ic_settings(params, header)

    # Softenings
    profile = load_swift_profile()
    if not params.get("Cosmology") and not header.is_zoom:
        warn(
            "softenings for a non-cosmological run come from the box size "
            "and particle count, which means little for an isolated object; "
            "check them and set them with --set Gravity:KEY=VALUE"
        )
    _, gravity = compute_softenings(
        header, params, profile.softening_coeff, profile.softening_pivot_z
    )
    _section(params, "Gravity").update(gravity)

    # Zoom ICs: SWIFT's zoom-region code (src/zoom_region) has its own
    # optional section and run-time flag
    zoom = False
    if header.is_zoom and params.get("Cosmology"):
        swift_dir = swift_dir or _profile_swift_dir()
        zoom = bool(swift_dir and (swift_dir / "src" / "zoom_region").is_dir())
        if zoom:
            _section(params, "ZoomRegion").setdefault("region_pad_factor", 1.1)
            note("zoom ICs: added a ZoomRegion section; run SWIFT with --zoom")
        else:
            note(
                "zoom ICs (PartType2 background), but the profile's SWIFT has "
                "no zoom-region support; background particles are softened "
                "with Gravity:softening_ratio_background"
            )

    apply_overrides(params, overide_params)
    _report_softenings(params)

    # Check against SWIFT's parser before writing anything
    from io import StringIO

    buffer = StringIO()
    yaml.dump(params, buffer)
    text = _tidy_blank_lines(buffer.getvalue())
    problems = check_swift_parameter_text(text)
    if problems:
        raise ValueError(
            "the parameter file would be rejected by SWIFT:\n  "
            + "\n  ".join(problems)
        )

    # Write atomically so a failure never leaves a truncated params.yml
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = output_param_file.with_name(output_param_file.name + ".tmp")
    try:
        tmp_file.write_text(text)
        os.replace(tmp_file, output_param_file)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise
    flags = swift_flags(params, header, zoom)
    print(f"Wrote {output_param_file}")
    print(f"Run with: swift {' '.join(flags)} params.yml")
    return params, flags


def check_with_swift(
    run_dir: Path, flags: list[str], swift_dir: Path | None = None
) -> None:
    """Run ``swift --dry-run`` on params.yml in the run directory.

    Args:
        run_dir: The run directory holding params.yml.
        flags: The physics flags to run with.
        swift_dir: The SWIFT worktree (default: the profile's).

    Raises:
        ValueError: If there is no SWIFT binary or the dry run fails.
    """
    swift_dir = swift_dir or _profile_swift_dir()
    binary = swift_dir / "swift" if swift_dir else None
    if binary is None or not binary.is_file():
        raise ValueError(
            "--check needs a built SWIFT binary in the profile's worktree "
            f"({binary or 'no worktree'}); build it with 'swift-cli make'"
        )
    command = [str(binary), "--dry-run", *flags, "params.yml"]
    print(f"Checking with: {' '.join(command)}")
    result = subprocess.run(
        command, cwd=run_dir, capture_output=True, text=True
    )
    output = result.stdout.strip().splitlines()
    if result.returncode != 0:
        # SWIFT's error is on stderr; fall back to the end of stdout
        lines = result.stderr.strip().splitlines() or output
        raise ValueError(
            "SWIFT rejected the parameter file:\n  " + "\n  ".join(lines[-10:])
        )
    print("SWIFT dry run passed: " + (output[-1] if output else ""))


def make_new_run_dir(
    output_dir: Path,
    inicond_file: Path,
    swift_dir: Path | None = None,
    overide_params: dict | None = None,
    force: bool = False,
    template: str | None = None,
    params_file: Path | None = None,
    check: bool = False,
) -> None:
    """Create a new SWIFT run directory with a parameter file.

    Args:
        output_dir: The run directory.
        inicond_file: The initial conditions file.
        swift_dir: The SWIFT worktree (default: the profile's).
        overide_params: {"SECTION:KEY": "VALUE"} overrides.
        force: Overwrite an existing params.yml.
        template: The template to start from.
        params_file: The user's own parameter file to start from.
        check: Run ``swift --dry-run`` on the result.
    """
    _, flags = make_new_parameter_file(
        output_dir,
        inicond_file,
        swift_dir,
        overide_params,
        force,
        template,
        params_file,
    )
    if check:
        check_with_swift(output_dir.resolve(), flags, swift_dir)
