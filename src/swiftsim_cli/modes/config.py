"""Config mode for configuring the active SWIFT profile."""

import argparse

from swiftsim_cli.profile import get_profile
from swiftsim_cli.swiftsim_dir import (
    _run_command_in_swift_dir,
    configure_profile,
    ensure_configure_script,
    get_swiftsim_dir,
)
from swiftsim_cli.utilities.messages import warn

# Preset flag -> (description, configure options)
PRESETS: dict[str, tuple[str, list[str]]] = {
    "--debug-build": (
        "debug",
        [
            "--enable-debug",
            "--enable-debugging-checks",
            "--disable-optimization",
        ],
    ),
    "--gravity": (
        "gravity-only",
        ["--enable-ipo", "--with-tbbmalloc", "--with-parmetis"],
    ),
    "--eagle": (
        "EAGLE",
        [
            "--with-subgrid=EAGLE",
            "--with-hydro=sphenix",
            "--with-kernel=wendland-C2",
            "--enable-ipo",
            "--with-tbbmalloc",
            "--with-parmetis",
        ],
    ),
    "--eaglexl": (
        "EAGLE-XL",
        [
            "--with-subgrid=EAGLE-XL",
            "--with-hydro=sphenix",
            "--with-kernel=wendland-C2",
            "--enable-ipo",
            "--with-tbbmalloc",
            "--with-parmetis",
        ],
    ),
}

# Presets that set the same options differently
EXCLUSIVE_PRESETS = ("--eagle", "--eaglexl")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for configuring SWIFT itself."""
    parser.epilog = (
        "Any other --<option>[=value] flags and VAR=value assignments are "
        "passed to ./configure after being checked against its options "
        "(list them with --list-options)."
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List the available ./configure options (./configure --help) "
        "and exit.",
    )
    exclusive = parser.add_mutually_exclusive_group()
    for flag, (what, opts) in PRESETS.items():
        target = exclusive if flag in EXCLUSIVE_PRESETS else parser
        target.add_argument(
            flag,
            action="store_true",
            help=f"Configure using the preset {what} options "
            f"({' '.join(opts)}).",
        )


def merge_flags(opts: list[str]) -> list[str]:
    """Drop repeated configure flags; a later value for an option wins.

    Args:
        opts: The flags in order (presets first, then the user's).

    Returns:
        The flags with one entry per option, warning about overrides.
    """
    merged: dict[str, str] = {}
    for opt in opts:
        key = opt.split("=", 1)[0]
        old = merged.get(key)
        if old == opt:
            continue
        if old is not None:
            warn(f"{opt} overrides {old}")
            del merged[key]
        merged[key] = opt
    return list(merged.values())


def run(args: argparse.Namespace) -> None:
    """Execute the config mode.

    The presets and any extra ``--<option>`` flags become the active
    profile's configure flags (validated against ``./configure --help``).
    With no presets or flags, the profile is reconfigured with its stored
    flags.

    Args:
        args: The parsed arguments from the command line.
    """
    if args.list_options:
        show_config_options()
        return
    opts = [
        opt
        for flag, (_, preset) in PRESETS.items()
        if getattr(args, flag[2:].replace("-", "_"))
        for opt in preset
    ]
    opts.extend(args.options)
    config_swiftsim(opts=merge_flags(opts) or None)


def config_swiftsim(opts: list[str] | None = None) -> None:
    """Configure the active profile's worktree.

    Args:
        opts: Flags for ``./configure``; these replace the profile's stored
            configure flags. If None, the stored flags are used.

    Raises:
        ProfileError: If a flag is unknown or configure fails.
    """
    configure_profile(get_profile(), opts)


def show_config_options() -> None:
    """Show the configure options for the active profile's worktree."""
    swift_dir = get_swiftsim_dir()
    ensure_configure_script(swift_dir)
    _run_command_in_swift_dir(["./configure", "--help"], swift_dir)
