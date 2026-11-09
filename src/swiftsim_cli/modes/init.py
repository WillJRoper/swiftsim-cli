"""Init mode: set up swift-cli in one command.

Uses an existing SWIFT clone (``--swift-path``) or clones SWIFT into
``./swiftsim``, then creates the first profile (a worktree of that clone)
and prints where everything lives and the profile's settings. Later
profiles are added with 'swift-cli profile --new NAME'.
"""

import argparse
import shlex
import subprocess
from pathlib import Path

from swiftsim_cli import profile as profile_mod
from swiftsim_cli.modes.common_args import add_params_argument
from swiftsim_cli.modes.profile import DEFAULT_UPSTREAM, display_profile
from swiftsim_cli.profile import (
    ProfileError,
    ProfileStore,
    add_profile,
    default_worktree_root,
    load_store,
    main_clone_of,
    read_raw_profile_file,
    save_store,
    validate_profile_name,
)
from swiftsim_cli.utilities.tables import create_ascii_table

SWIFT_URL = "https://github.com/SWIFTSIM/SWIFT.git"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the 'init' mode."""
    parser.add_argument(
        "--swift-path",
        metavar="DIR",
        help="An existing SWIFT clone to use (default: clone SWIFT into "
        "./swiftsim).",
    )
    parser.add_argument(
        "--swift-url",
        default=SWIFT_URL,
        metavar="URL",
        help=f"Repository to clone without --swift-path (default: "
        f"{SWIFT_URL}).",
    )
    parser.add_argument(
        "--worktree-root",
        metavar="DIR",
        help="Directory profile worktrees are created in (default: "
        "~/.swiftsim-cli/worktrees).",
    )
    parser.add_argument(
        "--name",
        default="default",
        help="The first profile's name (default: default).",
    )
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        metavar="BRANCH",
        help=f"The first profile's upstream branch (default: "
        f"{DEFAULT_UPSTREAM}).",
    )
    parser.add_argument(
        "--configure-flags",
        default="",
        metavar="FLAGS",
        help="The first profile's configure flags, as one string.",
    )
    add_params_argument(
        parser,
        help="The first profile's parameter file, the source of its "
        "cosmology (default: the worktree's examples/parameter_example.yml).",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the init mode."""
    raw = read_raw_profile_file()
    if raw is not None:
        existing = load_store()
        if existing.profiles:
            raise ProfileError(
                f"Already set up with {existing.repo} (profiles: "
                f"{', '.join(existing.profiles)}). Add profiles with "
                "'swift-cli profile --new NAME'."
            )
    # Check before a long clone that would then be wasted
    validate_profile_name(args.name)
    params = args.params.expanduser().resolve() if args.params else None
    if params is not None and not params.is_file():
        raise ProfileError(f"Parameter file not found: {params}")

    if args.swift_path is not None:
        path = Path(args.swift_path).expanduser().resolve()
        if not path.is_dir():
            raise ProfileError(f"SWIFT clone not found: {path}")
        # A subdirectory or another worktree of the clone means the clone
        try:
            repo = main_clone_of(path)
        except ProfileError:
            raise ProfileError(f"Not a git repository: {path}") from None
    else:
        repo = Path.cwd() / "swiftsim"
        if repo.exists():
            raise ProfileError(
                f"{repo} already exists; pass it with --swift-path to use it."
            )
        print(f"Cloning {args.swift_url} into {repo}...", flush=True)
        subprocess.run(["git", "clone", args.swift_url, str(repo)], check=True)

    root = (
        Path(args.worktree_root).expanduser().resolve()
        if args.worktree_root
        else default_worktree_root()
    )
    store = ProfileStore(repo=repo, worktree_root=root)
    add_profile(
        store,
        args.name,
        args.upstream,
        configure_flags=shlex.split(args.configure_flags),
        data_dir=repo.parent,
        parameter_file=params,
        # After starting over (~/.swiftsim-cli deleted) the clone still
        # has the profile's branch: pick it up again
        reuse_branch=True,
    )
    store.current = args.name
    save_store(store)
    print_summary(store, args.name)


def print_summary(store: ProfileStore, name: str) -> None:
    """Print the new profile, where everything lives and what to do next."""
    display_profile(name=name)
    home = profile_mod.PROFILE_FILE.parent
    rows = [
        ["profile settings", str(profile_mod.PROFILE_FILE)],
        ["SWIFT clone", str(store.repo)],
        ["profile worktrees", str(store.worktree_root)],
        ["caches", str(home / "cache" / name)],
    ]
    print()
    print(create_ascii_table(["What", "Where"], rows, title="LOCATIONS"))
    print(
        f"""
Created profile '{name}'. Next steps:
  swift-cli make -j <N>                    build it
  swift-cli config --list-options          list its configure options
  swift-cli profile --edit NAME            change these settings
  swift-cli new -o RUN --ics ICS.hdf5      set up a run directory
  swift-cli <mode> --help                  help for any mode"""
    )
