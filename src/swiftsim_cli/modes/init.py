"""Init mode: set up swift-cli in one command.

Uses an existing SWIFT clone (``--swift-path``) or clones SWIFT into
``./swiftsim``, then creates the first profile (a worktree of that clone).
Later profiles are added with 'swift-cli profile --new NAME'.
"""

import argparse
import shlex
import subprocess
from pathlib import Path

from swiftsim_cli import profile as profile_mod
from swiftsim_cli.modes.profile import DEFAULT_UPSTREAM
from swiftsim_cli.profile import (
    ProfileError,
    ProfileStore,
    add_profile,
    default_worktree_root,
    git,
    load_store,
    read_raw_profile_file,
    save_store,
    validate_profile_name,
)

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

    if args.swift_path is not None:
        repo = Path(args.swift_path).expanduser().resolve()
        if not repo.is_dir():
            raise ProfileError(f"SWIFT clone not found: {repo}")
        git("rev-parse", "--git-dir", cwd=repo)  # fail early if not a repo
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
    )
    store.current = args.name
    save_store(store)
    print(
        f"\nCreated profile '{args.name}'. Profiles saved to "
        f"{profile_mod.PROFILE_FILE}"
    )
    print("Build it with: swift-cli make -j <N>")
