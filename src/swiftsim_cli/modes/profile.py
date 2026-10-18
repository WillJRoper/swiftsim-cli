"""Profile mode for managing SWIFT profiles (one git worktree each)."""

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from swiftsim_cli import profile as profile_mod
from swiftsim_cli.modes.common_args import (
    add_nr_threads_argument,
    existing_dir,
)
from swiftsim_cli.profile import (
    ProfileError,
    ProfileStore,
    SWIFTCLIProfile,
    add_profile,
    ask,
    confirm,
    fetch_for_upstream,
    get_profile,
    git,
    list_worktrees,
    load_store,
    read_raw_profile_file,
    ref_exists,
    save_store,
    validate_profile_name,
)
from swiftsim_cli.swiftsim_dir import build_profile, update_profile
from swiftsim_cli.utilities.io import ascii_art
from swiftsim_cli.utilities.messages import note, warn
from swiftsim_cli.utilities.tables import create_ascii_table

DEFAULT_UPSTREAM = "origin/master"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the profile mode."""
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "-i",
        "--init",
        action="store_true",
        help="Interactively set up the main SWIFT clone and first profile.",
    )
    action.add_argument(
        "-n",
        "--new",
        metavar="NAME",
        help="Create a profile with its own worktree and branch.",
    )
    action.add_argument(
        "-s",
        "--switch",
        metavar="NAME",
        help="Make NAME the current profile.",
    )
    action.add_argument(
        "-e",
        "--edit",
        metavar="NAME",
        help="Interactively edit a profile.",
    )
    action.add_argument(
        "-r",
        "--remove",
        metavar="NAME",
        help="Remove a profile and its worktree.",
    )
    action.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all profiles (the default action).",
    )
    action.add_argument(
        "--show",
        nargs="?",
        const="",
        metavar="NAME",
        help="Print the settings of profile NAME (default: the current one).",
    )
    action.add_argument(
        "-u",
        "--update",
        nargs="?",
        const="",
        metavar="NAME",
        help="Pull (rebase onto upstream, stashing uncommitted changes), "
        "reconfigure if needed and build a profile (default: the current "
        "one).",
    )
    action.add_argument(
        "--update-all",
        action="store_true",
        help="Update and build every profile, then print a summary.",
    )
    action.add_argument(
        "--migrate",
        action="store_true",
        help="Convert an old-format profile file to the worktree format.",
    )
    action.add_argument(
        "--adopt",
        type=existing_dir,
        metavar="DIR",
        help="Register an existing worktree of the SWIFT clone as a profile "
        "(requires --name).",
    )

    # Options modifying the actions above
    parser.add_argument(
        "--name",
        metavar="NAME",
        help="Profile name for --adopt.",
    )
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        metavar="BRANCH",
        help=f"Upstream branch for --new (default: {DEFAULT_UPSTREAM}).",
    )
    parser.add_argument(
        "--configure-flags",
        default="",
        metavar="FLAGS",
        help="Configure flags for --new, as one string, e.g. "
        '--configure-flags="--enable-debug --with-hydro=sphenix".',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --remove, delete the worktree even if it has untracked "
        "or modified files.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Also configure and build the profile created with --new.",
    )
    add_nr_threads_argument(
        parser,
        help="Number of make threads for --build, --update and --update-all "
        "(default: 1).",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the profile mode."""
    if args.init:
        init_profiles()
    elif args.new is not None:
        new_profile(
            args.new,
            upstream=args.upstream,
            configure_flags=shlex.split(args.configure_flags),
            build=args.build,
            nr_threads=args.nr_threads,
        )
    elif args.switch is not None:
        switch_profile(args.switch)
    elif args.edit is not None:
        edit_profile(args.edit)
    elif args.remove is not None:
        remove_profile(args.remove, force=args.force)
    elif args.show is not None:
        display_profile(name=args.show or None)
    elif args.update is not None:
        update_profile(get_profile(args.update or None), args.nr_threads)
    elif args.update_all:
        update_all_profiles(args.nr_threads)
    elif args.migrate:
        migrate_profiles()
    elif args.adopt is not None:
        if not args.name:
            raise ProfileError("--adopt requires --name NAME.")
        adopt_worktree(Path(args.adopt), args.name)
    else:
        list_profiles()


def _base_settings(store: ProfileStore) -> dict[str, Any]:
    """Per-user settings new profiles inherit from the current profile."""
    base = store.profiles.get(store.current or "")
    if base is None:
        return {}
    return {
        "data_dir": base.data_dir,
        "softening_coeff": base.softening_coeff,
        "softening_pivot_z": base.softening_pivot_z,
    }


def _path(text: str) -> Path:
    """Expand and resolve a user-supplied path."""
    return Path(text).expanduser().resolve()


def init_profiles() -> None:
    """Interactively set up the main SWIFT clone and a first profile.

    Existing profiles are always kept.
    """
    print("\nWelcome to SWIFTsim-CLI!\n")
    print("\n".join(ascii_art))
    print("\nLet's set up your profiles...\n")

    raw = read_raw_profile_file()
    existing: ProfileStore | None = None
    if raw is not None:
        if profile_mod.is_legacy_format(raw):
            raise ProfileError(
                f"{profile_mod.PROFILE_FILE} uses the old profile format. "
                "Run 'swift-cli profile --migrate' instead."
            )
        existing = load_store()
        names = ", ".join(existing.profiles) or "(none)"
        if not confirm(
            f"{profile_mod.PROFILE_FILE} already exists (profiles: {names}). "
            "Existing profiles will be kept. Continue?"
        ):
            print("Aborting.")
            return

    repo = _path(
        ask(
            "Main SWIFT clone: ",
            default=str(existing.repo) if existing else "",
            kind="dir",
            must_exist=True,
        )
    )
    git("rev-parse", "--git-dir", cwd=repo)  # fail early if not a repo
    root = _path(
        ask(
            "Worktree root: ",
            default=str(
                existing.worktree_root
                if existing
                else repo.parent / "swiftsim-profiles"
            ),
            kind="dir",
        )
    )
    name = ask("First profile name: ", default="default")
    upstream = ask("Upstream branch: ", default=DEFAULT_UPSTREAM)
    flags = shlex.split(ask("Configure flags: ", default=""))
    data_dir = _path(
        ask(
            "Extra data directory: ",
            default=str(repo.parent),
            kind="dir",
            must_exist=True,
        )
    )
    param = ask(
        "Parameter file (blank: <worktree>/examples/parameter_example.yml): ",
        kind="file",
    )
    softening_coeff = float(
        ask("Softening (in units of mean separation): ", default="0.04")
    )
    softening_pivot_z = float(
        ask("Maximal softening pivot redshift: ", default="2.7")
    )

    store = existing or ProfileStore(repo=repo, worktree_root=root)
    store.repo = repo
    store.worktree_root = root
    add_profile(
        store,
        name,
        upstream,
        configure_flags=flags,
        data_dir=data_dir,
        parameter_file=_path(param) if param else None,
        softening_coeff=softening_coeff,
        softening_pivot_z=softening_pivot_z,
    )
    store.current = name
    save_store(store)
    print(
        f"\nCreated profile '{name}'. Profiles saved to "
        f"{profile_mod.PROFILE_FILE}"
    )
    print(f"Build it with: swift-cli profile --update {name} -j <N>")


def new_profile(
    name: str,
    upstream: str = DEFAULT_UPSTREAM,
    configure_flags: list[str] | None = None,
    build: bool = False,
    nr_threads: int = 1,
) -> None:
    """Create a profile with its own branch and worktree.

    Args:
        name: The new profile name.
        upstream: The branch the profile follows.
        configure_flags: Flags for ``./configure``.
        build: Configure and build the new profile.
        nr_threads: Number of make threads when building.
    """
    store = load_store()
    profile = add_profile(
        store,
        name,
        upstream,
        configure_flags=configure_flags,
        **_base_settings(store),
    )
    if store.current is None:
        store.current = name
    save_store(store)
    print(f"Created profile '{name}' at {profile.worktree}")
    if build:
        build_profile(profile, nr_threads)


def switch_profile(name: str) -> None:
    """Make ``name`` the current profile."""
    store = load_store()
    profile = store.get(name)
    store.current = name
    save_store(store)
    print(f"Switched to profile '{name}' ({profile.worktree})")


def edit_profile(name: str) -> None:
    """Interactively edit a profile.

    Changing the upstream re-points the profile's branch at it; changing the
    configure flags makes the next update/config/make reconfigure.
    """
    store = load_store()
    p = store.get(name)
    print(f"Editing profile '{name}'. Press Enter to keep existing values.\n")
    upstream = ask("Upstream branch: ", default=p.upstream or "")
    flags = shlex.split(
        ask("Configure flags: ", default=shlex.join(p.configure_flags))
    )
    data_dir = ask(
        "Extra data directory: ",
        default=str(p.data_dir or ""),
        kind="dir",
        must_exist=True,
    )
    param = ask(
        "Parameter file: ",
        default=str(p.parameter_file or ""),
        kind="file",
    )
    softening_coeff = float(
        ask(
            "Softening (in units of mean separation): ",
            default=str(p.softening_coeff),
        )
    )
    softening_pivot_z = float(
        ask(
            "Maximal softening pivot redshift: ",
            default=str(p.softening_pivot_z),
        )
    )

    if upstream and upstream != p.upstream:
        fetch_for_upstream(store.repo, upstream)
        git(
            "branch",
            f"--set-upstream-to={upstream}",
            *([p.branch] if p.branch else []),
            cwd=p.worktree,
        )
        p.upstream = upstream
    if flags != p.configure_flags:
        print(
            "Configure flags changed: the next update/config/make will "
            "reconfigure."
        )
    p.configure_flags = flags
    p.data_dir = _path(data_dir) if data_dir else None
    p.parameter_file = _path(param) if param else None
    p.softening_coeff = softening_coeff
    p.softening_pivot_z = softening_pivot_z
    save_store(store)
    print(f"\nProfile '{name}' updated.")


def remove_profile(name: str, force: bool = False) -> None:
    """Remove a profile, its worktree and (optionally) its branch.

    Args:
        name: The profile to remove.
        force: Remove the worktree even if it has untracked or modified
            files (``git worktree remove --force``).
    """
    store = load_store()
    p = store.get(name)
    if name == store.current:
        raise ProfileError(
            f"'{name}' is the current profile. Switch to another one first "
            "('swift-cli profile --switch OTHER')."
        )
    if not confirm(f"Remove profile '{name}' and its worktree {p.worktree}?"):
        print("Aborting.")
        return

    if p.worktree is not None and p.worktree.exists():
        force_args = ["--force"] if force else []
        git("worktree", "remove", *force_args, str(p.worktree), cwd=store.repo)
    else:
        git("worktree", "prune", cwd=store.repo)
    del store.profiles[name]
    save_store(store)
    shutil.rmtree(profile_mod.PROFILE_FILE.parent / "cache" / name, True)
    print(f"Removed profile '{name}'.")

    if p.branch and confirm(
        f"Also delete local branch '{p.branch}'? Unpushed commits on it "
        "will be lost."
    ):
        git("branch", "-D", p.branch, cwd=store.repo)
        print(f"Deleted branch '{p.branch}'.")


def list_profiles() -> None:
    """List all profiles, marking the current one."""
    store = load_store()
    if not store.profiles:
        print("No profiles found. Create one with 'swift-cli profile --new'.")
        return
    rows = [
        [
            ("* " if name == store.current else "  ") + name,
            p.upstream or "-",
            str(p.worktree),
            "yes" if p.is_built else "no",
        ]
        for name, p in store.profiles.items()
    ]
    print(
        create_ascii_table(
            ["Profile", "Upstream", "Worktree", "Built"],
            rows,
            title=f"PROFILES (repo: {store.repo})",
        )
    )


def display_profile(
    print_header: bool = True,
    title: str | None = None,
    show_profile: bool = True,
    show_cosmology: bool = True,
    name: str | None = None,
) -> None:
    """Display a profile (default: the active one). Has no side effects.

    Args:
        print_header: Whether to print the ASCII art header.
        title: Title above the profile table (default: profile name).
        show_profile: Whether to display the profile information.
        show_cosmology: Whether to display the parameter file's cosmology.
        name: The profile to show.
    """
    store = load_store()
    p = store.get(name or profile_mod.active_profile_name(store))

    if print_header:
        print()
        print("\n".join(ascii_art))
        print()

    if show_profile:
        configured = (
            "(never)"
            if p.configured_flags is None
            else shlex.join(p.configured_flags) or "(none)"
        )
        rows = [
            ["current", "yes" if p.name == store.current else "no"],
            ["repo", str(store.repo)],
            ["worktree", str(p.worktree)],
            ["branch", str(p.branch)],
            ["upstream", str(p.upstream)],
            ["configure_flags", shlex.join(p.configure_flags) or "(none)"],
            ["configured_flags", configured],
            ["built", "yes" if p.is_built else "no"],
            ["data_dir", str(p.data_dir)],
            ["parameter_file", str(p.parameter_file)],
            ["softening_coeff", str(p.softening_coeff)],
            ["softening_pivot_z", str(p.softening_pivot_z)],
        ]
        print(
            create_ascii_table(
                ["Key", "Value"], rows, title=title or f"PROFILE: {p.name}"
            )
        )

    if show_cosmology and p.parameter_file and p.parameter_file.exists():
        display_cosmology(p.parameter_file)


def display_cosmology(parameter_file: Path) -> None:
    """Display the Cosmology section of a SWIFT parameter file."""
    params = YAML(typ="safe").load(parameter_file.read_text().expandtabs(4))
    cosmology = (params or {}).get("Cosmology")
    if not cosmology:
        return
    rows = [[k, str(v)] for k, v in cosmology.items()]
    print()
    print(
        create_ascii_table(
            ["Parameter", "Value"], rows, title="PARAMETER FILE COSMOLOGY"
        )
    )


def update_all_profiles(nr_threads: int = 1) -> None:
    """Update and build every profile, continuing past failures.

    Raises:
        ProfileError: After printing the summary, if any profile failed.
    """
    store = load_store()
    rows = []
    failed = []
    for name, p in store.profiles.items():
        try:
            update_profile(p, nr_threads)
            rows.append([name, "ok"])
        except (ProfileError, OSError) as e:
            print(f"\n{e}\n")
            failed.append(name)
            rows.append([name, "FAILED: " + str(e).splitlines()[0]])
    print(create_ascii_table(["Profile", "Outcome"], rows, title="UPDATE"))
    if failed:
        raise ProfileError(
            f"{len(failed)} profile(s) failed to update: {', '.join(failed)}"
        )


def _resolve_legacy_upstream(repo: Path, branch: str) -> str:
    """Pick the upstream for a migrated branch (prefer origin/<branch>)."""
    if ref_exists(repo, f"refs/remotes/origin/{branch}"):
        return f"origin/{branch}"
    if ref_exists(repo, f"refs/heads/{branch}"):
        return branch
    return f"origin/{branch}"


# Timer databases of the old log analysis, superseded by the timing
# package's cache (cache/<profile>/timer_db/).
LEGACY_TIMER_FILES = ("timers.yaml", "timer_nesting.yaml")


def _legacy_timer_files() -> list[Path]:
    """Old timer databases in the profile dir root and cache/<profile>/."""
    base = profile_mod.PROFILE_FILE.parent
    cache = base / "cache"
    dirs = [base]
    if cache.is_dir():
        dirs += sorted(p for p in cache.iterdir() if p.is_dir())
    return [
        d / name
        for d in dirs
        for name in LEGACY_TIMER_FILES
        if (d / name).is_file()
    ]


def migrate_profiles() -> None:
    """Convert an old-format profile file into version 2.

    Each old named profile (``Default`` becomes ``default``; ``Current`` is
    skipped) gets its own worktree tracking ``origin/<old branch>``. The old
    file is backed up to ``profiles.yaml.bak``. Old timer databases
    (``timers.yaml``, ``timer_nesting.yaml``) are deleted.
    """
    raw = read_raw_profile_file()
    if raw is None:
        raise ProfileError(
            "No profile file to migrate. Run 'swift-cli profile --init'."
        )
    if not profile_mod.is_legacy_format(raw):
        print("Profile file is already in the current format.")
        return

    old_current = raw.get("Current") or {}
    named = {
        ("default" if key == "Default" else key): entry
        for key, entry in raw.items()
        if key != "Current" and isinstance(entry, dict)
    }
    if not named:
        raise ProfileError("No named profiles found to migrate.")
    old_repo = old_current.get("swiftsim_dir") or next(
        iter(named.values())
    ).get("swiftsim_dir")
    if not old_repo:
        raise ProfileError("Could not find 'swiftsim_dir' in the old file.")
    repo = Path(old_repo)
    root = repo.parent / "swiftsim-profiles"

    fetch_for_upstream(repo, "origin/master")  # fetches origin, if any
    plan = {
        name: _resolve_legacy_upstream(repo, entry.get("branch") or "master")
        for name, entry in named.items()
    }
    current = next(
        (
            name
            for name, entry in named.items()
            if entry.get("swiftsim_dir") == old_current.get("swiftsim_dir")
            and entry.get("branch") == old_current.get("branch")
        ),
        "default" if "default" in named else next(iter(named)),
    )

    print(f"Main SWIFT clone: {repo}\nWorktree root:    {root}\n")
    rows = [
        [name, upstream, str(root / name), "yes" if name == current else ""]
        for name, upstream in plan.items()
    ]
    print(
        create_ascii_table(
            ["Profile", "Upstream", "Worktree", "Current"],
            rows,
            title="MIGRATION PLAN",
        )
    )
    if old_current and not any(
        entry.get("branch") == old_current.get("branch")
        for entry in named.values()
    ):
        note(
            f"the old 'Current' profile (branch "
            f"{old_current.get('branch')}) matches no named profile and will "
            "not be migrated; add it later with 'swift-cli profile --new'."
        )
    legacy = _legacy_timer_files()
    if legacy:
        print("Old timer databases to delete (no longer used):")
        for path in legacy:
            print(f"  {path}")
    if not confirm("Proceed with the migration?"):
        print("Aborting.")
        return

    backup = profile_mod.PROFILE_FILE.with_suffix(".yaml.bak")
    shutil.copy2(profile_mod.PROFILE_FILE, backup)
    print(f"Backed up old profile file to {backup}")
    for path in legacy:
        path.unlink()
    if legacy:
        print(f"Deleted {len(legacy)} old timer database file(s).")

    store = ProfileStore(repo=repo, worktree_root=root)
    failed = []
    for name, upstream in plan.items():
        entry = named[name]
        param = (
            entry.get("parameter_file")
            or entry.get("template_parameter_file")
            or entry.get("template_params")
        )
        param_path = Path(param) if param else None
        if param_path is not None and param_path.is_relative_to(repo):
            # Point at the same file inside the new worktree
            param_path = root / name / param_path.relative_to(repo)
        try:
            add_profile(
                store,
                name,
                upstream,
                data_dir=Path(entry["data_dir"])
                if entry.get("data_dir")
                else None,
                parameter_file=param_path,
                softening_coeff=float(entry.get("softening_coeff", 0.04)),
                softening_pivot_z=float(entry.get("softening_pivot_z", 2.7)),
            )
        except ProfileError as e:
            warn(f"could not migrate '{name}': {e}")
            failed.append(name)
    store.current = (
        current
        if current in store.profiles
        else next(iter(store.profiles), None)
    )
    save_store(store)
    print(f"Migrated {len(store.profiles)} profile(s).")
    if failed:
        raise ProfileError(
            f"Failed to migrate: {', '.join(failed)} (see above; the old file "
            f"is at {backup})."
        )


def _read_configured_flags(worktree: Path) -> list[str] | None:
    """Read the flags an existing build was configured with, if any."""
    if not (worktree / "config.status").exists():
        return None
    result = subprocess.run(
        ["./config.status", "--config"],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    return shlex.split(result.stdout) if result.returncode == 0 else None


def adopt_worktree(path: Path, name: str) -> None:
    """Register an existing worktree of the main clone as a profile.

    The worktree's checked-out branch becomes the profile's branch and that
    branch's upstream (if any) its upstream. If the worktree is already
    configured, its configure flags are recorded so it is not reconfigured.
    """
    store = load_store()
    validate_profile_name(name)
    if name in store.profiles:
        raise ProfileError(f"Profile '{name}' already exists.")
    path = _path(str(path))
    worktrees = list_worktrees(store.repo)
    if path not in worktrees:
        known = "\n  ".join(str(w) for w in worktrees)
        raise ProfileError(
            f"{path} is not a worktree of {store.repo}. Known worktrees:\n"
            f"  {known}"
        )
    branch = worktrees[path]
    if branch is None:
        raise ProfileError(
            f"{path} has a detached HEAD; check out a branch first."
        )
    try:
        upstream: str | None = git(
            "rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}", cwd=path
        )
    except ProfileError:
        upstream = None
    flags = _read_configured_flags(path)

    store.profiles[name] = SWIFTCLIProfile(
        name=name,
        worktree=path,
        branch=branch,
        upstream=upstream,
        configure_flags=list(flags or []),
        configured_flags=flags,
        parameter_file=path / "examples" / "parameter_example.yml",
        **_base_settings(store),
    )
    if store.current is None:
        store.current = name
    save_store(store)
    print(
        f"Adopted {path} as profile '{name}' (branch {branch}, upstream "
        f"{upstream or 'none'})."
    )
