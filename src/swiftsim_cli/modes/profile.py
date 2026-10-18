"""Profile mode for managing SWIFT profiles (one git worktree each)."""

import argparse
import functools
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
    check_new_profile_name,
    confirm,
    fetch_for_upstream,
    get_profile,
    git,
    list_worktrees,
    load_store,
    main_clone_of,
    read_raw_profile_file,
    ref_exists,
    save_store,
)
from swiftsim_cli.swiftsim_dir import (
    build_profile,
    get_swiftsim_dir,
    update_profile,
)
from swiftsim_cli.utilities.io import ascii_art
from swiftsim_cli.utilities.messages import note, warn
from swiftsim_cli.utilities.tables import create_ascii_table

DEFAULT_UPSTREAM = "origin/master"

# Profile settings set by --init/--edit prompts or --set KEY=VALUE, in
# prompt order, with their prompt labels and kinds
SETTINGS = {
    "upstream": ("Upstream branch: ", "text"),
    "configure_flags": ("Configure flags: ", "text"),
    "data_dir": ("Extra data directory: ", "dir"),
    "parameter_file": (
        "Parameter file (blank: <worktree>/examples/parameter_example.yml): ",
        "file",
    ),
    "softening_coeff": ("Softening (in units of mean separation): ", "text"),
    "softening_pivot_z": ("Maximal softening pivot redshift: ", "text"),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the profile mode."""
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "-i",
        "--init",
        action="store_true",
        help="Set up the main SWIFT clone and first profile (interactive "
        "unless --repo is given).",
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
        help="Edit a profile (interactive unless --set is given).",
    )
    action.add_argument(
        "-r",
        "--remove",
        metavar="NAME",
        help="Remove a profile and its worktree (worktrees not created by "
        "swift-cli, e.g. adopted ones, are kept).",
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
        help="Profile name for --adopt, and the first profile's name for "
        "--init (default: default).",
    )
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        metavar="BRANCH",
        help=f"Upstream branch for --new and --init (default: "
        f"{DEFAULT_UPSTREAM}).",
    )
    parser.add_argument(
        "--configure-flags",
        default="",
        metavar="FLAGS",
        help="Configure flags for --new and --init, as one string, e.g. "
        '--configure-flags="--enable-debug --with-hydro=sphenix".',
    )
    parser.add_argument(
        "--repo",
        metavar="DIR",
        help="Main SWIFT clone for --init; makes --init non-interactive.",
    )
    parser.add_argument(
        "--worktree-root",
        metavar="DIR",
        help="Directory profile worktrees are created in, for --init and "
        "--migrate (default: swiftsim-profiles next to the main clone).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set a profile setting without prompting, for --edit and "
        f"--init (repeatable). KEY is one of: {', '.join(SETTINGS)}.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Answer yes to the confirmation of --remove, --migrate and "
        "--init (--remove then keeps the profile's branch).",
    )
    parser.add_argument(
        "--reuse-branch",
        action="store_true",
        help="With --new, use the profile's branch (swift-cli/NAME) if it "
        "already exists, e.g. kept from a removed profile.",
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
    settings = parse_settings(args.set)
    if args.init:
        init_profiles(
            repo=args.repo,
            worktree_root=args.worktree_root,
            name=args.name,
            upstream=args.upstream,
            configure_flags=args.configure_flags,
            settings=settings,
            yes=args.yes,
        )
    elif args.new is not None:
        new_profile(
            args.new,
            upstream=args.upstream,
            configure_flags=shlex.split(args.configure_flags),
            build=args.build,
            nr_threads=args.nr_threads,
            reuse_branch=args.reuse_branch,
        )
    elif args.switch is not None:
        switch_profile(args.switch)
    elif args.edit is not None:
        edit_profile(args.edit, settings=settings or None)
    elif args.remove is not None:
        remove_profile(args.remove, force=args.force, yes=args.yes)
    elif args.show is not None:
        display_profile(name=args.show or None)
    elif args.update is not None:
        update_profile(get_profile(args.update or None), args.nr_threads)
    elif args.update_all:
        update_all_profiles(args.nr_threads)
    elif args.migrate:
        migrate_profiles(yes=args.yes, worktree_root=args.worktree_root)
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


def parse_settings(items: list[str]) -> dict[str, str]:
    """Parse ``--set KEY=VALUE`` options into a validated dict.

    Raises:
        ProfileError: For an unknown key, a missing '=' or a bad value.
    """
    settings = {}
    for item in items:
        key, sep, value = item.partition("=")
        key = key.strip().replace("-", "_")
        if not sep or key not in SETTINGS:
            raise ProfileError(
                f"--set expects KEY=VALUE with KEY one of: "
                f"{', '.join(SETTINGS)} (got {item!r})."
            )
        check_setting(key, value.strip())
        settings[key] = value.strip()
    return settings


def check_setting(key: str, text: str) -> None:
    """Check a profile setting given as text.

    Raises:
        ProfileError: If the value is unusable.
    """
    if key in ("softening_coeff", "softening_pivot_z"):
        try:
            float(text)
        except ValueError:
            raise ProfileError(
                f"{key} must be a number, got {text!r}."
            ) from None
    elif key in ("data_dir", "parameter_file"):
        if text and not Path(text).expanduser().exists():
            raise ProfileError(f"{key}: path does not exist: {text}")
    elif key == "configure_flags":
        try:
            shlex.split(text)
        except ValueError as e:
            raise ProfileError(f"configure_flags: {e}") from None


def _ask_settings(defaults: dict[str, str]) -> dict[str, str]:
    """Prompt for every profile setting, checking each answer."""
    values = {}
    for key, (label, kind) in SETTINGS.items():
        values[key] = ask(
            label,
            default=defaults[key],
            kind=kind,
            check=functools.partial(check_setting, key),
        )
    return values


def _apply_settings(p: SWIFTCLIProfile, values: dict[str, str]) -> None:
    """Store checked settings (except the upstream) on a profile."""
    if "configure_flags" in values:
        flags = shlex.split(values["configure_flags"])
        if flags != p.configure_flags:
            print(
                "Configure flags changed: the next update/config/make will "
                "reconfigure."
            )
        p.configure_flags = flags
    if "data_dir" in values:
        p.data_dir = _path(values["data_dir"]) if values["data_dir"] else None
    if "parameter_file" in values:
        param = values["parameter_file"]
        p.parameter_file = (
            _path(param)
            if param
            else (p.worktree / "examples" / "parameter_example.yml")
            if p.worktree
            else None
        )
    if "softening_coeff" in values:
        p.softening_coeff = float(values["softening_coeff"])
    if "softening_pivot_z" in values:
        p.softening_pivot_z = float(values["softening_pivot_z"])


def init_profiles(
    repo: str | None = None,
    worktree_root: str | None = None,
    name: str | None = None,
    upstream: str = DEFAULT_UPSTREAM,
    configure_flags: str = "",
    settings: dict[str, str] | None = None,
    yes: bool = False,
) -> None:
    """Set up the main SWIFT clone and a first profile.

    Interactive unless ``repo`` is given, in which case the other arguments
    (or their defaults) are used without prompting. Existing profiles are
    always kept, and must all use the same main clone.

    Args:
        repo: The main SWIFT clone (makes this non-interactive).
        worktree_root: Where profile worktrees are created.
        name: The first profile's name (default: "default").
        upstream: The first profile's upstream branch.
        configure_flags: The first profile's configure flags, as one string.
        settings: Further settings (``--set``), already checked.
        yes: Skip the confirmation when a profile file already exists.
    """
    interactive = repo is None
    if interactive:
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
        if not (
            yes
            or confirm(
                f"{profile_mod.PROFILE_FILE} already exists (profiles: "
                f"{names}). Existing profiles will be kept. Continue?"
            )
        ):
            print("Aborting.")
            return

    # Every profile is a worktree of one clone: it can't change under them
    locked = existing is not None and bool(existing.profiles)
    if locked and existing is not None:
        if repo is not None and _path(repo) != existing.repo.resolve():
            raise ProfileError(
                f"Existing profiles ({', '.join(existing.profiles)}) are "
                f"worktrees of {existing.repo}; the main clone can't change "
                "while they exist. Remove them first, or use that clone."
            )
        clone = existing.repo
        print(f"Main SWIFT clone: {clone} (used by the existing profiles)")
    elif interactive:
        clone = _path(
            ask(
                "Main SWIFT clone: ",
                default=str(existing.repo) if existing else "",
                kind="dir",
                must_exist=True,
            )
        )
    else:
        clone = _path(str(repo))
    if not clone.is_dir():
        raise ProfileError(f"Main SWIFT clone not found: {clone}")
    git("rev-parse", "--git-dir", cwd=clone)  # fail early if not a repo

    default_root = str(
        existing.worktree_root
        if existing
        else clone.parent / "swiftsim-profiles"
    )
    store = existing or ProfileStore(repo=clone, worktree_root=Path())
    store.repo = clone
    defaults = {
        "upstream": upstream,
        "configure_flags": configure_flags,
        "data_dir": str(clone.parent),
        "parameter_file": "",
        "softening_coeff": "0.04",
        "softening_pivot_z": "2.7",
        **(settings or {}),
    }
    if interactive:
        root = ask(
            "Worktree root: ",
            default=worktree_root or default_root,
            kind="dir",
        )
        name = ask(
            "First profile name: ",
            default=name or "default",
            check=lambda text: check_new_profile_name(store, text),
        )
        values = _ask_settings(defaults)
    else:
        root = worktree_root or default_root
        name = name or "default"
        values = defaults
        for key, value in values.items():
            check_setting(key, value)
    store.worktree_root = _path(root)

    add_profile(
        store,
        name,
        values["upstream"],
        configure_flags=shlex.split(values["configure_flags"]),
        data_dir=_path(values["data_dir"]) if values["data_dir"] else None,
        parameter_file=(
            _path(values["parameter_file"])
            if values["parameter_file"]
            else None
        ),
        softening_coeff=float(values["softening_coeff"]),
        softening_pivot_z=float(values["softening_pivot_z"]),
    )
    store.current = name
    save_store(store)
    print(
        f"\nCreated profile '{name}'. Profiles saved to "
        f"{profile_mod.PROFILE_FILE}"
    )
    print("Build it with: swift-cli make -j <N>")


def new_profile(
    name: str,
    upstream: str = DEFAULT_UPSTREAM,
    configure_flags: list[str] | None = None,
    build: bool = False,
    nr_threads: int = 1,
    reuse_branch: bool = False,
) -> None:
    """Create a profile with its own branch and worktree.

    Args:
        name: The new profile name.
        upstream: The branch the profile follows.
        configure_flags: Flags for ``./configure``.
        build: Configure and build the new profile.
        nr_threads: Number of make threads when building.
        reuse_branch: Use an existing ``swift-cli/<name>`` branch.
    """
    store = load_store()
    profile = add_profile(
        store,
        name,
        upstream,
        configure_flags=configure_flags,
        reuse_branch=reuse_branch,
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


def _change_upstream(p: SWIFTCLIProfile, upstream: str) -> None:
    """Make a profile follow a new upstream branch.

    If the profile's branch has no commits of its own on top of the old
    upstream, it is moved to the new upstream (refusing if the worktree has
    uncommitted changes); otherwise it keeps its history and only its
    tracking changes, with a warning saying how to move its commits.

    Raises:
        ProfileError: For an unknown upstream or a dirty worktree.
    """
    wt = get_swiftsim_dir(p.worktree)
    fetch_for_upstream(wt, upstream)
    if not ref_exists(wt, upstream):
        raise ProfileError(f"Unknown upstream branch '{upstream}'.")
    old = p.upstream
    own = (
        int(git("rev-list", "--count", f"{old}..HEAD", cwd=wt))
        if old and ref_exists(wt, old)
        else None
    )
    if own == 0:
        if git("status", "--porcelain", "--untracked-files=no", cwd=wt):
            raise ProfileError(
                f"{wt} has uncommitted changes; commit or stash them so "
                f"branch {p.branch} can be moved to {upstream}."
            )
        git("reset", "--hard", "--quiet", upstream, cwd=wt)
        print(f"Moved branch {p.branch} to {upstream}.")
    git(
        "branch",
        f"--set-upstream-to={upstream}",
        *([p.branch] if p.branch else []),
        cwd=wt,
    )
    if own:
        warn(
            f"branch {p.branch} has {own} commit(s) of its own on top of "
            f"{old}; it now tracks {upstream} but still contains {old}'s "
            f"history. To move just your commits: "
            f"git -C {wt} rebase --onto {upstream} {old}"
        )
    elif own is None:
        warn(
            f"could not compare branch {p.branch} with its old upstream, so "
            f"it was not moved (it now tracks {upstream}). To move it: "
            f"git -C {wt} reset --hard {upstream}"
        )
    p.upstream = upstream


def edit_profile(name: str, settings: dict[str, str] | None = None) -> None:
    """Edit a profile, interactively unless ``settings`` is given.

    Changing the upstream moves the profile's branch to it (if the branch
    has no commits of its own); changing the configure flags makes the next
    update/config/make reconfigure.

    Args:
        name: The profile to edit.
        settings: Settings to change (``--set``), already checked; only
            these are changed and nothing is prompted for.
    """
    store = load_store()
    p = store.get(name)
    if settings is None:
        print(
            f"Editing profile '{name}'. Press Enter to keep existing values.\n"
        )
        values = _ask_settings(
            {
                "upstream": p.upstream or "",
                "configure_flags": shlex.join(p.configure_flags),
                "data_dir": str(p.data_dir or ""),
                "parameter_file": str(p.parameter_file or ""),
                "softening_coeff": str(p.softening_coeff),
                "softening_pivot_z": str(p.softening_pivot_z),
            }
        )
    else:
        values = settings

    upstream = values.get("upstream")
    if upstream and upstream != p.upstream:
        _change_upstream(p, upstream)
    _apply_settings(p, values)
    save_store(store)
    print(f"\nProfile '{name}' updated.")


def remove_profile(name: str, force: bool = False, yes: bool = False) -> None:
    """Remove a profile, its worktree and (optionally) its branch.

    Only worktrees swift-cli created are deleted: an adopted worktree (one
    outside the worktree root), the main clone, or a worktree another
    profile also uses is just unregistered.

    Args:
        name: The profile to remove.
        force: Remove the worktree even if it has untracked or modified
            files (``git worktree remove --force``).
        yes: Skip the confirmation (the branch is then kept).
    """
    store = load_store()
    p = store.get(name)
    if name == store.current:
        raise ProfileError(
            f"'{name}' is the current profile. Switch to another one first "
            "('swift-cli profile --switch OTHER')."
        )
    exists = p.worktree is not None and p.worktree.exists()
    repo = main_clone_of(p.worktree) if exists and p.worktree else store.repo
    keep_reason = None
    if exists and p.worktree is not None:
        wt = p.worktree.resolve()
        sharing = [
            n
            for n, o in store.profiles.items()
            if n != name and o.worktree and o.worktree.resolve() == wt
        ]
        if wt == repo:
            keep_reason = "it is the main SWIFT clone"
        elif sharing:
            keep_reason = f"profile '{sharing[0]}' uses it too"
        elif not wt.is_relative_to(store.worktree_root.resolve()):
            keep_reason = "it was not created by swift-cli"

    if keep_reason:
        question = (
            f"Remove profile '{name}'? Its worktree {p.worktree} is kept "
            f"({keep_reason})."
        )
    else:
        question = f"Remove profile '{name}' and its worktree {p.worktree}?"
    if not (yes or confirm(question)):
        print("Aborting.")
        return

    if keep_reason is None:
        if exists:
            force_args = ["--force"] if force else []
            git("worktree", "remove", *force_args, str(p.worktree), cwd=repo)
        else:
            git("worktree", "prune", cwd=repo)
    del store.profiles[name]
    save_store(store)
    shutil.rmtree(profile_mod.PROFILE_FILE.parent / "cache" / name, True)
    print(f"Removed profile '{name}'.")

    # Only offer to delete a branch swift-cli made and no worktree uses
    if (
        keep_reason is None
        and p.branch
        and p.branch.startswith(profile_mod.BRANCH_PREFIX)
    ):
        delete_cmd = f"git -C {repo} branch -D {p.branch}"
        if yes:
            note(f"kept branch '{p.branch}'; delete it with '{delete_cmd}'.")
        elif confirm(
            f"Also delete local branch '{p.branch}'? Unpushed commits on it "
            "will be lost."
        ):
            git("branch", "-D", p.branch, cwd=repo)
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

    if not show_cosmology:
        return
    if p.parameter_file and p.parameter_file.is_file():
        display_cosmology(p.parameter_file)
        return
    problem = (
        f"profile '{p.name}' has no parameter file"
        if p.parameter_file is None
        else f"parameter file not found: {p.parameter_file}"
    )
    if not show_profile:  # the cosmology was all that was asked for
        raise ProfileError(
            f"Can't show the cosmology: {problem}. Set one with "
            f"'swift-cli profile --edit {p.name} --set parameter_file=PATH'."
        )
    if p.parameter_file is not None:
        note(f"{problem}; no cosmology to show.")


def display_cosmology(parameter_file: Path) -> None:
    """Display the Cosmology section of a SWIFT parameter file.

    The file is read like any parameter file (a duplicate key is a warning,
    not an error).
    """
    from swiftsim_cli.params import load_parameters

    cosmology = load_parameters(parameter_file).get("Cosmology")
    if not cosmology:
        note(f"{parameter_file} has no Cosmology section.")
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


# Keys of the old profile format that are migrated; others are reported
LEGACY_KEYS = {
    "swiftsim_dir",
    "branch",
    "data_dir",
    "parameter_file",
    "template_parameter_file",
    "template_params",
    "softening_coeff",
    "softening_pivot_z",
}


def _other_files(skip: list[Path]) -> list[Path]:
    """Files in the profile directory that swift-cli does not use."""
    base = profile_mod.PROFILE_FILE.parent
    used = {profile_mod.PROFILE_FILE.name, "profiles.yaml.bak", "cache"}
    return sorted(
        path
        for path in base.iterdir()
        if path.name not in used and path not in skip
    )


def migrate_profiles(
    yes: bool = False, worktree_root: str | None = None
) -> None:
    """Convert an old-format profile file into version 2.

    Each old named profile (``Default`` becomes ``default``) gets its own
    worktree tracking ``origin/<old branch>``. An old ``Current`` profile on
    a branch no named profile uses becomes the profile ``current``. Nothing
    is changed unless at least one profile migrates: then the old file is
    backed up to ``profiles.yaml.bak`` and old timer databases
    (``timers.yaml``, ``timer_nesting.yaml``) are deleted. Rerunning after
    an interrupted migration reuses the worktrees already created.

    Args:
        yes: Skip the confirmation.
        worktree_root: Where to create the worktrees (default:
            ``swiftsim-profiles`` next to the main clone).
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
        ("default" if key == "Default" else str(key)): entry
        for key, entry in raw.items()
        if key != "Current" and isinstance(entry, dict)
    }
    if (
        old_current
        and "current" not in named
        and not any(
            entry.get("branch") == old_current.get("branch")
            for entry in named.values()
        )
    ):
        named["current"] = old_current
    if not named:
        raise ProfileError("No named profiles found to migrate.")
    old_repo = old_current.get("swiftsim_dir") or next(
        iter(named.values())
    ).get("swiftsim_dir")
    if not old_repo:
        raise ProfileError("Could not find 'swiftsim_dir' in the old file.")
    repo = Path(old_repo)
    root = (
        _path(worktree_root)
        if worktree_root
        else repo.parent / "swiftsim-profiles"
    )

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
    if "current" in named and named["current"] is old_current:
        note(
            f"the old 'Current' profile (branch {old_current.get('branch')}) "
            "matches no named profile, so it becomes the profile 'current'."
        )
    dropped = {
        name: sorted(str(k) for k in set(entry) - LEGACY_KEYS)
        for name, entry in named.items()
    }
    dropped = {name: keys for name, keys in dropped.items() if keys}
    if dropped:
        note(
            "these old settings are no longer stored in profiles and will "
            "not be migrated (the cosmology now comes from the parameter "
            "file): "
            + "; ".join(f"{n}: {', '.join(k)}" for n, k in dropped.items())
        )
    legacy = _legacy_timer_files()
    if legacy:
        print("Old timer databases to delete (no longer used):")
        for path in legacy:
            print(f"  {path}")
    others = _other_files(legacy)
    if others:
        note(
            f"other files in {profile_mod.PROFILE_FILE.parent} are not used "
            "by swift-cli and are left alone: "
            + ", ".join(path.name for path in others)
        )
    if not (yes or confirm("Proceed with the migration?")):
        print("Aborting.")
        return

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
                reuse_branch=True,
            )
        except (ProfileError, ValueError) as e:
            warn(f"could not migrate '{name}': {e}")
            failed.append(name)
    if not store.profiles:
        raise ProfileError(
            "No profile could be migrated (see above); nothing was changed."
        )

    backup = profile_mod.PROFILE_FILE.with_suffix(".yaml.bak")
    shutil.copy2(profile_mod.PROFILE_FILE, backup)
    print(f"Backed up old profile file to {backup}")
    store.current = (
        current if current in store.profiles else next(iter(store.profiles))
    )
    save_store(store)
    print(f"Migrated {len(store.profiles)} profile(s).")
    for path in legacy:
        path.unlink()
    if legacy:
        print(f"Deleted {len(legacy)} old timer database file(s).")
    if failed:
        raise ProfileError(
            f"Failed to migrate: {', '.join(failed)} (see above; the old file "
            f"is at {backup}). Add them with 'swift-cli profile --new'."
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
    check_new_profile_name(store, name)
    path = _path(str(path))
    for other, o in store.profiles.items():
        if o.worktree and o.worktree.resolve() == path:
            raise ProfileError(
                f"{path} is already the worktree of profile '{other}'."
            )
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
