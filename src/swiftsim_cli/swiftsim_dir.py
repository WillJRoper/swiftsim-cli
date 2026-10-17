"""Tools for working with a profile's SWIFT worktree.

Every mode that touches SWIFT source (config, make, update, new, ...) acts on
the active profile's worktree, resolved by `get_swiftsim_dir`.
"""

import difflib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from swiftsim_cli.profile import (
    ProfileError,
    SWIFTCLIProfile,
    get_profile,
    get_profile_cache_dir,
    git,
    update_profile_fields,
)
from swiftsim_cli.utilities import run_command_in_dir

# Matches an option at the start of a `./configure --help` line, e.g.
# "  --enable-debug=[yes/no]" or "  -h, --help". Placeholder entries such as
# "--enable-FEATURE[=ARG]" do not match (the name must end at a delimiter).
_CONFIGURE_OPTION_RE = re.compile(
    r"^\s+(?:-\w, )?(--[a-z0-9][a-z0-9_-]*[a-z0-9])(?=[\s=\[,]|$)"
)

# autoconf accepts the complement of every enable/with option
_COMPLEMENTS = (
    ("--enable-", "--disable-"),
    ("--disable-", "--enable-"),
    ("--with-", "--without-"),
    ("--without-", "--with-"),
)


def get_swiftsim_dir(swift_dir: Path | None = None) -> Path:
    """Get the SWIFT source directory.

    Args:
        swift_dir: Optional explicit path. If None, uses the active profile's
            worktree (the current profile, or the one given with the global
            ``--profile`` option).

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ValueError: If no SWIFT directory can be determined.
    """
    if swift_dir is None:
        swift_dir = get_profile().worktree

    if swift_dir is None:
        raise ValueError("The active profile has no worktree set.")

    if not swift_dir.exists():
        raise FileNotFoundError(f"SWIFT directory does not exist: {swift_dir}")

    return swift_dir


def _run_command_in_swift_dir(command: Sequence[str], swift_dir: Path) -> None:
    """Run a command in a SWIFT directory, streaming its output.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        ProfileError: If the command fails.
    """
    swift_dir = get_swiftsim_dir(swift_dir)
    sys.stdout.flush()  # keep our messages ordered before the tool's
    try:
        run_command_in_dir(command, swift_dir)
    except subprocess.CalledProcessError as e:
        raise ProfileError(
            f"'{' '.join(command)}' failed (exit code {e.returncode}) "
            f"in {swift_dir}"
        ) from e


# ---------- configure ----------


def parse_configure_options(help_text: str) -> set[str]:
    """Parse the option names from ``./configure --help`` output.

    The complement of every ``--enable-``/``--with-`` option (and vice versa)
    is included since autoconf accepts both.
    """
    options = set()
    for line in help_text.splitlines():
        match = _CONFIGURE_OPTION_RE.match(line)
        if match:
            options.add(match.group(1))
    for opt in list(options):
        for prefix, other in _COMPLEMENTS:
            if opt.startswith(prefix):
                options.add(other + opt[len(prefix) :])
    return options


def ensure_configure_script(worktree: Path) -> None:
    """Run ``./autogen.sh`` if ``./configure`` does not exist yet."""
    if not (worktree / "configure").exists():
        _run_command_in_swift_dir(["./autogen.sh"], worktree)


def get_configure_options(worktree: Path, profile_name: str) -> set[str]:
    """Get the valid configure options for a worktree.

    The parsed list is cached in the profile's cache directory, keyed by the
    modification time of ``./configure``.
    """
    ensure_configure_script(worktree)
    mtime = (worktree / "configure").stat().st_mtime
    cache = get_profile_cache_dir(profile_name) / "configure_options.json"
    try:
        data = json.loads(cache.read_text())
        if data["mtime"] == mtime:
            return set(data["options"])
    except (OSError, ValueError, KeyError):
        pass

    result = subprocess.run(
        ["./configure", "--help"],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProfileError(
            f"'./configure --help' failed in {worktree}:\n{result.stderr}"
        )
    options = parse_configure_options(result.stdout)
    cache.write_text(json.dumps({"mtime": mtime, "options": sorted(options)}))
    return options


def validate_configure_flags(
    flags: Sequence[str], worktree: Path, profile_name: str
) -> None:
    """Check configure flags against the options ``./configure`` accepts.

    Arguments not starting with ``--`` (e.g. ``CC=gcc``) are not checked.

    Raises:
        ProfileError: Listing unknown flags with close-match suggestions.
    """
    known = get_configure_options(worktree, profile_name)
    problems = []
    for flag in flags:
        if not flag.startswith("--"):
            continue
        name = flag.split("=", 1)[0]
        if name in known:
            continue
        close = difflib.get_close_matches(name, known, n=3)
        hint = f" (did you mean {', '.join(close)}?)" if close else ""
        problems.append(f"  {flag}{hint}")
    if problems:
        raise ProfileError(
            "Unknown configure option(s):\n"
            + "\n".join(problems)
            + "\nRun 'swift-cli config --show' to list valid options."
        )


def needs_configure(profile: SWIFTCLIProfile) -> bool:
    """Whether a profile must be (re)configured before building."""
    return (
        profile.worktree is None
        or not (profile.worktree / "config.status").exists()
        or profile.configured_flags != profile.configure_flags
    )


def configure_profile(
    profile: SWIFTCLIProfile, flags: Sequence[str] | None = None
) -> None:
    """Configure a profile's worktree and record the flags used.

    Runs ``./autogen.sh`` first if needed and validates the flags.

    Args:
        profile: The profile to configure.
        flags: Flags to use (and store as the profile's configure flags).
            Defaults to the profile's stored configure flags.
    """
    worktree = get_swiftsim_dir(profile.worktree)
    flags = list(profile.configure_flags if flags is None else flags)
    validate_configure_flags(flags, worktree, profile.name)
    _run_command_in_swift_dir(["./configure", *flags], worktree)
    update_profile_fields(
        profile.name, configure_flags=flags, configured_flags=flags
    )
    profile.configure_flags = profile.configured_flags = flags


def run_make(worktree: Path, nr_threads: int = 1, clean: bool = False) -> None:
    """Run ``make`` (optionally after ``make clean``) in a worktree."""
    if nr_threads < 1:
        raise ValueError("Number of threads must be at least 1.")
    if clean:
        _run_command_in_swift_dir(["make", "clean"], worktree)
    _run_command_in_swift_dir(["make", "-j", str(nr_threads)], worktree)


def build_profile(
    profile: SWIFTCLIProfile, nr_threads: int = 1, clean: bool = False
) -> None:
    """Build a profile, (re)configuring first if needed."""
    if nr_threads < 1:
        raise ValueError("Number of threads must be at least 1.")
    worktree = get_swiftsim_dir(profile.worktree)
    if needs_configure(profile):
        configure_profile(profile)
    run_make(worktree, nr_threads, clean)


def _rebase_in_progress(worktree: Path) -> bool:
    """Whether a rebase is stopped part-way in a worktree."""
    for name in ("rebase-merge", "rebase-apply"):
        path = Path(git("rev-parse", "--git-path", name, cwd=worktree))
        if (worktree / path).exists():  # no-op join if path is absolute
            return True
    return False


def update_profile(profile: SWIFTCLIProfile, nr_threads: int = 1) -> None:
    """Pull (rebasing local commits onto upstream), then build a profile.

    Raises:
        ProfileError: If the pull/rebase fails (nothing is built) or the
            build fails.
    """
    worktree = get_swiftsim_dir(profile.worktree)
    print(f"Updating profile '{profile.name}' from {profile.upstream}...")
    try:
        git("pull", "--rebase", cwd=worktree)
    except ProfileError as e:
        if not _rebase_in_progress(worktree):
            raise
        raise ProfileError(
            f"{e}\n\nRebasing profile '{profile.name}' onto "
            f"{profile.upstream} hit conflicts; nothing was built. To fix:\n"
            f"  cd {worktree}\n"
            "  # resolve the conflicts, then 'git add <files>'\n"
            "  git rebase --continue   # or 'git rebase --abort' to undo\n"
            f"then rerun 'swift-cli profile --update {profile.name}'."
        ) from e
    build_profile(profile, nr_threads)
