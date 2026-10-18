"""Tools for working with a profile's SWIFT worktree.

Every mode that touches SWIFT source (config, make, update, new, ...) acts on
the active profile's worktree, resolved by `get_swiftsim_dir`.
"""

import difflib
import json
import re
import shlex
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
from swiftsim_cli.utilities.io import run_command_in_dir

# Matches an option at the start of a `./configure --help` line, e.g.
# "  --enable-debug=[yes/no]" or "  -h, --help". Placeholder entries such as
# "--enable-FEATURE[=ARG]" do not match (the name must end at a delimiter).
_CONFIGURE_OPTION_RE = re.compile(
    r"^\s+(?:-\w, )?(--[a-z0-9][a-z0-9_-]*[a-z0-9])(?=[\s=\[,]|$)"
)

# A ``VAR=value`` argument to ./configure (e.g. ``CC=gcc``)
_ENV_ASSIGN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")

# How to recreate a profile whose worktree is missing or unset
_RECREATE_HINT = (
    "Recreate the profile with 'swift-cli profile --remove NAME' then "
    "'swift-cli profile --new NAME', register an existing checkout with "
    "'swift-cli profile --adopt DIR --name NAME', or pick another profile "
    "with 'swift-cli --profile NAME ...'."
)


class CommandError(ProfileError):
    """A build command failed; ``returncode`` is the exit code to use."""

    def __init__(self, message: str, returncode: int) -> None:
        """Store the message and the shell-style exit code."""
        super().__init__(message)
        # A negative code means death by signal: report it like a shell
        self.returncode = 128 - returncode if returncode < 0 else returncode


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
        raise ValueError(
            f"The active profile has no worktree set. {_RECREATE_HINT}"
        )

    if not swift_dir.exists():
        raise FileNotFoundError(
            f"SWIFT directory does not exist: {swift_dir}. {_RECREATE_HINT}"
        )

    return swift_dir


def _run_command_in_swift_dir(command: Sequence[str], swift_dir: Path) -> None:
    """Run a command in a SWIFT directory, streaming its output.

    Raises:
        FileNotFoundError: If the SWIFT directory does not exist.
        CommandError: If the command fails (carrying its exit code).
    """
    swift_dir = get_swiftsim_dir(swift_dir)
    sys.stdout.flush()  # keep our messages ordered before the tool's
    try:
        run_command_in_dir(command, swift_dir)
    except subprocess.CalledProcessError as e:
        raise CommandError(
            f"'{shlex.join(command)}' failed (exit code {e.returncode}) "
            f"in {swift_dir}",
            e.returncode,
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

    Every argument must be an option (``--opt`` or ``--opt=value``; short
    ``-x`` options are not checked) or a ``VAR=value`` assignment.

    Raises:
        ProfileError: Listing bad arguments with suggestions.
    """
    problems = []
    for i, flag in enumerate(flags):
        if flag.startswith("-") or _ENV_ASSIGN_RE.match(flag):
            continue
        prev = flags[i - 1] if i else ""
        hint = (
            f" (did you mean {prev}={flag}?)"
            if prev.startswith("--") and "=" not in prev
            else ""
        )
        problems.append(f"  {flag} (not an option or VAR=value){hint}")
    options = [f for f in flags if f.startswith("--")]
    known = get_configure_options(worktree, profile_name) if options else set()
    for flag in options:
        name = flag.split("=", 1)[0]
        if name in known:
            continue
        close = difflib.get_close_matches(name, known, n=3)
        hint = f" (did you mean {', '.join(close)}?)" if close else ""
        problems.append(f"  {flag}{hint}")
    if problems:
        raise ProfileError(
            "Invalid configure argument(s):\n"
            + "\n".join(problems)
            + "\nOption values go after '=', e.g. --with-hydro=sphenix. Run "
            "'swift-cli config --list-options' to list valid options."
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
    ensure_configure_script(worktree)
    validate_configure_flags(flags, worktree, profile.name)
    print(
        f"Configuring profile '{profile.name}' with: "
        f"{shlex.join(flags) or '(no flags)'}"
    )
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
    """Build a profile, (re)configuring first if needed.

    Raises:
        ProfileError: If the worktree is mid-rebase or has unresolved
            conflicts, or configure/make fails.
    """
    if nr_threads < 1:
        raise ValueError("Number of threads must be at least 1.")
    worktree = get_swiftsim_dir(profile.worktree)
    check_buildable(worktree)
    if needs_configure(profile):
        configure_profile(profile)
    print(f"Building profile '{profile.name}' with make -j {nr_threads}")
    run_make(worktree, nr_threads, clean)


def _rebase_in_progress(worktree: Path) -> bool:
    """Whether a rebase is stopped part-way in a worktree."""
    for name in ("rebase-merge", "rebase-apply"):
        path = Path(git("rev-parse", "--git-path", name, cwd=worktree))
        if (worktree / path).exists():  # no-op join if path is absolute
            return True
    return False


def _conflicted_files(worktree: Path) -> list[str]:
    """Files with unresolved merge conflicts in a worktree."""
    out = git("diff", "--name-only", "--diff-filter=U", cwd=worktree)
    return out.splitlines()


def check_buildable(worktree: Path) -> None:
    """Refuse to build a worktree that is mid-rebase or has conflicts.

    Raises:
        ProfileError: Saying how to finish or undo the operation.
    """
    if _rebase_in_progress(worktree):
        raise ProfileError(
            f"A rebase is in progress in {worktree}; finish it ('git rebase "
            "--continue') or undo it ('git rebase --abort') before building."
        )
    conflicted = _conflicted_files(worktree)
    if conflicted:
        raise ProfileError(
            f"{worktree} has unresolved conflicts in: {', '.join(conflicted)}"
            ". Resolve them and 'git add' the files before building."
        )


def update_profile(
    profile: SWIFTCLIProfile, nr_threads: int = 1, build: bool = True
) -> None:
    """Pull (rebasing local commits onto upstream), then build a profile.

    Args:
        profile: The profile to update.
        nr_threads: Number of make threads.
        build: Whether to (re)configure and build after pulling.

    Raises:
        ProfileError: If the branch has no upstream, the pull/rebase fails
            or leaves conflicts (nothing is built), or the build fails.
    """
    if nr_threads < 1:
        raise ValueError("Number of threads must be at least 1.")
    worktree = get_swiftsim_dir(profile.worktree)
    try:
        tracking = git(
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
            cwd=worktree,
        )
    except ProfileError:
        raise ProfileError(
            f"Profile '{profile.name}' has no upstream branch to update "
            f"from (branch {profile.branch} tracks nothing). Set one with "
            f"'swift-cli profile --edit {profile.name} "
            "--set upstream=origin/master'."
        ) from None
    print(f"Updating profile '{profile.name}' from {tracking}...")
    old_head = git("rev-parse", "HEAD", cwd=worktree)
    if git("status", "--porcelain", "--untracked-files=no", cwd=worktree):
        print(
            f"Uncommitted changes in {worktree} will be stashed for the "
            "rebase and re-applied afterwards."
        )
    try:
        git("pull", "--rebase", "--autostash", cwd=worktree)
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
    conflicted = _conflicted_files(worktree)
    if conflicted:
        # git leaves the autostash in the stash list when re-applying it
        # conflicts, but still exits 0
        raise ProfileError(
            f"Updated profile '{profile.name}', but re-applying your "
            f"uncommitted changes conflicted in: {', '.join(conflicted)}. "
            "Nothing was built. To fix:\n"
            f"  cd {worktree}\n"
            "  # resolve the conflicts, then 'git add <files>'\n"
            "  git stash drop   # git kept a copy of your changes until now\n"
            f"then rerun 'swift-cli make' (or 'swift-cli profile --update "
            f"{profile.name}')."
        )
    if git("rev-parse", "HEAD", cwd=worktree) == old_head:
        print(f"Already up to date with {tracking}.")
    else:
        count = git(
            "rev-list", "--count", f"{old_head}..@{{upstream}}", cwd=worktree
        )
        print(f"Pulled {count} new commit(s) from {tracking}.")
    if build:
        build_profile(profile, nr_threads)
