"""Profile storage and git worktree management for SWIFT-CLI.

A profile is a named, fully independent SWIFT build: its own git worktree of
the main SWIFT clone, on its own local branch (``swift-cli/<name>``) which
tracks an upstream branch, with its own configure flags and build products.

Profiles live in ``~/.swiftsim-cli/profiles.yaml`` (format version 2)::

    version: 2
    repo: /abs/path/to/swiftsim
    worktree_root: /abs/path/to/swiftsim-profiles
    current: default
    profiles:
      default:
        upstream: origin/master
        branch: swift-cli/default
        worktree: /abs/path/to/swiftsim-profiles/default
        configure_flags: [--enable-debug]
        configured_flags: null
        data_dir: ...
        parameter_file: ...
        softening_coeff: 0.04
        softening_pivot_z: 2.7
"""

import copy
import difflib
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from swiftsim_cli.utilities.messages import note

# Define the path to the profile file
PROFILE_FILE = Path.home() / ".swiftsim-cli" / "profiles.yaml"

# The directory older versions kept their files in
LEGACY_DIR_NAME = ".swiftsim-utils"

# The profile file format version this module reads and writes
FORMAT_VERSION = 2

# Prefix for the local branch owned by each profile
BRANCH_PREFIX = "swift-cli/"

# Profile name forced by the global ``--profile`` option (None = current)
_PROFILE_OVERRIDE: str | None = None

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ProfileError(ValueError):
    """A user-facing profile error (unknown profile, git failure, etc.)."""


def _opt_path(value: Any) -> Path | None:
    """Convert an optional string to a Path."""
    return Path(value) if value else None


def _opt_str(value: Path | None) -> str | None:
    """Convert an optional Path to a string."""
    return None if value is None else str(value)


@dataclass
class SWIFTCLIProfile:
    """A single SWIFT-CLI profile.

    Attributes:
        name: The profile name.
        worktree: The git worktree holding this profile's SWIFT checkout.
        branch: The local branch owned by this profile.
        upstream: The branch this profile follows (e.g. ``origin/master``).
        configure_flags: Flags to pass to ``./configure``.
        configured_flags: Flags used at the last successful configure, or
            None if never configured by SWIFT-CLI.
        data_dir: Path to the directory containing additional data files.
        parameter_file: Path to the template SWIFT parameter file.
        softening_coeff: Softening coefficient in units of mean separation.
        softening_pivot_z: Maximal softening pivot redshift.
    """

    name: str
    worktree: Path | None = None
    branch: str | None = None
    upstream: str | None = None
    configure_flags: list[str] = field(default_factory=list)
    configured_flags: list[str] | None = None
    data_dir: Path | None = None
    parameter_file: Path | None = None
    softening_coeff: float = 0.04
    softening_pivot_z: float = 2.7

    @property
    def swiftsim_dir(self) -> Path | None:
        """The SWIFT source directory (this profile's worktree)."""
        return self.worktree

    @property
    def is_built(self) -> bool:
        """Whether a SWIFT binary exists in the worktree."""
        return self.worktree is not None and any(
            (self.worktree / exe).exists() for exe in ("swift", "swift_mpi")
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the profile for the profile file."""
        return {
            "upstream": self.upstream,
            "branch": self.branch,
            "worktree": _opt_str(self.worktree),
            "configure_flags": list(self.configure_flags),
            "configured_flags": (
                None
                if self.configured_flags is None
                else list(self.configured_flags)
            ),
            "data_dir": _opt_str(self.data_dir),
            "parameter_file": _opt_str(self.parameter_file),
            "softening_coeff": float(self.softening_coeff),
            "softening_pivot_z": float(self.softening_pivot_z),
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "SWIFTCLIProfile":
        """Build a profile from its profile file entry."""
        configured = data.get("configured_flags")
        return cls(
            name=name,
            worktree=_opt_path(data.get("worktree")),
            branch=data.get("branch"),
            upstream=data.get("upstream"),
            configure_flags=list(data.get("configure_flags") or []),
            configured_flags=None if configured is None else list(configured),
            data_dir=_opt_path(data.get("data_dir")),
            parameter_file=_opt_path(data.get("parameter_file")),
            softening_coeff=float(data.get("softening_coeff", 0.04)),
            softening_pivot_z=float(data.get("softening_pivot_z", 2.7)),
        )


@dataclass
class ProfileStore:
    """The full contents of the profile file.

    Attributes:
        repo: The main SWIFT clone all worktrees are created from.
        worktree_root: Directory new profile worktrees are created in.
        current: Name of the current profile.
        profiles: All profiles keyed by name.
    """

    repo: Path
    worktree_root: Path
    current: str | None = None
    profiles: dict[str, SWIFTCLIProfile] = field(default_factory=dict)

    def get(self, name: str) -> SWIFTCLIProfile:
        """Get a profile by name.

        Raises:
            ProfileError: If no such profile exists.
        """
        if name not in self.profiles:
            close = difflib.get_close_matches(name, list(self.profiles))
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            known = ", ".join(sorted(self.profiles)) or "(none)"
            raise ProfileError(
                f"Unknown profile '{name}'.{hint} Available profiles: {known}"
            )
        return self.profiles[name]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the store for the profile file."""
        return {
            "version": FORMAT_VERSION,
            "repo": str(self.repo),
            "worktree_root": str(self.worktree_root),
            "current": self.current,
            "profiles": {n: p.to_dict() for n, p in self.profiles.items()},
        }


# ---------- reading and writing the profile file ----------


def move_legacy_dir() -> None:
    """Rename the old ``~/.swiftsim-utils`` directory to the current one.

    Does nothing if the old directory is missing or the new one exists.
    """
    new = PROFILE_FILE.parent
    old = new.with_name(LEGACY_DIR_NAME)
    if old.is_dir() and not new.exists():
        old.rename(new)
        _load_store_cached.cache_clear()
        note(f"moved {old} to {new}")


def read_raw_profile_file() -> dict[str, Any] | None:
    """Read the raw profile file, returning None if missing or empty."""
    if not PROFILE_FILE.exists():
        return None
    try:
        data = YAML(typ="safe").load(PROFILE_FILE)
    except YAMLError as e:
        raise ProfileError(
            f"Malformed profile file {PROFILE_FILE}:\n{e}"
        ) from e
    if not data:
        return None
    if not isinstance(data, dict):
        raise ProfileError(f"Malformed profile file: {PROFILE_FILE}")
    return data


def is_legacy_format(data: dict[str, Any]) -> bool:
    """Whether raw profile data is in the old (pre-version 2) format."""
    return "version" not in data


@lru_cache(maxsize=None)
def _load_store_cached(path: Path) -> ProfileStore:
    """Load and validate the profile file (cached per path)."""
    data = read_raw_profile_file()
    if data is None:
        raise ProfileError(
            f"No profiles found ({path} is missing or empty). Run "
            "'swift-cli profile --init' to create your first profile."
        )
    if is_legacy_format(data):
        raise ProfileError(
            f"{path} uses the old profile format. Run "
            "'swift-cli profile --migrate' to convert it."
        )
    if data["version"] != FORMAT_VERSION:
        raise ProfileError(
            f"Unsupported profile file version {data['version']!r} in "
            f"{path} (expected {FORMAT_VERSION})."
        )
    try:
        return ProfileStore(
            repo=Path(data["repo"]),
            worktree_root=Path(data["worktree_root"]),
            current=data.get("current"),
            profiles={
                str(name): SWIFTCLIProfile.from_dict(str(name), entry or {})
                for name, entry in (data.get("profiles") or {}).items()
            },
        )
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        what = f"missing key {e}" if isinstance(e, KeyError) else str(e)
        raise ProfileError(
            f"Malformed profile file {path} ({what}). Fix it by hand."
        ) from e


def load_store() -> ProfileStore:
    """Load the profile store.

    Returns a private copy, so callers may mutate it and pass it to
    `save_store`.

    Raises:
        ProfileError: If the file is missing, empty, old-format or invalid.
    """
    return copy.deepcopy(_load_store_cached(PROFILE_FILE))


def save_store(store: ProfileStore) -> None:
    """Atomically write the profile store and clear the load cache.

    The file is created with normal permissions (subject to the umask).
    """
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    yaml.default_flow_style = False
    tmp = PROFILE_FILE.with_name(f".{PROFILE_FILE.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            yaml.dump(store.to_dict(), f)
        os.replace(tmp, PROFILE_FILE)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        _load_store_cached.cache_clear()


# ---------- resolving the active profile ----------


def set_profile_override(name: str | None) -> None:
    """Force all modes to use profile ``name`` (None = the current one).

    Raises:
        ProfileError: If ``name`` is not a known profile, so a typo fails
            before any mode runs.
    """
    global _PROFILE_OVERRIDE
    if name is not None:
        load_store().get(name)
    _PROFILE_OVERRIDE = name


def active_profile_name(store: ProfileStore | None = None) -> str:
    """Name of the profile modes should act on (override or current)."""
    if store is None:
        store = load_store()
    name = _PROFILE_OVERRIDE or store.current
    if not name:
        raise ProfileError(
            "No current profile set. Run 'swift-cli profile --switch NAME'."
        )
    return name


def get_profile(name: str | None = None) -> SWIFTCLIProfile:
    """Get a profile by name, defaulting to the active profile."""
    store = load_store()
    return store.get(name or active_profile_name(store))


def load_swift_profile() -> SWIFTCLIProfile:
    """Load the active profile.

    If no profile file exists yet, an empty placeholder profile is returned
    (``swiftsim_dir`` is None) so callers can report a clean error.

    Returns:
        SWIFTCLIProfile: The active profile.
    """
    if not PROFILE_FILE.exists():
        return SWIFTCLIProfile(name="")
    return get_profile()


def update_profile_fields(name: str, **changes: Any) -> SWIFTCLIProfile:
    """Update fields of a stored profile and save.

    Args:
        name: The profile to update.
        **changes: Attribute names and their new values.

    Returns:
        SWIFTCLIProfile: The updated profile.
    """
    store = load_store()
    profile = store.get(name)
    for key, value in changes.items():
        setattr(profile, key, value)
    save_store(store)
    return profile


def get_profile_cache_dir(name: str | None = None) -> Path:
    """Get (and create) the cache directory for a profile.

    Args:
        name: The profile name. Defaults to the active profile. If no profile
            file exists yet, the shared cache root is used.

    Returns:
        Path: ``~/.swiftsim-cli/cache/<name>/``.
    """
    path = PROFILE_FILE.parent / "cache"
    if name is None and PROFILE_FILE.exists():
        name = active_profile_name()
    if name:
        path = path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------- git helpers ----------


def git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command, returning its stripped stdout.

    Raises:
        ProfileError: If git is missing or the command fails (the message
            includes git's own error output).
    """
    cmd = ["git"]
    if cwd is not None:
        cmd += ["-C", str(cwd)]
    cmd += list(args)
    sys.stdout.flush()  # keep our messages ordered before git's
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
    except FileNotFoundError as e:
        raise ProfileError("git executable not found.") from e
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        raise ProfileError(f"'{shlex.join(cmd)}' failed:\n{msg}") from e
    return result.stdout.strip()


def validate_profile_name(name: str) -> None:
    """Check a profile name is usable as a branch and directory name."""
    if (
        not _NAME_RE.fullmatch(name)
        or ".." in name
        or name.endswith((".", ".lock"))
    ):
        raise ProfileError(
            f"Invalid profile name '{name}': use letters, digits, '.', '_' "
            "and '-' (starting with a letter or digit, without '..', and not "
            "ending in '.' or '.lock')."
        )


def check_new_profile_name(store: ProfileStore, name: str) -> None:
    """Check ``name`` is valid and not taken (ignoring case).

    Names differing only in case would share a worktree directory on
    case-insensitive file systems (the macOS default).
    """
    validate_profile_name(name)
    for other in store.profiles:
        if other.lower() == name.lower():
            extra = (
                "" if other == name else " (names may not differ only in case)"
            )
            raise ProfileError(f"Profile '{other}' already exists{extra}.")


def main_clone_of(worktree: Path) -> Path:
    """The main clone a git worktree belongs to."""
    common = git("rev-parse", "--git-common-dir", cwd=worktree)
    return (worktree / common).resolve().parent


def fetch_for_upstream(repo: Path, upstream: str) -> None:
    """Fetch the remote an upstream ref lives on (no-op for local refs)."""
    remote = upstream.split("/", 1)[0]
    if "/" in upstream and remote in git("remote", cwd=repo).split():
        print(f"Fetching {remote}...")
        git("fetch", remote, cwd=repo)


def ref_exists(repo: Path, ref: str) -> bool:
    """Whether a git ref resolves in ``repo``."""
    try:
        git("rev-parse", "--verify", "--quiet", ref, cwd=repo)
    except ProfileError:
        return False
    return True


def list_worktrees(repo: Path) -> dict[Path, str | None]:
    """Map each worktree of ``repo`` to its checked-out branch (or None)."""
    out = git("worktree", "list", "--porcelain", cwd=repo)
    worktrees: dict[Path, str | None] = {}
    path: Path | None = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            path = Path(line[len("worktree ") :]).resolve()
            worktrees[path] = None
        elif line.startswith("branch ") and path is not None:
            worktrees[path] = line[len("branch ") :].removeprefix(
                "refs/heads/"
            )
    return worktrees


def add_profile(
    store: ProfileStore,
    name: str,
    upstream: str,
    configure_flags: list[str] | None = None,
    data_dir: Path | None = None,
    parameter_file: Path | None = None,
    softening_coeff: float = 0.04,
    softening_pivot_z: float = 2.7,
    reuse_branch: bool = False,
) -> SWIFTCLIProfile:
    """Create a profile's branch and worktree and add it to ``store``.

    The store is not saved; the caller does that.

    Args:
        store: The store to add the profile to.
        name: The new profile's name.
        upstream: The branch the profile follows.
        configure_flags: Flags for ``./configure``.
        data_dir: Directory of additional data files.
        parameter_file: Template parameter file (default: the worktree's
            ``examples/parameter_example.yml``).
        softening_coeff: Softening in units of the mean separation.
        softening_pivot_z: Maximal softening pivot redshift.
        reuse_branch: If the profile's branch (``swift-cli/<name>``) already
            exists, check it out in the new worktree (re-pointing its
            upstream) instead of failing; if its worktree already exists on
            that branch, register it as is.

    Raises:
        ProfileError: For invalid/duplicate names, existing paths, or any
            git failure (e.g. unknown upstream, branch already exists).
    """
    check_new_profile_name(store, name)
    worktree = store.worktree_root / name
    branch = BRANCH_PREFIX + name

    if worktree.exists():
        if not (
            reuse_branch
            and list_worktrees(store.repo).get(worktree.resolve()) == branch
        ):
            raise ProfileError(f"Worktree path already exists: {worktree}")
        print(f"Reusing existing worktree {worktree} on {branch}")
    elif ref_exists(store.repo, f"refs/heads/{branch}"):
        if not reuse_branch:
            raise ProfileError(
                f"Branch '{branch}' already exists (kept from an earlier "
                f"profile '{name}'?). Use 'swift-cli profile --new {name} "
                "--reuse-branch' to give the new profile that branch, or "
                f"delete it with 'git -C {store.repo} branch -D {branch}'."
            )
        fetch_for_upstream(store.repo, upstream)
        print(f"Creating worktree {worktree} on existing branch {branch}")
        store.worktree_root.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", str(worktree), branch, cwd=store.repo)
        git("branch", f"--set-upstream-to={upstream}", branch, cwd=worktree)
    else:
        fetch_for_upstream(store.repo, upstream)
        print(
            f"Creating worktree {worktree} on {branch} (tracking {upstream})"
        )
        store.worktree_root.mkdir(parents=True, exist_ok=True)
        git(
            "worktree",
            "add",
            "--track",
            "-b",
            branch,
            str(worktree),
            upstream,
            cwd=store.repo,
        )

    profile = SWIFTCLIProfile(
        name=name,
        worktree=worktree,
        branch=branch,
        upstream=upstream,
        configure_flags=list(configure_flags or []),
        data_dir=data_dir,
        parameter_file=parameter_file
        or worktree / "examples" / "parameter_example.yml",
        softening_coeff=softening_coeff,
        softening_pivot_z=softening_pivot_z,
    )
    store.profiles[name] = profile
    return profile


# ---------- interactive prompts ----------


@lru_cache(maxsize=None)
def _prompt_toolkit_extras() -> tuple[Any, Any]:
    """Build the prompt style and key bindings.

    prompt_toolkit is slow to import, so it is only loaded for a prompt.
    """
    from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
    from prompt_toolkit.styles import Style

    style = Style.from_dict(
        {
            "completion-menu": "bg:#2b2e3b",
            "completion-menu.completion": "bg:#2b2e3b #c0caf5",
            "completion-menu.completion.current": "bg:#3b4252 #ffffff",
            "scrollbar.background": "bg:#3b4252",
            "scrollbar.button": "bg:#ffffff",
            "prompt": "bold",
        }
    )
    kb = KeyBindings()

    @kb.add("enter")
    def _(event: KeyPressEvent) -> None:
        """Enter accepts the open completion, otherwise submits."""
        buf = event.current_buffer
        if getattr(buf, "completer", None) is not None and buf.complete_state:
            comp = buf.complete_state.current_completion
            if comp is not None:
                buf.apply_completion(comp)
        else:
            buf.validate_and_handle()

    @kb.add("tab")
    def _(event: KeyPressEvent) -> None:
        """Tab opens/cycles completions, if the prompt has a completer."""
        buf = event.current_buffer
        if getattr(buf, "completer", None) is not None:
            if buf.complete_state:
                buf.complete_next()
            else:
                buf.start_completion(select_first=True)

    return style, kb


def ask(
    label: str,
    default: str = "",
    kind: str = "text",
    must_exist: bool = False,
    check: Callable[[str], Any] | None = None,
) -> str:
    """Prompt the user for a value.

    Path prompts (``kind`` "dir" or "file") have tab completion; while the
    completion menu is open Enter applies the highlighted completion. A bad
    answer is reported and the question asked again.

    When stdin is not a terminal (answers piped in), one line is read per
    question and an empty line means the default; a bad answer is an error.

    Args:
        label: The prompt text.
        default: The pre-filled default value.
        kind: "text", "dir" or "file".
        must_exist: Reject paths that do not exist (blank is allowed).
        check: Called with the answer; raising ValueError rejects it with
            that message.

    Returns:
        str: The stripped answer.

    Raises:
        EOFError: If stdin is closed.
        ProfileError: For a bad piped answer.
    """

    def problem(text: str) -> str | None:
        if must_exist and text and not Path(text).expanduser().exists():
            return f"Path does not exist: {text}"
        if check is not None:
            try:
                check(text)
            except ValueError as e:
                return str(e)
        return None

    if not sys.stdin.isatty():
        text = input(label).strip() or default
        error = problem(text)
        if error:
            raise ProfileError(f"{label.strip()} {error}")
        return text

    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import PathCompleter
    from prompt_toolkit.document import Document
    from prompt_toolkit.validation import ValidationError, Validator

    class _Check(Validator):
        def validate(self, document: Document) -> None:
            error = problem(document.text.strip())
            if error:
                raise ValidationError(
                    message=error, cursor_position=len(document.text)
                )

    style, kb = _prompt_toolkit_extras()
    is_path = kind != "text"
    session: PromptSession[str] = PromptSession(
        style=style,
        key_bindings=kb if is_path else None,
        complete_while_typing=False,
        reserve_space_for_menu=8 if is_path else 0,
    )
    completer = (
        PathCompleter(expanduser=True, only_directories=kind == "dir")
        if is_path
        else None
    )
    return session.prompt(
        [("class:prompt", label)],
        default=default,
        completer=completer,
        validator=_Check(),
        validate_while_typing=False,
    ).strip()


def confirm(question: str) -> bool:
    """Ask a yes/no question, defaulting to no."""
    return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")
