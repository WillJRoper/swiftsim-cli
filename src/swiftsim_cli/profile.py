"""Profile storage and git worktree management for SWIFT-CLI.

A profile is a named, fully independent SWIFT build: its own git worktree of
the main SWIFT clone, on its own local branch (``swift-cli/<name>``) which
tracks an upstream branch, with its own configure flags and build products.

Profiles live in ``~/.swiftsim-utils/profiles.yaml`` (format version 2)::

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
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator
from ruamel.yaml import YAML

# Define the path to the profile file
PROFILE_FILE = Path.home() / ".swiftsim-utils" / "profiles.yaml"

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


def read_raw_profile_file() -> dict[str, Any] | None:
    """Read the raw profile file, returning None if missing or empty."""
    if not PROFILE_FILE.exists():
        return None
    data = YAML(typ="safe").load(PROFILE_FILE)
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
    return ProfileStore(
        repo=Path(data["repo"]),
        worktree_root=Path(data["worktree_root"]),
        current=data.get("current"),
        profiles={
            name: SWIFTCLIProfile.from_dict(name, entry or {})
            for name, entry in (data.get("profiles") or {}).items()
        },
    )


def load_store() -> ProfileStore:
    """Load the profile store.

    Returns a private copy, so callers may mutate it and pass it to
    `save_store`.

    Raises:
        ProfileError: If the file is missing, empty, old-format or invalid.
    """
    return copy.deepcopy(_load_store_cached(PROFILE_FILE))


def save_store(store: ProfileStore) -> None:
    """Atomically write the profile store and clear the load cache."""
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    yaml.default_flow_style = False
    fd, tmp = tempfile.mkstemp(
        dir=PROFILE_FILE.parent, prefix=".profiles-", suffix=".yaml"
    )
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(store.to_dict(), f)
        os.replace(tmp, PROFILE_FILE)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    finally:
        _load_store_cached.cache_clear()


# ---------- resolving the active profile ----------


def set_profile_override(name: str | None) -> None:
    """Force all modes to use profile ``name`` (None = the current one)."""
    global _PROFILE_OVERRIDE
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
        Path: ``~/.swiftsim-utils/cache/<name>/``.
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
    if not _NAME_RE.fullmatch(name):
        raise ProfileError(
            f"Invalid profile name '{name}': use letters, digits, '.', '_' "
            "and '-' (starting with a letter or digit)."
        )


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
) -> SWIFTCLIProfile:
    """Create a profile's branch and worktree and add it to ``store``.

    The store is not saved; the caller does that.

    Raises:
        ProfileError: For invalid/duplicate names, existing paths, or any
            git failure (e.g. unknown upstream, branch already exists).
    """
    validate_profile_name(name)
    if name in store.profiles:
        raise ProfileError(f"Profile '{name}' already exists.")
    worktree = store.worktree_root / name
    if worktree.exists():
        raise ProfileError(f"Worktree path already exists: {worktree}")
    branch = BRANCH_PREFIX + name

    fetch_for_upstream(store.repo, upstream)
    print(f"Creating worktree {worktree} on {branch} (tracking {upstream})")
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


class PathExistsValidator(Validator):
    """Validator to check if a given path exists."""

    def validate(self, document: Document) -> None:
        """Validate that the path exists."""
        p = Path(document.text).expanduser()
        if not p.exists():
            raise ValidationError(
                message="Path does not exist.",
                cursor_position=len(document.text),
            )


PTK_STYLE = Style.from_dict(
    {
        "completion-menu": "bg:#2b2e3b",
        "completion-menu.completion": "bg:#2b2e3b #c0caf5",
        "completion-menu.completion.current": "bg:#3b4252 #ffffff",
        "scrollbar.background": "bg:#3b4252",
        "scrollbar.button": "bg:#ffffff",
        "prompt": "bold",
    }
)

KB = KeyBindings()


@KB.add("enter")
def _(event: KeyPressEvent) -> None:
    """Enter accepts current completion if menu is open, otherwise submits."""
    buf = event.current_buffer
    if getattr(buf, "completer", None) is not None and buf.complete_state:
        comp = buf.complete_state.current_completion
        if comp is not None:
            buf.apply_completion(comp)
    else:
        buf.validate_and_handle()


@KB.add("tab")
def _(event: KeyPressEvent) -> None:
    """Tab opens/cycles completions, if the current prompt has a completer."""
    buf = event.current_buffer
    if getattr(buf, "completer", None) is not None:
        if buf.complete_state:
            buf.complete_next()
        else:
            buf.start_completion(select_first=True)


def ask(
    label: str,
    default: str = "",
    kind: str = "text",
    must_exist: bool = False,
) -> str:
    """Prompt the user for a value.

    Path prompts (``kind`` "dir" or "file") have tab completion; while the
    completion menu is open Enter applies the highlighted completion.

    Args:
        label: The prompt text.
        default: The pre-filled default value.
        kind: "text", "dir" or "file".
        must_exist: Reject paths that do not exist.

    Returns:
        str: The stripped answer.
    """
    is_path = kind != "text"
    session: PromptSession[str] = PromptSession(
        style=PTK_STYLE,
        key_bindings=KB if is_path else None,
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
        validator=PathExistsValidator() if must_exist else None,
        validate_while_typing=False,
    ).strip()


def confirm(question: str) -> bool:
    """Ask a yes/no question, defaulting to no."""
    return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")
