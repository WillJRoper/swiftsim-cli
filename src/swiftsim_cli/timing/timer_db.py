"""Timer sites and static call graph parsed from SWIFT source.

SWIFT reports timings with ``message("... took %.3f %s.", ...)``. This
module finds every such call with tree-sitter and turns each format string
into a regex matching the body of the log line it produces (the text after
``function: ``). It also records which functions each function calls (plus
function identifiers passed as arguments, e.g. threadpool mappers) so the
runtime nesting of the timers can be checked against the source.

Classification follows one static rule: a timer is a ``function`` timer
(timing its whole enclosing function) when no literal text precedes
"took" and none follows the time other than a parenthesised remark, e.g.
``took %.3f %s.``, ``(%s) took %.3f %s.`` or ``took %.3f %s (including
reweight).``. Everything else is an ``operation`` timer.

Sites with an identical (function, regex) pair are indistinguishable in a
log, so they are grouped into one logical timer.

The database is cached per profile, keyed by the parser version, the SWIFT
``HEAD`` commit and the state of any uncommitted changes, so it is only
rebuilt when the source changes. A log written by a different revision is
analysed with a database built from that revision (extracted with
``git archive`` into the cache; the repository itself is never touched).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tree_sitter
import tree_sitter_c

from swiftsim_cli.profile import (
    ProfileError,
    get_profile_cache_dir,
    git,
    load_swift_profile,
)

# Bump whenever the database contents or format change.
PARSER_VERSION = 1

# Directories (relative to the parse root) that are never parsed.
EXCLUDED_DIRS = {"tests", "examples", "argparse"}

# Top-level source paths of a SWIFT checkout.
SWIFT_SOURCE_PATHS = ("src", "swift.c", "swift_fof.c")

# Milliseconds per unit printed by clocks_getunit().
UNIT_TO_MS = {"ms": 1.0, "s": 1000.0, "us": 1e-3, "ns": 1e-6}

# C string literal content.
_RE_CSTRING = re.compile(r'"((?:[^"\\]|\\.)*)"')

# printf-style conversion specifier.
_PRINTF_SPEC = re.compile(
    r"%(?:\d+\$)?[#0\- +']*(?:\d+|\*)?(?:\.(?:\d+|\*))?"
    r"(?:hh|h|l|ll|j|z|t|L)?[diuoxXfFeEgGaAcspn%]"
)

_ESCAPES = {'\\"': '"', "\\\\": "\\", "\\n": "\n", "\\t": "\t"}
_RE_ESCAPE = re.compile(r"\\[\\\"nt]")

_TOOK = re.compile(r"took\b")
_NUMBER = r"(?P<ms>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_UNIT = r"(?P<unit>\w+)"
_LITERAL_UNIT = re.compile(r"^\s*(?:ms|s|seconds|milliseconds)\b")

# Comments and string/char literals, blanked before brace counting.
_RE_COMMENT_OR_STRING = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.S,
)
_RE_HEADER_END = re.compile(r"(\w+)\s*\([^;{}]*\)\s*$")
_NOT_FUNCTIONS = {"if", "for", "while", "switch", "do", "else", "return"}

_LANGUAGE = tree_sitter.Language(tree_sitter_c.language())
_QUERY = tree_sitter.Query(
    _LANGUAGE,
    "(function_definition) @func "
    "(call_expression function: (identifier) @callee) "
    "(argument_list (identifier) @arg)",
)


# =============================================================================
# Format strings
# =============================================================================


def _unescape(text: str) -> str:
    """Unescape the C escapes that matter for matching log text."""
    return _RE_ESCAPE.sub(lambda m: _ESCAPES[m.group()], text)


def _literal(text: str) -> str:
    """Regex for literal format text, matching any whitespace run."""
    return re.sub(r"(?:\\ )+", r"\\s+", re.escape(text))


def _has_text(text: str) -> bool:
    """Whether text has letters or digits once format specs are removed."""
    return re.search(r"[A-Za-z0-9]", _PRINTF_SPEC.sub("", text)) is not None


@dataclass(frozen=True)
class TimerFormat:
    """What a timer format string tells us.

    Attributes:
        pattern: Regex matching the log-line body, with groups ``ms``,
            optionally ``unit`` and one ``l<i>`` per ``%s`` before "took".
        label: Format text before "took" (or a description of the text
            after the time for operations with none before it).
        kind: "function" or "operation".
    """

    pattern: str
    label: str
    kind: str


def parse_timer_format(fmt: str) -> TimerFormat | None:
    """Convert a ``message`` format containing "took" into a log regex.

    Args:
        fmt: The unescaped format string.

    Returns:
        The parsed format, or None if no number follows "took".
    """
    fmt = " ".join(fmt.split())
    took = _TOOK.search(fmt)
    if took is None:
        return None
    prefix, suffix = fmt[: took.start()].rstrip(), fmt[took.end() :]

    specs = list(_PRINTF_SPEC.finditer(suffix))
    floats = [i for i, m in enumerate(specs) if m.group()[-1] in "fFeEgG"]
    ints = [i for i, m in enumerate(specs) if m.group()[-1] in "diu"]
    if not floats and not ints:
        return None
    ms_index = (floats or ints)[0]

    parts = ["^"]
    pos = 0
    n_labels = 0
    for m in _PRINTF_SPEC.finditer(prefix):
        parts.append(_literal(prefix[pos : m.start()]))
        spec = m.group()
        if spec == "%%":
            parts.append("%")
        elif spec.endswith("s"):
            parts.append(f"(?P<l{n_labels}>.*?)")
            n_labels += 1
        else:
            parts.append(".*?")
        pos = m.end()
    parts.append(_literal(prefix[pos:]))
    parts.append(r"\s*took" if prefix else "took")

    pos = 0
    tail_start = 0
    for i, m in enumerate(specs):
        parts.append(_literal(suffix[pos : m.start()]))
        spec = m.group()
        if i == ms_index:
            parts.append(_NUMBER)
            tail_start = m.end()
        elif (
            i == ms_index + 1
            and spec.endswith("s")
            and not suffix[specs[i - 1].end() : m.start()].strip()
        ):
            parts.append(_UNIT)
            tail_start = m.end()
        elif spec == "%%":
            parts.append("%")
        else:
            parts.append(".*?")
        pos = m.end()
    parts.append(_literal(suffix[pos:].rstrip()))
    parts.append(r"\s*$")

    tail = _LITERAL_UNIT.sub("", suffix[tail_start:])
    tail = re.sub(r"\([^)]*\)", "", tail)
    if _has_text(prefix):
        kind, label = "operation", prefix
    elif _has_text(tail):
        kind = "operation"
        label = f"{prefix} took ... {tail.strip(' .:')}".strip()
    else:
        kind, label = "function", prefix
    return TimerFormat("".join(parts), label, kind)


# =============================================================================
# Database model
# =============================================================================


@dataclass
class LogicalTimer:
    """Timer sites that produce indistinguishable log lines.

    Attributes:
        function: Enclosing function (the name printed in the log).
        kind: "function" or "operation".
        pattern: Regex matching the log-line body.
        label: Label template (format text before "took").
        sites: Site ids ("file:line" of each message call).
        ranges: (file, start line, end line) of each site; the start is
            the ``getticks()`` call the time is measured from.
    """

    function: str
    kind: str
    pattern: str
    label: str
    sites: list[str]
    ranges: list[tuple[str, int, int]]
    regex: re.Pattern[str] = field(init=False, repr=False, compare=False)
    _label_format: str = field(init=False, repr=False, compare=False)
    _n_labels: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compile the regex and the key template."""
        self.regex = re.compile(self.pattern)
        self._n_labels = sum(
            1 for name in self.regex.groupindex if name.startswith("l")
        )
        template = self.label.replace("{", "{{").replace("}", "}}")
        self._label_format = _PRINTF_SPEC.sub(
            lambda m: "{}" if m.group().endswith("s") else m.group(),
            template,
        )

    @property
    def specificity(self) -> int:
        """Literal characters in the label (more specific timers first)."""
        return len(_PRINTF_SPEC.sub("", self.label))

    def key(self, match: re.Match[str]) -> str:
        """Display key of one log line matched by this timer."""
        label = self.label
        if self._n_labels:
            values = [match.group(f"l{i}") for i in range(self._n_labels)]
            label = self._label_format.format(*values)
        return timer_key(self.function, self.kind, label)


def timer_key(function: str, kind: str, label: str) -> str:
    """Display key of a timer.

    Function timers are keyed ``function`` (plus any runtime label, e.g.
    ``engine_launch (tasks)``); operations are ``function: label``.
    """
    label = " ".join(label.split())
    if not label:
        return function
    if kind == "function":
        return f"{function} {label}"
    return f"{function}: {label}"


class TimerDB:
    """Timer definitions and static call graph of one SWIFT source tree.

    Attributes:
        timers: All logical timers.
        calls: Function name -> names of the functions it calls.
        call_sites: Function name -> (file, line, callee) for functions
            with operation timers.
        by_function: Function name -> its timers, most specific first.
        stats: Counts collected while parsing.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        """Build the database from its JSON payload."""
        self.timers = [
            LogicalTimer(
                function=t["function"],
                kind=t["kind"],
                pattern=t["pattern"],
                label=t["label"],
                sites=t["sites"],
                ranges=[tuple(r) for r in t["ranges"]],  # type: ignore[misc]
            )
            for t in payload["timers"]
        ]
        self.calls: dict[str, list[str]] = payload["calls"]
        self.call_sites: dict[str, list[list[Any]]] = payload["call_sites"]
        self.stats: dict[str, int] = payload.get("stats", {})
        self.by_function: dict[str, list[LogicalTimer]] = defaultdict(list)
        for timer in self.timers:
            self.by_function[timer.function].append(timer)
        for timers in self.by_function.values():
            timers.sort(key=lambda t: -t.specificity)
        self.by_function = dict(self.by_function)
        self._reach: dict[str, frozenset[str]] = {}
        self._op_reach: dict[int, frozenset[str]] = {}

    def reach(self, function: str) -> frozenset[str]:
        """Every function reachable from ``function`` through calls."""
        cached = self._reach.get(function)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = [function]
        while stack:
            for callee in self.calls.get(stack.pop(), ()):
                if callee not in seen:
                    seen.add(callee)
                    stack.append(callee)
        result = frozenset(seen)
        self._reach[function] = result
        return result

    def op_reach(self, timer: LogicalTimer) -> frozenset[str]:
        """Functions reachable from calls inside an operation's line range."""
        cached = self._op_reach.get(id(timer))
        if cached is not None:
            return cached
        reached: set[str] = set()
        for file, line, callee in self.call_sites.get(timer.function, ()):
            if any(
                f == file and start <= line <= end
                for f, start, end in timer.ranges
            ):
                reached.add(callee)
                reached |= self.reach(callee)
        result = frozenset(reached)
        self._op_reach[id(timer)] = result
        return result

    @staticmethod
    def lexically_inside(inner: LogicalTimer, outer: LogicalTimer) -> bool:
        """Whether an operation's code lies inside another's line range."""
        return any(
            fi == fo and so <= si and ei < eo
            for fi, si, ei in inner.ranges
            for fo, so, eo in outer.ranges
        )


# =============================================================================
# Source parsing
# =============================================================================


def _source_files(root: Path) -> list[Path]:
    """The C files to parse under a SWIFT checkout (or any source dir)."""
    base = root / "src" if (root / "src").is_dir() else root
    files = [
        path
        for pattern in ("*.c", "*.h")
        for path in base.rglob(pattern)
        if not EXCLUDED_DIRS.intersection(path.relative_to(base).parts)
    ]
    if base != root:
        files += [root / n for n in ("swift.c", "swift_fof.c")]
    return sorted(p for p in files if p.is_file())


def _declarator_name(node: tree_sitter.Node) -> str | None:
    """Name of a function definition, descending through its declarators.

    Handles pointer returns (``struct cell *f(...)``), parenthesised and
    attributed declarators.
    """
    current = node.child_by_field_name("declarator")
    while current is not None:
        if current.type == "identifier":
            return current.text.decode() if current.text else None
        nxt = current.child_by_field_name("declarator")
        if nxt is None:
            nxt = next(
                (
                    c
                    for c in current.named_children
                    if c.type == "identifier" or c.type.endswith("declarator")
                ),
                None,
            )
        current = nxt
    return None


def _fallback_function_name(clean: str, offset: int) -> str | None:
    """Enclosing function of an offset by counting braces backwards.

    Only used for the rare call sites tree-sitter could not place inside a
    function definition (e.g. functions whose braces are unbalanced across
    ``#if`` branches). Comments and strings must already be blanked.
    """
    depth = 0
    for i in range(offset - 1, -1, -1):
        ch = clean[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            if depth:
                depth -= 1
                continue
            m = _RE_HEADER_END.search(clean[max(0, i - 2000) : i])
            if m and m.group(1) not in _NOT_FUNCTIONS:
                return m.group(1)
    return None


@dataclass
class _Function:
    """A function definition found in one file."""

    name: str
    start: int
    end: int
    tics: list[tuple[int, str]] = field(default_factory=list)


def _tic_variable(call: tree_sitter.Node) -> str | None:
    """Variable a ``getticks()`` call is assigned to, if any."""
    parent = call.parent
    if parent is None:
        return None
    if parent.type == "init_declarator":
        target = parent.child_by_field_name("declarator")
    elif parent.type == "assignment_expression":
        target = parent.child_by_field_name("left")
    else:
        return None
    if target is None or target.type != "identifier" or not target.text:
        return None
    return target.text.decode()


def _choose_tic(
    tics: list[tuple[int, str]], line: int, args_text: str
) -> int | None:
    """Start line of a timer: the nearest tic whose variable is used.

    Prefers the last ``getticks()`` assignment before the message whose
    variable appears in the message arguments, falling back to the last
    one before the message.
    """
    before = sorted((ln, var) for ln, var in tics if ln <= line)
    if not before:
        return None
    used = set(re.findall(r"\b[A-Za-z_]\w*\b", args_text))
    for ln, var in reversed(before):
        if var in used:
            return ln
    return before[-1][0]


def build_timer_db(root: Path) -> dict[str, Any]:
    """Parse a SWIFT source tree into a timer database payload.

    Args:
        root: SWIFT checkout (containing ``src/``), or a source directory.

    Returns:
        The JSON-serialisable payload accepted by :class:`TimerDB`.
    """
    parser = tree_sitter.Parser(_LANGUAGE)
    cursor = tree_sitter.QueryCursor(_QUERY)
    calls: dict[str, set[str]] = defaultdict(set)
    call_sites: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    sites: list[dict[str, Any]] = []
    fallback_sites = 0
    files = _source_files(root)

    for path in files:
        src = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        captures = cursor.captures(parser.parse(src).root_node)

        funcs: list[_Function] = []
        for node in sorted(
            captures.get("func", []), key=lambda n: n.start_byte
        ):
            fname = _declarator_name(node)
            if fname:
                funcs.append(_Function(fname, node.start_byte, node.end_byte))
        starts = [f.start for f in funcs]

        def owner(node: tree_sitter.Node) -> _Function | None:
            i = bisect_right(starts, node.start_byte) - 1
            while i >= 0:
                if funcs[i].end >= node.end_byte:
                    return funcs[i]
                i -= 1
            return None

        messages: list[tuple[tree_sitter.Node, _Function | None]] = []
        for node in captures.get("callee", []):
            name = node.text.decode() if node.text else ""
            func = owner(node)
            if name == "message":
                messages.append((node.parent, func))  # type: ignore[arg-type]
            if func is None:
                continue
            line = node.start_point.row + 1
            if name == "getticks":
                var = _tic_variable(node.parent)  # type: ignore[arg-type]
                if var:
                    func.tics.append((line, var))
            calls[func.name].add(name)
            call_sites[func.name].append((rel, line, name))
        for node in captures.get("arg", []):
            func = owner(node)
            if func is not None and node.text:
                name = node.text.decode()
                calls[func.name].add(name)
                call_sites[func.name].append(
                    (rel, node.start_point.row + 1, name)
                )

        clean: str | None = None
        for call, func in messages:
            args = call.child_by_field_name("arguments")
            if args is None or not args.named_children:
                continue
            first = args.named_children[0]
            if first.type not in ("string_literal", "concatenated_string"):
                continue
            fmt = _unescape(
                "".join(
                    _RE_CSTRING.findall(first.text.decode(errors="replace"))
                )
            )
            if not _TOOK.search(fmt):
                continue
            line = call.start_point.row + 1
            owner_name: str | None
            if func is not None:
                owner_name = func.name
                args_text = (args.text or b"").decode(errors="replace")
                start = _choose_tic(func.tics, line, args_text)
            else:
                if clean is None:
                    text = src.decode(errors="replace")
                    clean = _RE_COMMENT_OR_STRING.sub(
                        lambda m: re.sub(r"[^\n]", " ", m.group()), text
                    )
                owner_name = _fallback_function_name(
                    clean, len(src[: call.start_byte].decode(errors="replace"))
                )
                start = None
                fallback_sites += 1
            if owner_name:
                sites.append(
                    dict(
                        file=rel,
                        line=line,
                        function=owner_name,
                        start=start,
                        format=fmt,
                    )
                )

    defined = set(calls) | {s["function"] for s in sites}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for site in sites:
        parsed = parse_timer_format(site["format"])
        if parsed is None:
            continue
        entry = grouped.setdefault(
            (site["function"], parsed.pattern),
            dict(
                function=site["function"],
                kind=parsed.kind,
                pattern=parsed.pattern,
                label=parsed.label,
                sites=[],
                ranges=[],
            ),
        )
        entry["sites"].append(f"{site['file']}:{site['line']}")
        start = site["start"] if site["start"] is not None else site["line"]
        entry["ranges"].append([site["file"], start, site["line"]])

    op_functions = {
        e["function"] for e in grouped.values() if e["kind"] == "operation"
    }
    return dict(
        version=PARSER_VERSION,
        timers=list(grouped.values()),
        calls={
            f: sorted(c for c in callees if c in defined and c != f)
            for f, callees in calls.items()
            if any(c in defined and c != f for c in callees)
        },
        call_sites={
            f: [list(s) for s in call_sites[f] if s[2] in defined]
            for f in op_functions
            if f in call_sites
        },
        stats=dict(
            files=len(files),
            sites=len(sites),
            timers=len(grouped),
            fallback_sites=fallback_sites,
            functions=len(calls),
        ),
    )


# =============================================================================
# Caching and revision matching
# =============================================================================


def _git_head(root: Path) -> str | None:
    """HEAD commit of a git checkout, or None if root is not the top of one."""
    try:
        top = git("rev-parse", "--show-toplevel", cwd=root)
        if Path(top).resolve() != root.resolve():
            return None
        return git("rev-parse", "HEAD", cwd=root)
    except ProfileError:
        return None


def source_key(root: Path) -> str:
    """Cache key of a source tree's current state.

    For a git checkout: the parser version, HEAD and a hash of the
    uncommitted changes to the source paths (their ``git status`` plus the
    size and modification time of each changed file, so further edits to
    an already-modified file are noticed). Otherwise a hash of the size and
    modification time of every parsed file.
    """
    digest = hashlib.sha1()
    head = _git_head(root)
    if head is not None:
        status = git(
            "status", "--porcelain", "--", *SWIFT_SOURCE_PATHS, cwd=root
        )
        digest.update(status.encode())
        for line in status.splitlines():
            name = line.split(maxsplit=1)[-1].split(" -> ")[-1]
            path = root / name.strip('"')
            if path.is_file():
                st = path.stat()
                digest.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
        return f"v{PARSER_VERSION}-{head}-{digest.hexdigest()[:12]}"
    for path in _source_files(root):
        st = path.stat()
        digest.update(f"{path}:{st.st_size}:{st.st_mtime_ns}".encode())
    return f"v{PARSER_VERSION}-files-{digest.hexdigest()[:12]}"


def _db_cache_dir() -> Path:
    """Directory holding the cached databases of the active profile."""
    path = get_profile_cache_dir() / "timer_db"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_cached(path: Path, key: str) -> dict[str, Any] | None:
    """Load a cached payload if it exists and has the expected key."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if payload.get("key") == key else None


def _store(path: Path, key: str, payload: dict[str, Any]) -> None:
    """Write a payload to the cache with its key."""
    payload["key"] = key
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_source_db(root: Path, regenerate: bool = False) -> TimerDB:
    """Database of a source tree, rebuilt only when the tree changes.

    Args:
        root: SWIFT checkout or source directory.
        regenerate: Rebuild even if the cached database is current.
    """
    root = root.resolve()
    key = source_key(root)
    name = hashlib.sha1(str(root).encode()).hexdigest()[:12]
    path = _db_cache_dir() / f"tree-{name}.json"
    payload = None if regenerate else _load_cached(path, key)
    if payload is None:
        print(f"Parsing timers in {root} ...")
        payload = build_timer_db(root)
        _store(path, key, payload)
    return TimerDB(payload)


def load_revision_db(
    repo: Path, revision: str, regenerate: bool = False
) -> TimerDB:
    """Database of a commit, built from a ``git archive`` of its source.

    The archive is extracted into a temporary directory in the cache and
    removed once parsed; the repository is not modified.

    Args:
        repo: The SWIFT git repository.
        revision: A commit-ish present in the repository.
        regenerate: Rebuild even if the cached database exists.
    """
    sha = git("rev-parse", f"{revision}^{{commit}}", cwd=repo)
    key = f"v{PARSER_VERSION}-{sha}"
    path = _db_cache_dir() / f"rev-{sha}.json"
    payload = None if regenerate else _load_cached(path, key)
    if payload is None:
        print(f"Parsing timers at revision {sha[:8]} (git archive) ...")
        present = set(git("ls-tree", "--name-only", sha, cwd=repo).split())
        paths = [p for p in SWIFT_SOURCE_PATHS if p in present]
        archive = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", sha, *paths],
            capture_output=True,
            check=True,
        ).stdout
        with tempfile.TemporaryDirectory(dir=_db_cache_dir()) as tmp:
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(tmp, filter="data")
            payload = build_timer_db(Path(tmp))
        _store(path, key, payload)
    return TimerDB(payload)


def resolve_timer_db(
    log_revision: str | None,
    swift_src: Path | None = None,
    match_revision: bool = True,
    regenerate: bool = False,
) -> tuple[TimerDB, str]:
    """Pick the timer database to analyse a log with.

    By default the database is built from the revision the log reports
    (if the profile's repository has it), otherwise from the profile's
    working tree.

    Args:
        log_revision: Commit hash from the log's "Revision:" line.
        swift_src: Explicit source tree to use instead of the profile's.
        match_revision: Whether to build from the log's revision.
        regenerate: Rebuild the database even if cached.

    Returns:
        The database and a one-line description of the choice.

    Raises:
        ValueError: If no source tree is available.
    """
    if swift_src is not None:
        return (
            load_source_db(swift_src, regenerate),
            f"source tree {swift_src} (--swift-src)",
        )
    root = load_swift_profile().swiftsim_dir
    if root is None:
        raise ValueError(
            "Cannot build the timer database: the active profile has no "
            "SWIFT checkout. Run 'swift-cli profile --init' or pass "
            "--swift-src."
        )
    head = _git_head(root)
    if log_revision is None:
        status = "log has no revision; using the profile's working tree"
    elif not match_revision:
        status = "revision matching disabled; using the profile's working tree"
    elif head is not None and head.startswith(log_revision):
        status = f"log revision {log_revision} matches the working tree HEAD"
    elif head is not None:
        try:
            db = load_revision_db(root, log_revision, regenerate)
            return db, f"matched log revision {log_revision} (git archive)"
        except (ProfileError, subprocess.CalledProcessError):
            status = (
                f"log revision {log_revision} not found in {root}; using "
                f"the working tree at {head[:8]} (timers may not match)"
            )
    else:
        status = f"{root} is not a git checkout; using it as is"
    return load_source_db(root, regenerate), status
