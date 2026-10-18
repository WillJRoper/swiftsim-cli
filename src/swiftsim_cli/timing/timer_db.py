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
import sys
import tarfile
import tempfile
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from swiftsim_cli.profile import (
    ProfileError,
    get_profile_cache_dir,
    git,
    load_swift_profile,
)

if TYPE_CHECKING:
    import tree_sitter

# Bump whenever the database contents or format change.
PARSER_VERSION = 3

# Directories (relative to the parse root) that are never parsed.
EXCLUDED_DIRS = {"tests", "examples", "argparse"}

# Source paths of a SWIFT checkout (older revisions keep main() in
# examples/main.c and examples/main_fof.c instead of swift.c/swift_fof.c).
MAIN_FILES = (
    "swift.c",
    "swift_fof.c",
    "examples/main.c",
    "examples/main_fof.c",
)
SWIFT_SOURCE_PATHS = ("src", *MAIN_FILES)

# The function running one step: timers outside its reach print between
# the steps (e.g. task dumps in main()) and are not part of any step.
STEP_FUNCTION = "engine_step"

# Milliseconds per unit printed by clocks_getunit().
UNIT_TO_MS = {
    "ms": 1.0,
    "s": 1000.0,
    "us": 1e-3,
    "ns": 1e-6,
    "milliseconds": 1.0,
    "seconds": 1000.0,
}

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
# A unit written literally after the time, e.g. ``took %.3f ms``.
_LITERAL_UNIT = re.compile(r"\s*(milliseconds|seconds|ms|us|ns|s)\b")

# Comments and string/char literals, blanked before brace counting.
_RE_COMMENT_OR_STRING = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.S,
)
_NOT_FUNCTIONS = {"if", "for", "while", "switch", "do", "else", "return"}


@lru_cache(maxsize=1)
def _tree_sitter() -> tuple[
    tree_sitter.Language, tree_sitter.Query, tree_sitter.Query
]:
    """Return the C language, the site query and the ``#if 0`` query.

    Built on first use so that importing this module stays cheap.
    """
    import tree_sitter
    import tree_sitter_c

    language = tree_sitter.Language(tree_sitter_c.language())
    query = tree_sitter.Query(
        language,
        "(function_definition) @func "
        "(call_expression function: (identifier) @callee) "
        "(argument_list (identifier) @arg) "
        "(argument_list (pointer_expression argument: (identifier) @arg))",
    )
    # ``#if 0`` blocks: their code (up to any #elif/#else) is never compiled.
    dead_query = tree_sitter.Query(
        language,
        '(preproc_if condition: (number_literal) @cond (#eq? @cond "0"))',
    )
    return language, query, dead_query


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
            unit = _LITERAL_UNIT.match(suffix, m.end())
            if unit is not None:
                parts.append(rf"\s*(?P<unit>{unit.group(1)})")
                pos = tail_start = unit.end()
                continue
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

    tail = re.sub(r"\([^)]*\)", "", suffix[tail_start:])
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
    _key: str = field(init=False, repr=False, compare=False)
    _prefix: re.Pattern[str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

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
        self._key = sys.intern(timer_key(self.function, self.kind, self.label))

    @property
    def prefix_regex(self) -> re.Pattern[str]:
        """Regex matching the body up to the time (for truncated lines)."""
        if self._prefix is None:
            end = self.pattern.find(_NUMBER)
            end = len(self.pattern) if end < 0 else end + len(_NUMBER)
            self._prefix = re.compile(self.pattern[:end])
        return self._prefix

    @property
    def specificity(self) -> int:
        """Literal characters in the label (more specific timers first)."""
        return len(_PRINTF_SPEC.sub("", self.label))

    def key(self, match: re.Match[str]) -> str:
        """Display key of one log line matched by this timer."""
        if not self._n_labels:
            return self._key
        values = [match.group(f"l{i}") for i in range(self._n_labels)]
        label = self._label_format.format(*values)
        return sys.intern(timer_key(self.function, self.kind, label))


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
                ranges=[tuple(r) for r in t["ranges"]],
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
        self.functions: frozenset[str] = frozenset(
            {*self.calls, *self.by_function}.union(*self.calls.values())
        )
        self._reach: dict[str, frozenset[str]] = {}
        self._op_reach: dict[int, frozenset[str]] = {}
        self._step_reach: frozenset[str] | None = None
        self._untimed_reach: frozenset[str] | None = None

    def knows(self, function: str) -> bool:
        """Whether the source defines (or calls) a function of this name."""
        return function in self.functions

    def in_step(self, function: str) -> bool:
        """Whether a function's timers can print inside a step.

        False only for functions the source shows cannot run inside
        ``engine_step``. Functions the source does not know, and every
        function when the source has no ``engine_step``, count as inside.
        """
        if not self.knows(STEP_FUNCTION):
            return True
        if self._step_reach is None:
            self._step_reach = self.reach(STEP_FUNCTION) | {STEP_FUNCTION}
        return function in self._step_reach or not self.knows(function)

    def outside_step_root(self, function: str) -> bool:
        """Whether a top-level timer of this function ran between steps.

        True for functions ``main`` calls directly that a step can only
        reach through functions with their own function timers (e.g.
        ``task_dump_stats``, also called by ``engine_repartition``): inside
        a step such a timer would be nested under one of those, so if it
        is top level, ``main`` called it after the step.
        """
        if not self.knows(STEP_FUNCTION):
            return False
        if self._untimed_reach is None:
            seen = {STEP_FUNCTION}
            stack = [STEP_FUNCTION]
            while stack:
                for callee in self.calls.get(stack.pop(), ()):
                    if callee not in seen:
                        seen.add(callee)
                        if not any(
                            t.kind == "function"
                            for t in self.by_function.get(callee, ())
                        ):
                            stack.append(callee)
            self._untimed_reach = frozenset(seen)
        return (
            function in self.calls.get("main", ())
            and function not in self._untimed_reach
        )

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
        files += [root / n for n in MAIN_FILES]
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


def _header_name(text: str) -> str | None:
    """Function name of a header ending ``name(...)`` (parens balanced)."""
    text = text.rstrip()
    if not text.endswith(")"):
        return None
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                m = re.search(r"\w+$", text[max(0, i - 200) : i].rstrip())
                return m.group() if m else None
        elif ch in ";{}":
            return None
    return None


def _fallback_function_name(clean: str, offset: int) -> str | None:
    """Enclosing function of an offset by counting braces backwards.

    Only used for the rare nodes tree-sitter could not place inside a
    function definition (e.g. functions whose braces are unbalanced across
    ``#if`` branches). Comments and strings must already be blanked.
    """
    if clean.count("{", 0, offset) <= clean.count("}", 0, offset):
        return None  # at file scope
    depth = 0
    for i in range(offset - 1, -1, -1):
        ch = clean[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            if depth:
                depth -= 1
                continue
            name = _header_name(clean[max(0, i - 2000) : i])
            if name and name not in _NOT_FUNCTIONS:
                return name
    return None


@dataclass
class _Function:
    """A function definition found in one file.

    ``start``/``end`` are byte offsets (-1 for a function only found by
    the brace-counting fallback); ``lines`` are its 1-based line span.
    """

    name: str
    start: int
    end: int
    lines: tuple[int, int] = (0, 0)
    tics: list[tuple[int, str]] = field(default_factory=list)


def _tic_variable(call: tree_sitter.Node) -> str | None:
    """Variable a ``getticks()``/``clocks_gettime(&v)`` call sets, if any."""
    function = call.child_by_field_name("function")
    target: tree_sitter.Node | None
    if function is not None and function.text == b"clocks_gettime":
        args = call.child_by_field_name("arguments")
        first = (
            args.named_children[0] if args and args.named_children else None
        )
        if first is None or first.type != "pointer_expression":
            return None
        target = first.child_by_field_name("argument")
    else:
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
) -> tuple[int, str] | None:
    """Start of a timer: the tic its message measures from.

    Takes the last assignment before the message of each tic variable
    used in the message arguments and returns the earliest of those (so
    ``clocks_diff(&tic, &toc)`` starts at ``tic``), falling back to the
    last tic before the message.
    """
    last: dict[str, int] = {}
    for ln, var in sorted(tics):
        if ln <= line:
            last[var] = ln
    if not last:
        return None
    used = set(re.findall(r"\b[A-Za-z_]\w*\b", args_text))
    candidates = [(ln, var) for var, ln in last.items() if var in used]
    if candidates:
        return min(candidates)
    return max((ln, var) for var, ln in last.items())


@dataclass
class _ParsedFile:
    """What one source file contributes to the database.

    Attributes:
        functions: Function definitions, in source order.
        sites: Timer message calls (file, line, function, start, tic,
            format).
        calls: (caller, line, callee) for each call or function
            identifier passed as an argument.
        fallback_sites: Sites placed by the brace-counting fallback.
        unplaced: Ids ("file:line") of timer messages no function could
            be found for (dropped).
    """

    functions: list[_Function]
    sites: list[dict[str, Any]]
    calls: list[tuple[str, int, str]]
    fallback_sites: int = 0
    unplaced: list[str] = field(default_factory=list)


def _dead_ranges(root: tree_sitter.Node) -> list[tuple[int, int]]:
    """Byte ranges of the code inside ``#if 0`` (up to any #elif/#else)."""
    import tree_sitter

    dead_query = _tree_sitter()[2]
    ranges = []
    for cond in (
        tree_sitter.QueryCursor(dead_query).captures(root).get("cond", [])
    ):
        block = cond.parent
        if block is None:
            continue
        alternative = block.child_by_field_name("alternative")
        end = alternative.start_byte if alternative else block.end_byte
        ranges.append((cond.end_byte, end))
    return ranges


def parse_source(src: bytes, rel: str) -> _ParsedFile:
    """Parse one C file into its functions, timer sites and calls.

    Args:
        src: The file contents.
        rel: The file's path as recorded in site ids.
    """
    import tree_sitter

    language, query, _ = _tree_sitter()
    tree = tree_sitter.Parser(language).parse(src)
    captures = tree_sitter.QueryCursor(query).captures(tree.root_node)
    dead = _dead_ranges(tree.root_node) if b"#if 0" in src else []

    funcs: list[_Function] = []
    for node in sorted(captures.get("func", []), key=lambda n: n.start_byte):
        fname = _declarator_name(node)
        if fname and not any(s <= node.start_byte < e for s, e in dead):
            funcs.append(
                _Function(
                    fname,
                    node.start_byte,
                    node.end_byte,
                    (node.start_point.row + 1, node.end_point.row + 1),
                )
            )
    starts = [f.start for f in funcs]
    fallback: dict[str, _Function] = {}
    clean: str | None = None

    def owner(node: tree_sitter.Node) -> _Function | None:
        nonlocal clean
        i = bisect_right(starts, node.start_byte) - 1
        while i >= 0:
            if funcs[i].end >= node.end_byte:
                return funcs[i]
            i -= 1
        if not tree.root_node.has_error:
            return None
        if clean is None:
            clean = _RE_COMMENT_OR_STRING.sub(
                lambda m: re.sub(r"[^\n]", " ", m.group()),
                src.decode(errors="replace"),
            )
        name = _fallback_function_name(
            clean, len(src[: node.start_byte].decode(errors="replace"))
        )
        if name is None:
            return None
        return fallback.setdefault(name, _Function(name, -1, -1))

    result = _ParsedFile(funcs, [], [])
    messages: list[tuple[tree_sitter.Node, _Function | None]] = []
    for node in sorted(
        captures.get("callee", []) + captures.get("arg", []),
        key=lambda n: n.start_byte,
    ):
        if not node.text or any(s <= node.start_byte < e for s, e in dead):
            continue
        func = owner(node)
        name = node.text.decode()
        call = node.parent
        if call is None or call.type != "call_expression":
            call = None
        if call is not None and name == "message":
            messages.append((call, func))
        if func is None:
            continue
        line = node.start_point.row + 1
        result.calls.append((func.name, line, name))
        if call is not None and name in ("getticks", "clocks_gettime"):
            var = _tic_variable(call)
            if var:
                func.tics.append((line, var))

    for call, func in messages:
        args = call.child_by_field_name("arguments")
        named = [
            c
            for c in (args.named_children if args else [])
            if c.type != "comment"
        ]
        if args is None or not named:
            continue
        first = named[0]
        if first.type not in ("string_literal", "concatenated_string"):
            continue
        # Only the literals: comments between them may contain quotes.
        pieces = (
            [first]
            if first.type == "string_literal"
            else [
                c for c in first.named_children if c.type == "string_literal"
            ]
        )
        fmt = _unescape(
            "".join(
                "".join(_RE_CSTRING.findall(c.text.decode(errors="replace")))
                for c in pieces
                if c.text
            )
        )
        if not _TOOK.search(fmt):
            continue
        line = call.start_point.row + 1
        if func is None:
            result.unplaced.append(f"{rel}:{line}")
            continue
        args_text = (args.text or b"").decode(errors="replace")
        tic = _choose_tic(func.tics, line, args_text)
        if func.start < 0:
            result.fallback_sites += 1
        result.sites.append(
            dict(
                file=rel,
                line=line,
                function=func.name,
                start=tic[0] if tic else None,
                tic=tic[1] if tic else None,
                format=fmt,
            )
        )
    return result


def build_timer_db(root: Path) -> dict[str, Any]:
    """Parse a SWIFT source tree into a timer database payload.

    Args:
        root: SWIFT checkout (containing ``src/``), or a source directory.

    Returns:
        The JSON-serialisable payload accepted by :class:`TimerDB`.
    """
    calls: dict[str, set[str]] = defaultdict(set)
    call_sites: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    sites: list[dict[str, Any]] = []
    defined: set[str] = set()
    fallback_sites = 0
    unplaced: list[str] = []
    files = _source_files(root)
    if not files:
        raise ValueError(
            f"no C source files found in {root}; point --swift-src at a "
            "SWIFT checkout (or its src/ directory)"
        )

    for path in files:
        rel = path.relative_to(root).as_posix()
        source = parse_source(path.read_bytes(), rel)
        for caller, line, callee in source.calls:
            calls[caller].add(callee)
            call_sites[caller].append((rel, line, callee))
        sites += source.sites
        defined |= {f.name for f in source.functions}
        fallback_sites += source.fallback_sites
        unplaced += source.unplaced

    defined |= set(calls) | {s["function"] for s in sites}
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

    # "fof_link_attachable_particles() took ..." inside that very function
    # times the whole function (unless it has a function timer already).
    with_function_timer = {
        e["function"] for e in grouped.values() if e["kind"] == "function"
    }
    for entry in grouped.values():
        name = entry["function"]
        if (
            entry["kind"] == "operation"
            and name not in with_function_timer
            and re.fullmatch(
                rf"\s*{re.escape(name)}\s*(?:\(\s*\))?", entry["label"]
            )
        ):
            entry["kind"], entry["label"] = "function", ""
            with_function_timer.add(name)

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
            unplaced_sites=len(unplaced),
            functions=len(defined),
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


def _source_status(root: Path) -> str:
    """``git status --porcelain`` of the source paths of a checkout."""
    return git("status", "--porcelain", "--", *SWIFT_SOURCE_PATHS, cwd=root)


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
        status = _source_status(root)
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
        paths = git(
            "ls-tree", "--name-only", sha, "--", *SWIFT_SOURCE_PATHS, cwd=repo
        ).split()
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
    log_dirty: bool = False,
) -> tuple[TimerDB, str]:
    """Pick the timer database to analyse a log with.

    By default the database is built from the revision the log reports
    (if the profile's repository has it), otherwise from the profile's
    working tree. When the log's revision is the checkout's HEAD but the
    checkout has uncommitted changes, a clean log is analysed with the
    committed source (``git archive``) and a dirty one with the working
    tree.

    Args:
        log_revision: Commit hash from the log's "Revision:" line.
        swift_src: Explicit source tree to use instead of the profile's.
        match_revision: Whether to build from the log's revision.
        regenerate: Rebuild the database even if cached.
        log_dirty: Whether the log's revision was marked ``-dirty``.

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
        if _source_status(root):
            if not log_dirty:
                db = load_revision_db(root, log_revision, regenerate)
                return db, (
                    f"{status}; the working tree has uncommitted changes, so "
                    "the committed source was used (git archive)"
                )
            status += (
                " (with uncommitted changes; the log was also written by a "
                "modified tree)"
            )
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
