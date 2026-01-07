"""Fast parser for SWIFT-style timing sites with timer classification.

This module scans a C/C++ codebase for timing messages of the form:

    message("... took ...", ...)

and builds a YAML database that includes:
  * Per-site log-matching regex (anchored on `FUNCTION:` and capturing `<ms>`),
  * The **tic** variable and start line (nearest relevant `getticks()`),
  * The message line (end of the timer),
  * Timer classification: 'function' for simple "took X ms", 'operation' for
    "description took X ms".

Public helpers (imported by your analysis function):
  - load_timer_db() -> Dict[str, TimerDef]
  - compile_site_patterns(timer_db) -> List[(timer_id, re.Pattern)]
  - scan_log_instances_by_step(log_file, compiled, timer_db)

The parser is designed for speed:
  * single pass per file,
  * small regexes,
  * tiny parenthesis-aware scanner for `message(...)`,
  * no full-file comment stripping or AST building.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tree_sitter
import tree_sitter_c
from ruamel.yaml import YAML
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from swiftsim_cli.profile import load_swift_profile

TIMER_FILE = Path.home() / ".swiftsim-utils" / "timers.yaml"

__all__ = [
    "CTimerParser",
    "TimerSite",
    "TimerDef",
    "TimerInstance",
    "load_timer_db",
    "compile_site_patterns",
    "scan_log_instances_by_step",
    "TimerNestingGenerator",
    "generate_timer_nesting_database",
    "TaskCountSnapshot",
    "scan_task_counts_by_step",
]

# =============================================================================
# Small regex helpers (kept fast & simple)
# =============================================================================

# Log step lines: first integer is the step number.
_RE_STEP_LINE = re.compile(r"^\s*(?P<step>\d+)\b")

# C string literal content.
_RE_CSTRING = re.compile(r'"((?:[^"\\]|\\.)*)"')

# printf-style specifier to replace with `.*?` for tolerant matching.
_PRINTF_SPEC = re.compile(
    r"%(?:\d+\$)?[#0\- +']*(?:\d+|\*)?(?:\.(?:\d+|\*))?"
    r"(?:hh|h|l|ll|j|z|t|L)?[diuoxXfFeEgGaAcspn%]"
)


# =============================================================================
# Data model
# =============================================================================


@dataclass
class TimerSite:
    """One timing site discovered in source code.

    Attributes:
        timer_id: Stable id like 'relative/path.c:LINE' (message line).
        file: Relative file path (relative to parse root).
        function: Enclosing function name.
        start_line: 1-based line of nearest relevant `getticks()` (tic) if
            found.
        end_line: 1-based line of the `message(...)` (timer completion).
        tic_var: Chosen tic variable name (if any).
        label_text: Human-readable label derived from the format string.
        log_pattern: Regex that matches the log line and captures `<ms>` as
            group 1.
    """

    timer_id: str
    file: str
    function: str
    start_line: Optional[int]
    end_line: int
    tic_var: Optional[str]
    label_text: str
    log_pattern: str
    timer_type: str  # 'function' or 'operation'


@dataclass
class TimerDef:
    """Timer definition loaded from YAML for runtime matching and nesting.

    Attributes:
        timer_id: Site identifier (file:line).
        function: Enclosing function.
        log_pattern: Regex string that matches the log line and captures
            `<ms>`.
        start_line: Static start line (tic) if known.
        end_line: Static end line (message) in the source file.
        label_text: Human-readable description of what this timer measures.
        timer_type: 'function' or 'operation'.
    """

    timer_id: str
    function: str
    log_pattern: str
    start_line: Optional[int]
    end_line: int
    label_text: str
    timer_type: str  # 'function' or 'operation'


@dataclass
class TimerInstance:
    """One runtime occurrence of a timer in the log.

    Attributes:
        timer_id: Identifier of the site that produced this log line.
        function: Enclosing function (from DB).
        step: Detected step number (if any), else None.
        time_ms: Time printed on the line (milliseconds).
        line_index: Line index in the log file (0-based), used for ordering.
        timer_type: 'function' or 'operation' (from DB).
    """

    timer_id: str
    function: str
    step: Optional[int]
    time_ms: float
    line_index: int
    timer_type: str


# =============================================================================
# Fast string/format helpers
# =============================================================================


def _unescape_minimal(s: str) -> str:
    """Unescape a minimal set for readability (fast).

    Args:
        s: Raw C-string content.

    Returns:
        Minimally unescaped text.
    """
    import re

    def replace_escape(match):
        escape_seq = match.group(0)
        if escape_seq == '\\"':
            return '"'
        elif escape_seq == "\\\\":
            return "\\"
        elif escape_seq == "\\n":
            return "\n"
        elif escape_seq == "\\t":
            return "\t"
        else:
            return escape_seq  # Leave unknown sequences unchanged

    # Use regex to handle all escape sequences in one pass
    return re.sub(r"\\[\\\"nt]", replace_escape, s)


def _printf_to_regex(fmt: str) -> Tuple[str, str]:
    """Convert printf-like format to a tolerant regex fragment.

    Args:
        fmt: Joined C-format string (first arg to message).

    Returns:
        (label_text, regex_fragment) for the body after 'FUNCTION: '.
    """
    label = re.sub(r"\s+", " ", fmt).strip()
    body = _PRINTF_SPEC.sub(".*?", fmt)
    body = re.escape(body).replace(r"\.\*\?", ".*?")
    body = re.sub(r"(?:\\\s|\\ )+", r"\\s+", body)
    return label, body


def _classify_timer_type(fmt_text: str) -> str:
    """Classify timer type - simplified to collect all timers.

    All timers are initially classified generically. Actual classification into
    function vs operation timers happens during analysis using the nesting
    database and intelligent heuristics.

    Args:
        fmt_text: The format string, e.g. "took %.3f %s" or "Rebuilt cells
            took %.3f %s"

    Returns:
        Always returns 'timer' - actual classification happens during analysis
    """
    # All timers are collected generically - classification happens during
    # analysis using nesting database guidance and intelligent heuristics
    return "timer"


def _build_log_pattern(function: str, fmt_text: str) -> Tuple[str, str]:
    """Create a site-specific regex capturing `<ms>` as group 1.

    Args:
        function: Function name printed by the macro.
        fmt_text: Joined first argument string (must contain 'took').

    Returns:
        (label_text, full_regex_pattern).
    """
    label, body = _printf_to_regex(fmt_text)

    # The body contains format patterns like "took\s+.*?\s+.*?\." where the
    # first .*? should capture the numeric value and the second .*? should be
    # "ms" We need to replace the pattern around "took" to capture the number

    if r"took\s+.*?\s+.*?" in body:
        # Pattern like "took %.3f %s" -> "took\s+.*?\s+.*?"
        # Replace with pattern that captures the number: "took\s+([\d.]+)\s+ms"
        body = re.sub(
            r"took\\s\+\.\*\?\\s\+\.\*\?", r"took\\s+([\\d.]+)\\s+ms", body
        )
    elif r"took\s+.*?" in body:
        # Simpler pattern like "took %f" -> "took\s+.*?"
        body = re.sub(r"took\\s\+\.\*\?", r"took\\s+([\\d.]+)\\s+ms", body)

    pattern = (
        r"^(?:\[\d{4}\]\s+)?"  # optional [rank]
        r"\S+\s+"  # elapsed token
        + re.escape(function)
        + r":\s+"
        + body
        + r"\.?\s*$"
    )
    return label, pattern


def _extract_function_name_from_header(header: str) -> Optional[str]:
    """Best-effort function-name extraction: last identifier before first '('.

    Args:
        header: Concatenated header text containing '(' and '{'.

    Returns:
        Function name or None.
    """
    try:
        pos = header.index("(")
    except ValueError:
        return None
    left = header[:pos]
    ids = re.findall(r"[A-Za-z_]\w*", left)

    if not ids:
        return None

    # Filter out common C attributes, keywords, and modifiers that shouldn't
    # be function names
    filtered_ids = []
    skip_keywords = {
        "__attribute__",
        "INLINE",
        "static",
        "extern",
        "inline",
        "const",
        "volatile",
        "restrict",
        "void",
        "int",
        "float",
        "double",
        "char",
        "short",
        "long",
        "signed",
        "unsigned",
        "struct",
        "union",
        "enum",
        "typedef",
        "auto",
        "register",
        "__inline__",
        "__restrict__",
        "__const__",
        "__volatile__",
        "always_inline",
        "if",
        "for",
        "while",
        "switch",
        "return",
        "case",
        "default",
        "else",
        "do",
        "goto",
        "break",
        "continue",
        "sizeof",
        "typeof",
        "__typeof__",
    }

    for identifier in ids:
        if identifier not in skip_keywords:
            filtered_ids.append(identifier)

    # Return the last non-keyword identifier (most likely to be the function
    # name)
    return filtered_ids[-1] if filtered_ids else None


# =============================================================================
# Core parser
# =============================================================================


class CTimerParser:
    """Fast, streaming parser that collects and classifies timers.

    The parser:
      * Walks files under `src_dir` matching C/C++ extensions,
      * Streams each file line-by-line,
      * Detects functions with a cheap header heuristic + brace depth,
      * Inside functions:
          - Records `getticks()` assignments (line + var name),
          - Records `message("...took...")` calls (end line),
          - Associates each message with the nearest sensible tic,
          - Classifies timers as 'function' or 'operation' types.

    The YAML DB contains all info needed to match logs and classify timer
    types.
    """

    def __init__(self, src_dir: str) -> None:
        """Initialize the parser.

        Args:
            src_dir: Root of the source tree (walked recursively).
        """
        self.src_dir = os.path.abspath(src_dir)
        self.yaml_out = os.path.abspath(TIMER_FILE)
        self.exts = {
            ".c",
            ".h",
            ".hpp",
            ".hh",
            ".hxx",
            ".cc",
            ".cpp",
            ".cxx",
            ".C",
        }
        self._yaml = YAML()
        self._yaml.default_flow_style = False

    def parse(self) -> Dict:
        r"""Parse all files and write the YAML timer database.

        Returns:
            In-memory payload that is also written to YAML. Shape:

            {
              "version": 1,
              "project_root": "<abs>",
              "timers": [
                {
                  "timer_id": "rel/file.c:LINE",
                  "file": "rel/file.c",
                  "function": "func",
                  "start_line": 120,
                  "end_line": 150,
                  "tic_var": "tic",
                  "label_text": "...",
                  "log_pattern": "^(?:\\[\\d{4}\\]...)... took ([\\d.]+) ms",
                  "timer_type": "function" or "operation"
                },
                ...
              ],
              "stats": {
                "files_scanned": 42,
                "timers_found": 123,
                "functions_found": 67,
                "function_timers": 45,
                "operation_timers": 78
              }
            }
        """
        files: List[str] = []
        for root, _, names in os.walk(self.src_dir):
            for n in names:
                if os.path.splitext(n)[1] in self.exts:
                    files.append(os.path.join(root, n))

        timers: List[TimerSite] = []
        functions: set[str] = set()

        for path in tqdm(
            files,
            desc="Parsing files",
            unit="file",
        ):
            rel = os.path.relpath(path, self.src_dir)
            self._parse_one_file(path, rel, functions, timers)

        payload = {
            "version": 1,
            "project_root": self.src_dir,
            "timers": [
                {
                    "timer_id": t.timer_id,
                    "file": t.file,
                    "function": t.function,
                    "start_line": t.start_line,
                    "end_line": t.end_line,
                    "tic_var": t.tic_var,
                    "label_text": t.label_text,
                    "log_pattern": t.log_pattern,
                    "timer_type": t.timer_type,
                }
                for t in timers
            ],
            "stats": {
                "files_scanned": len(files),
                "timers_found": len(timers),
                "functions_found": len(functions),
                "function_timers": len(
                    [t for t in timers if t.timer_type == "function"]
                ),
                "operation_timers": len(
                    [t for t in timers if t.timer_type == "operation"]
                ),
            },
        }

        os.makedirs(os.path.dirname(self.yaml_out) or ".", exist_ok=True)
        with open(self.yaml_out, "w", encoding="utf-8") as f:
            self._yaml.dump(payload, f)
        return payload

    def _parse_one_file(
        self,
        abs_path: str,
        rel_path: str,
        functions: set[str],
        timers: List[TimerSite],
    ) -> None:
        """Stream one file to discover functions and timers.

        Args:
            abs_path: Absolute file path.
            rel_path: Path relative to parse root (used in timer_id/file).
            functions: Set of discovered function names (mutated).
            timers: Output list of found TimerSite entries (mutated).
        """
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except OSError:
            return

        i, n = 0, len(lines)
        in_func = False
        func_name: Optional[str] = None
        brace = 0

        # Function-scope tic variables seen so far: list of (line, varname)
        tic_points: List[Tuple[int, str]] = []

        while i < n:
            line = lines[i]
            s = line.strip()

            if not in_func:
                # Cheap function header heuristic: has '(' and eventually
                # '{', not a control stmt or comment.
                if "(" in s and not s.startswith(
                    ("#", "if", "for", "while", "switch", "return", "*", "//")
                ):
                    header = s
                    par = s.count("(") - s.count(")")
                    j = i
                    brace_seen = "{" in s
                    while (par > 0 or not brace_seen) and j + 1 < n:
                        j += 1
                        s2 = lines[j].strip()
                        header += " " + s2
                        par += s2.count("(") - s2.count(")")
                        if "{" in s2:
                            brace_seen = True
                    if (
                        par == 0
                        and brace_seen
                        and ";" not in header.split("{", 1)[0]
                    ):
                        name = _extract_function_name_from_header(header)
                        if name:
                            functions.add(name)
                            in_func = True
                            func_name = name
                            # Count braces in the complete header to set
                            # initial brace depth
                            brace = header.count("{") - header.count("}")
                            tic_points = []
                            i = j + 1
                            continue
                i += 1
                continue

            # Adjust brace depth
            brace += line.count("{")
            brace -= line.count("}")
            if brace <= 0:
                in_func = False
                func_name = None
                tic_points = []
                i += 1
                continue

            # Track getticks() assignments (tic points)
            if "getticks" in line and "=" in line:
                m = re.search(
                    r"([A-Za-z_]\w*)\s*=\s*getticks\s*\(\s*\)\s*;", line
                )
                if m:
                    tic_points.append((i + 1, m.group(1)))  # 1-based line

            # Detect `message(` occurrence
            pos = line.find("message(")
            if pos != -1:
                call_text, end_line = _scan_balanced_call(
                    lines, i, pos + len("message")
                )
                # Extract first-argument concatenated strings
                m0 = re.match(
                    r"^\s*((?:\"(?:[^\"\\]|\\.)*\"\s*)+)", call_text, re.S
                )
                if m0:
                    parts = _RE_CSTRING.findall(m0.group(1))
                    fmt_text = _unescape_minimal("".join(parts))
                    if "took" in fmt_text:
                        # Choose tic var: prefer the last tic whose var name
                        # appears in the args
                        arg_idents = set(
                            re.findall(r"\b([A-Za-z_]\w*)\b", call_text)
                        )
                        start_line: Optional[int] = None
                        tic_var: Optional[str] = None
                        if tic_points:
                            # If we can find a named tic inside args, take the
                            # nearest such tic
                            for ln, v in reversed(tic_points):
                                if v in arg_idents and ln <= i + 1:
                                    start_line = ln
                                    tic_var = v
                                    break
                            # Else, fall back to the last tic before message
                            if start_line is None:
                                ln, v = tic_points[-1]
                                if ln <= i + 1:
                                    start_line = ln
                                    tic_var = v

                        label, pattern = _build_log_pattern(
                            func_name or "unknown", fmt_text
                        )
                        timer_type = _classify_timer_type(fmt_text)
                        timers.append(
                            TimerSite(
                                timer_id=f"{rel_path}:{i + 1}",
                                file=rel_path,
                                function=func_name or "unknown",
                                start_line=start_line,
                                end_line=i + 1,
                                tic_var=tic_var,
                                label_text=label,
                                log_pattern=pattern,
                                timer_type=timer_type,
                            )
                        )
                # Jump to end of the call for speed
                i = end_line + 1
                continue

            i += 1


def _scan_balanced_call(
    lines: List[str], start_idx: int, start_col: int
) -> Tuple[str, int]:
    """Extract text inside `message( ... )` starting at line/column and return.

    This tiny scanner balances parentheses while ignoring content in C string
    literals.

    Args:
        lines: File lines.
        start_idx: Line index where 'message(' appears.
        start_col: Column index just after 'message'.

    Returns:
        Tuple of (call_text_inside_parens, end_line_index).
    """
    par = 0
    ins = False
    esc = False
    buf: List[str] = []
    i = start_idx
    j = start_col
    while i < len(lines):
        line = lines[i]
        while j < len(line):
            ch = line[j]
            if ins:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    ins = False
                j += 1
                continue
            if ch == '"':
                ins = True
                buf.append(ch)
            elif ch == "(":
                par += 1
                if par > 1:
                    buf.append(ch)
            elif ch == ")":
                if par > 1:
                    buf.append(ch)
                par -= 1
                if par == 0:
                    return "".join(buf), i
            else:
                buf.append(ch)
            j += 1
        i += 1
        j = 0
        if i < len(lines):
            buf.append("\n")
    return "".join(buf), i


def load_timer_db() -> Dict[str, TimerDef]:
    """Load timer definitions from YAML.

    Returns:
        Mapping from `timer_id` to `TimerDef` for quick access in analysis.
    """
    # If file doesn't exist make it
    if not TIMER_FILE.exists():
        config = load_swift_profile()
        os.makedirs(os.path.dirname(TIMER_FILE) or ".", exist_ok=True)
        if config.swiftsim_dir is not None:
            parse_src_timers(str(config.swiftsim_dir))

    yaml_safe = YAML(typ="safe")
    with open(TIMER_FILE, "r", encoding="utf-8") as f:
        y = yaml_safe.load(f) or {}

    db: Dict[str, TimerDef] = {}
    for t in y.get("timers") or []:
        db[t["timer_id"]] = TimerDef(
            timer_id=t["timer_id"],
            function=t["function"],
            log_pattern=t["log_pattern"],
            start_line=t.get("start_line"),
            end_line=int(t["end_line"]),
            label_text=t.get("label_text", "Unknown"),
            timer_type=t.get(
                "timer_type", "operation"
            ),  # default to operation for old files
        )

    return db


def compile_site_patterns(
    timer_db: Dict[str, TimerDef],
) -> List[Tuple[str, re.Pattern]]:
    """Compile per-site log-line regex patterns.

    This function implements fail-fast behavior for corrupted timer patterns.
    Invalid regex patterns indicate data corruption that would cause incorrect
    timing analysis, so errors are raised rather than silently skipped.

    Args:
        timer_db: Mapping from timer_id to TimerDef.

    Returns:
        List of (timer_id, compiled_regex).

    Raises:
        ValueError: If any timer pattern fails to compile. This indicates
            corrupted timer database requiring regeneration from source.
    """
    out: List[Tuple[str, re.Pattern]] = []
    for tid, td in timer_db.items():
        try:
            out.append((tid, re.compile(td.log_pattern)))
        except re.error as e:
            raise ValueError(
                f"Invalid regex pattern for timer '{tid}': {e}\n"
                f"Pattern: {td.log_pattern}\n"
                f"This indicates corrupted timer database. "
                f"Try regenerating with fresh source parsing."
            ) from e
    return out


def scan_log_instances_by_step(
    log_file: str,
    compiled: List[Tuple[str, re.Pattern]],
    timer_db: Dict[str, TimerDef],
) -> Tuple[Dict[Optional[int], List[TimerInstance]], List[Tuple[int, int]]]:
    """Linear scan of the log, mapping matching lines to timer instances.

    Args:
        log_file: Path to SWIFT log.
        compiled: List of (timer_id, compiled_regex).
        timer_db: Timer definitions.

    Returns:
        (instances_by_step, step_lines) where `instances_by_step[step]` is in
            encounter order.
    """
    instances_by_step: Dict[Optional[int], List[TimerInstance]] = {}
    step_lines: List[Tuple[int, int]] = []

    # First, count total lines for progress tracking
    print("  Counting log file lines...")
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        total_lines = sum(1 for _ in f)

    print(
        f"  Processing {total_lines:,} lines against "
        f"{len(compiled)} patterns..."
    )

    # Pre-compile a check for "took" to quickly skip irrelevant lines
    took_pattern = re.compile(r"\btook\s+[\d.]+\s+ms")

    current_step = None  # Track the current step for timer association

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for idx, raw in tqdm(
            enumerate(f), total=total_lines, desc="  Scanning log", unit="line"
        ):
            line = raw.rstrip("\n")

            # Check if this line defines a new step
            sm = _RE_STEP_LINE.match(line)
            if sm:
                current_step = int(sm.group("step"))
                step_lines.append((idx, current_step))

            # Quick pre-filter: skip lines that don't contain timing info
            if not took_pattern.search(line):
                continue

            # Now test against all compiled patterns
            for tid, creg in compiled:
                m = creg.search(line)
                if not m:
                    continue
                try:
                    val_ms = float(m.group(1))
                except Exception:
                    break

                instances_by_step.setdefault(current_step, []).append(
                    TimerInstance(
                        timer_id=tid,
                        function=timer_db[tid].function,
                        step=current_step,
                        time_ms=val_ms,
                        line_index=idx,
                        timer_type=timer_db[tid].timer_type,
                    )
                )
                break  # one site per line
    return instances_by_step, step_lines


def parse_src_timers(src_dir: str) -> Dict:
    """Convenience function to parse source tree and write timers.yaml.

    Args:
        src_dir: Root of the source tree (walked recursively).

    Returns:
        In-memory payload that is also written to YAML.
    """
    parser = CTimerParser(src_dir)
    return parser.parse()


class TimerNestingGenerator:
    """Automatic generator for timer nesting relationships."""

    def __init__(self, src_dir: str, timer_data: List[Dict]):
        """Initialize with source directory and timer data.

        Args:
            src_dir: Root of the source tree to analyze
            timer_data: List of timer dictionaries from parse_src_timers
        """
        self.src_dir = Path(src_dir)
        self.timer_data = timer_data

        # Initialize tree-sitter parser
        self.parser = tree_sitter.Parser(
            tree_sitter.Language(tree_sitter_c.language())
        )

        # Build lookup tables
        self._build_function_maps()

    def _build_function_maps(self):
        """Build function name mappings from timer data."""
        # Map function names to their timer definitions
        self.function_timers = {}  # function_name -> timer_dict
        self.function_operations = {}  # function_name -> List[timer_dict]

        # Group all timers by function first
        timers_by_function = {}
        for timer_dict in self.timer_data:
            func_name = timer_dict["function"]
            if func_name not in timers_by_function:
                timers_by_function[func_name] = []
            timers_by_function[func_name].append(timer_dict)

        # For each function, identify the function-level timer based on label
        # Function timers have generic labels like "took %.3f %s."
        # Operation timers have specific descriptive text
        for func_name, timers in timers_by_function.items():
            function_timer = None
            operation_timers = []

            # Look for a timer with generic "took" label (function timer)
            for timer_dict in timers:
                label_text = timer_dict.get("label_text", "")
                if label_text in ["took %.3f %s.", "took %.3f %s"]:
                    function_timer = timer_dict
                else:
                    operation_timers.append(timer_dict)

            # If no generic timer found, use the one with the highest end_line
            # (last timer in function)
            if not function_timer and timers:
                function_timer = max(
                    timers, key=lambda t: t.get("end_line", 0)
                )
                operation_timers = [t for t in timers if t != function_timer]

            if function_timer:
                self.function_timers[func_name] = function_timer
            if operation_timers:
                self.function_operations[func_name] = operation_timers

    def generate_nesting_database(self) -> Dict:
        """Generate complete nesting database in YAML format.

        Returns:
            Dictionary suitable for writing to timer_nesting.yaml
        """
        nesting_data: dict[str, Any] = {
            "version": 1,
            "description": "Auto-generated timer nesting relationships "
            "for SWIFT",
            "created_from_analysis": "Automatic source code analysis of "
            f"{self.src_dir}/",
            "nesting": {},
        }

        # Process each function that has a function timer or operation timers
        all_timer_functions = set(self.function_timers.keys()) | set(
            self.function_operations.keys()
        )
        for func_name in all_timer_functions:
            function_data = self._analyze_function(func_name)
            if function_data:
                nesting_data["nesting"][func_name] = function_data

        return nesting_data

    def _analyze_function(self, func_name: str) -> Optional[Dict]:
        """Analyze a single function for nesting relationships.

        Args:
            func_name: Name of the function to analyze

        Returns:
            Dictionary with function timer, operations, and nested functions
        """
        timer_dict = self.function_timers.get(func_name)

        # If no function timer, check if there are operation timers
        if not timer_dict:
            operation_timers = self.function_operations.get(func_name, [])
            if not operation_timers:
                return None
            # Use the first operation timer to get the file info
            timer_dict = operation_timers[0]

        # Find the source file containing this function
        source_file = self._find_function_source_file(
            func_name, timer_dict["file"]
        )
        if not source_file:
            return None

        # Parse function calls
        nested_functions = self._parse_function_calls(source_file, func_name)

        # Get operation timers for this function
        nested_operations = []
        if func_name in self.function_operations:
            nested_operations = [
                op["label_text"] for op in self.function_operations[func_name]
            ]

        # Check if this is a true function timer or just an operation timer
        has_function_timer = func_name in self.function_timers

        return {
            "function_timer": timer_dict["label_text"]
            if has_function_timer
            else None,
            "file": timer_dict["file"],
            "nested_operations": nested_operations,
            "nested_functions": nested_functions,
        }

    def _find_function_source_file(
        self, func_name: str, relative_file: str
    ) -> Optional[Path]:
        """Find the actual source file for a function.

        Args:
            func_name: Name of the function
            relative_file: Relative file path from timer database

        Returns:
            Full path to source file if found
        """
        full_path = self.src_dir / relative_file
        if full_path.exists():
            return full_path
        return None

    def _parse_function_calls(
        self, source_file: Path, func_name: str
    ) -> List[str]:
        """Parse function calls within a specific function using tree-sitter.

        Args:
            source_file: Path to the source file
            func_name: Name of the function to analyze

        Returns:
            List of function names that are called and have timers
        """
        try:
            with open(source_file, "rb") as f:
                source_code = f.read()
        except Exception:
            return []

        tree = self.parser.parse(source_code)

        # Find the function definition
        function_node = self._find_function_definition(
            tree.root_node, func_name, source_code
        )
        if not function_node:
            return []

        # Extract function calls within this function
        function_calls = self._extract_function_calls(
            function_node, source_code
        )

        # Filter to only include functions that have timers (either function
        # or operation timers)
        timer_functions = set(self.function_timers.keys())
        operation_functions = set(self.function_operations.keys())
        all_timer_functions = timer_functions | operation_functions
        nested_functions = [
            call
            for call in function_calls
            if call in all_timer_functions and call != func_name
        ]

        return sorted(
            list(set(nested_functions))
        )  # Remove duplicates and sort

    def _find_function_definition(
        self, node, func_name: str, source_code: bytes
    ):
        """Find the tree-sitter node for a specific function definition."""
        if node.type == "function_definition":
            # Look for the function name in the declarator
            declarator = None
            for child in node.children:
                if child.type == "function_declarator":
                    declarator = child
                    break

            if declarator:
                # Find the identifier (function name)
                for child in declarator.children:
                    if child.type == "identifier":
                        name = source_code[
                            child.start_byte : child.end_byte
                        ].decode("utf-8")
                        if name == func_name:
                            return node

        # Recursively search children
        for child in node.children:
            result = self._find_function_definition(
                child, func_name, source_code
            )
            if result:
                return result

        return None

    def _extract_function_calls(
        self, function_node, source_code: bytes
    ) -> List[str]:
        """Extract all function calls from within a function body."""
        function_calls = []

        def traverse(node):
            if node.type == "call_expression":
                # Get the function name from the call
                function_expr = node.children[
                    0
                ]  # First child is the function expression
                if function_expr.type == "identifier":
                    func_name = source_code[
                        function_expr.start_byte : function_expr.end_byte
                    ].decode("utf-8")
                    function_calls.append(func_name)

            # Recursively traverse children
            for child in node.children:
                traverse(child)

        traverse(function_node)
        return function_calls


def generate_timer_nesting_database(
    src_dir: str, timer_data: Optional[List[Dict]] = None
) -> Dict:
    """Generate timer nesting database automatically.

    Args:
        src_dir: Root directory of source code to analyze
        timer_data: Timer data list (if None, will generate from src_dir)

    Returns:
        Dictionary suitable for writing to timer_nesting.yaml
    """
    if timer_data is None:
        timer_parsed = parse_src_timers(src_dir)
        timer_data = timer_parsed["timers"]

    generator = TimerNestingGenerator(src_dir, timer_data)
    return generator.generate_nesting_database()


@dataclass
class TaskCountSnapshot:
    """One engine_print_task_counts block discovered in a log.

    This represents the four-line summary printed once per step by
    engine_print_task_counts().
    """

    step: Optional[int]
    rank: int
    sim_time: float
    system_total: Optional[int]
    num_cells: Optional[int]
    total_tasks: Optional[int]
    per_cell_avg: Optional[float]
    per_cell_max: Optional[float]
    counts: Dict[str, int]
    line_index: int


# engine_print_task_counts:
# Matches both formats:
#   [rank] [time] engine_print_task_counts: ...
#   [time] engine_print_task_counts: ...
# Captures rank (optional, defaults to 0) and time
_RE_ENGINE_TASK_COUNTS = re.compile(
    r"^(?:\[(?P<rank>\d+)\]\s+)?\[(?P<time>[0-9.]+)\]"
    r"\s+engine_print_task_counts:\s*(?P<body>.*)$"
)

_RE_ENGINE_TASK_HEADER = re.compile(
    r"^System total:\s*(?P<system_total>\d+)\s*,"
    r"\s*no\. cells:\s*(?P<cells>\d+)\s*$"
)

_RE_ENGINE_TASK_TOTAL_PER_CELL = re.compile(
    r"^Total\s*=\s*(?P<total>\d+)\s*\(per cell\s*="
    r"\s*(?P<per_cell>[0-9.eE+\-]+)\s*\)\s*$"
)

_RE_ENGINE_TASK_TOTAL_MAX_PER_CELL = re.compile(
    r"^Total\s*=\s*(?P<total>\d+)\s*\(maximum per"
    r" cell\s*=\s*(?P<per_cell>[0-9.eE+\-]+)\s*\)\s*$"
)

_RE_ENGINE_TASK_COUNTS_BODY = re.compile(
    r"^task counts are\s*\[(?P<body>.*)\]\s*$"
)

# Regex to match step table data lines (start with spaces, then step number)
_RE_STEP_TABLE_LINE = re.compile(
    r"^\s+(?P<step>\d+)\s+(?P<time>[0-9.eE+\-]+)\s+"
)


def scan_task_counts_by_step(
    log_file: str,
) -> Tuple[
    Dict[Optional[int], List[TaskCountSnapshot]], List[Tuple[int, int]]
]:
    """Scan a SWIFT log for engine_print_task_counts summaries.

    Args:
        log_file: Path to the SWIFT log file.

    Returns:
        (snapshots_by_step, step_lines) where:

          * snapshots_by_step[step] is a list of TaskCountSnapshot instances
            in encounter order (step may be None),
          * step_lines contains (line_index, step) pairs for every step
            header line.
    """
    snapshots_by_step: Dict[Optional[int], List[TaskCountSnapshot]] = {}
    step_lines: List[Tuple[int, int]] = []

    print("  Counting log file lines for task count scan...")
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        total_lines = sum(1 for _ in f)

    print(
        f"  Processing {total_lines:,} lines to extract "
        "engine_print_task_counts blocks..."
    )

    # First pass: build step → simulation time mapping from step table
    print("  Extracting simulation times from step table...")
    step_to_sim_time: Dict[int, float] = {}
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _RE_STEP_TABLE_LINE.match(line)
            if m:
                step = int(m.group("step"))
                sim_time = float(m.group("time"))
                step_to_sim_time[step] = sim_time

    print(f"  Found simulation times for {len(step_to_sim_time)} steps")

    current_step: Optional[int] = None
    current_block: Optional[Dict[str, Any]] = None

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for idx, raw in tqdm(
            enumerate(f),
            total=total_lines,
            desc="  Scanning task counts",
            unit="line",
        ):
            line = raw.rstrip("\n")

            # Track the current step (first integer token on line)
            sm = _RE_STEP_LINE.match(line)
            if sm:
                current_step = int(sm.group("step"))
                step_lines.append((idx, current_step))

            # Look for engine_print_task_counts lines
            m = _RE_ENGINE_TASK_COUNTS.match(line)
            if not m:
                continue

            body = m.group("body").strip()

            # Extract rank (defaults to 0 if not present)
            rank_str = m.group("rank")
            rank = int(rank_str) if rank_str is not None else 0

            # Get simulation time from step table, fallback to wallclock
            # if unavailable
            if current_step is not None and current_step in step_to_sim_time:
                sim_time = step_to_sim_time[current_step]
            else:
                # Fallback: use wallclock time from [time] field
                sim_time = float(m.group("time"))

            # Header line: "System total: ..., no. cells: ..."
            header_m = _RE_ENGINE_TASK_HEADER.match(body)
            if header_m:
                # Drop unfinished block; we only commit once we see
                # "task counts are [...]".
                current_block = {
                    "step": current_step,
                    "rank": rank,
                    "sim_time": sim_time,
                    "system_total": int(header_m.group("system_total")),
                    "num_cells": int(header_m.group("cells")),
                    "total_tasks": None,
                    "per_cell_avg": None,
                    "per_cell_max": None,
                    "counts": {},
                    "line_index": idx,
                }
                continue

            # If we see a non-header engine_print_task_counts line
            # without an existing block, start a minimal one so we can
            # still attach counts.
            if current_block is None:
                current_block = {
                    "step": current_step,
                    "rank": rank,
                    "sim_time": sim_time,
                    "system_total": None,
                    "num_cells": None,
                    "total_tasks": None,
                    "per_cell_avg": None,
                    "per_cell_max": None,
                    "counts": {},
                    "line_index": idx,
                }

            # "Total = ... (per cell = ...)"
            avg_m = _RE_ENGINE_TASK_TOTAL_PER_CELL.match(body)
            if avg_m:
                try:
                    current_block["total_tasks"] = int(avg_m.group("total"))
                except ValueError:
                    pass
                try:
                    current_block["per_cell_avg"] = float(
                        avg_m.group("per_cell")
                    )
                except ValueError:
                    pass
                continue

            # "Total = ... (maximum per cell = ...)"
            max_m = _RE_ENGINE_TASK_TOTAL_MAX_PER_CELL.match(body)
            if max_m:
                try:
                    current_block["total_tasks"] = int(max_m.group("total"))
                except ValueError:
                    pass
                try:
                    current_block["per_cell_max"] = float(
                        max_m.group("per_cell")
                    )
                except ValueError:
                    pass
                continue

            # "task counts are [ name=value ... ]"
            counts_m = _RE_ENGINE_TASK_COUNTS_BODY.match(body)
            if counts_m:
                counts_body = counts_m.group("body")
                counts: Dict[str, int] = {}
                for token in counts_body.split():
                    if "=" not in token:
                        continue
                    name, val = token.split("=", 1)
                    name = name.strip()
                    val = val.strip().rstrip(",")
                    if not name:
                        continue
                    try:
                        counts[name] = int(val)
                    except ValueError:
                        continue

                current_block["counts"] = counts

                snap = TaskCountSnapshot(
                    step=current_block["step"],
                    rank=current_block["rank"],
                    sim_time=current_block["sim_time"],
                    system_total=current_block["system_total"],
                    num_cells=current_block["num_cells"],
                    total_tasks=current_block["total_tasks"],
                    per_cell_avg=current_block["per_cell_avg"],
                    per_cell_max=current_block["per_cell_max"],
                    counts=current_block["counts"],
                    line_index=current_block["line_index"],
                )

                snapshots_by_step.setdefault(current_block["step"], []).append(
                    snap
                )
                current_block = None
                continue

            # Any other engine_print_task_counts variants are ignored.

    return snapshots_by_step, step_lines
