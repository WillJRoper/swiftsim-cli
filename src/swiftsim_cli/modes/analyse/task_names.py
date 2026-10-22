"""Task, subtype and cell-type names read from the SWIFT source.

Task-debug dumps store integer indices, and their meaning depends on the
SWIFT version that wrote them (the zoom branch, for instance, inserts a
``progeny`` subtype). The names are therefore parsed with tree-sitter
from ``taskID_names``/``subtaskID_names`` in ``src/task.c`` and the
``cell_types``/``cell_subtypes`` enums in ``src/cell.h`` of the active
profile's checkout (or ``--swift-src``). The result is cached in the
profile cache, keyed like the timer database by HEAD and the state of
uncommitted changes.

Without a source tree, the built-in tables in :mod:`.task_plot_metadata`
are used (upstream master's for dumps without cell columns, the zoom
branch's otherwise) with a warning.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from swiftsim_cli.profile import (
    ProfileError,
    get_profile_cache_dir,
    load_swift_profile,
)
from swiftsim_cli.timing.timer_db import (
    _load_cached,
    _store,
    _tree_sitter,
    source_key,
)
from swiftsim_cli.utilities.messages import note, warn

from .task_plot_metadata import (
    LEGACY_ZOOM_CELL_TYPES,
    TASK_TYPES,
    UPSTREAM_SUBTYPES,
    ZOOM_CELL_SUBTYPES,
    ZOOM_CELL_TYPES,
    ZOOM_SUBTYPES,
)

# Bump whenever the cached payload changes.
NAMES_VERSION = 1

_ARRAYS = {"taskID_names": "task_types", "subtaskID_names": "subtypes"}
_ENUMS = {"cell_types": "cell_types", "cell_subtypes": "cell_subtypes"}
_CELL_ALIASES = {"bkg": "Background", "buff": "Buffer"}


@dataclass(frozen=True)
class TaskNames:
    """Name tables for decoding one task dump.

    Attributes:
        task_types: Task type names by index.
        subtypes: Task subtype names by index.
        cell_types: Cell type names by index.
        cell_subtypes: Cell subtype names by index.
        source: Where the names came from (for messages).
    """

    task_types: tuple[str, ...]
    subtypes: tuple[str, ...]
    cell_types: tuple[str, ...]
    cell_subtypes: tuple[str, ...]
    source: str

    def label(self, task_type: int, subtype: int) -> str:
        """Return the ``type/subtype`` label of a task (just ``type``)."""
        name = _name(self.task_types, task_type)
        sub = _name(self.subtypes, subtype)
        return name if sub == "none" else f"{name}/{sub}"

    def cell_name(self, index: int, subtype: bool = False) -> str | None:
        """Return a cell (sub)type name; None for "no cell" (-1)."""
        if index < 0:
            return None
        return _name(self.cell_subtypes if subtype else self.cell_types, index)

    def cell_index(self, value: int | str, subtype: bool = False) -> int:
        """Resolve a cell (sub)type given by name or integer.

        Raises:
            ValueError: For an unknown name (listing the valid ones).
        """
        if isinstance(value, int):
            return value
        table = self.cell_subtypes if subtype else self.cell_types
        lookup = {name.lower(): i for i, name in enumerate(table)}
        lookup.update(
            {
                alias: lookup[full.lower()]
                for alias, full in _CELL_ALIASES.items()
                if full.lower() in lookup
            }
        )
        key = value.lower()
        if key not in lookup:
            kind = "cell subtype" if subtype else "cell type"
            raise ValueError(
                f"unknown {kind} '{value}'; valid names ({self.source}): "
                + ", ".join(name.lower() for name in table)
                + " (or an integer)"
            )
        return lookup[key]

    def validate_labels(self, labels: Iterable[str]) -> None:
        """Check ``type`` or ``type/subtype`` labels against the tables.

        Raises:
            ValueError: Naming the unknown labels and the valid names.
        """
        bad = []
        for label in labels:
            name, _, sub = label.partition("/")
            if name not in self.task_types or (
                sub and sub not in self.subtypes
            ):
                bad.append(label)
        if bad:
            raise ValueError(
                f"unknown task label(s) {', '.join(bad)}; labels are "
                "'type' or 'type/subtype' with types: "
                + ", ".join(self.task_types[1:])
                + "; subtypes: "
                + ", ".join(self.subtypes[1:])
                + f" ({self.source})"
            )


def _name(table: tuple[str, ...], index: int) -> str:
    """``table[index]``, or ``unknown_<index>`` when out of range."""
    return table[index] if 0 <= index < len(table) else f"unknown_{index}"


def builtin_names(has_cells: bool, max_cell_type: int = 0) -> TaskNames:
    """Built-in tables chosen by dump layout.

    Dumps with cell columns come from the zoom branch; a cell type of 3
    means an older zoom dump with a buffer cell type.
    """
    if not has_cells:
        return TaskNames(
            TASK_TYPES,
            UPSTREAM_SUBTYPES,
            ZOOM_CELL_TYPES,
            ZOOM_CELL_SUBTYPES,
            "built-in upstream tables",
        )
    legacy = max_cell_type >= len(ZOOM_CELL_TYPES)
    return TaskNames(
        TASK_TYPES,
        ZOOM_SUBTYPES,
        LEGACY_ZOOM_CELL_TYPES if legacy else ZOOM_CELL_TYPES,
        ZOOM_CELL_SUBTYPES,
        "built-in zoom tables" + (" (legacy cell types)" if legacy else ""),
    )


# =============================================================================
# Source parsing
# =============================================================================


def _walk(node: Any) -> Iterable[Any]:
    """Yield every node of a tree-sitter tree, depth first."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _cell_display(enumerator: str) -> str:
    """Display name of a cell enumerator, e.g. ``cell_type_bkg``."""
    short = re.sub(r"^cell_(?:sub)?type_", "", enumerator)
    return _CELL_ALIASES.get(short, short.capitalize())


def parse_source_names(task_c: bytes, cell_h: bytes | None) -> dict[str, Any]:
    """Parse the name arrays of task.c and the cell enums of cell.h.

    Returns:
        A dict with ``task_types`` and ``subtypes`` lists and, when cell.h
        defines them, ``cell_types`` and ``cell_subtypes``.

    Raises:
        ValueError: If task.c lacks either name array.
    """
    import tree_sitter

    language = _tree_sitter()[0]
    parser = tree_sitter.Parser(language)
    names: dict[str, Any] = {}

    for node in _walk(parser.parse(task_c).root_node):
        if node.type != "init_declarator":
            continue
        declarator = node.child_by_field_name("declarator")
        value = node.child_by_field_name("value")
        if declarator is None or value is None:
            continue
        match = re.search(rb"\b(\w+)\s*\[", declarator.text or b"")
        if match is None or match[1].decode() not in _ARRAYS:
            continue
        names[_ARRAYS[match[1].decode()]] = [
            "".join(
                (part.text or b"").decode()
                for part in child.children
                if part.type == "string_content"
            )
            for child in value.named_children
            if child.type == "string_literal"
        ]

    missing = [n for n in _ARRAYS if _ARRAYS[n] not in names]
    if missing:
        raise ValueError(f"src/task.c does not define {', '.join(missing)}")

    if cell_h is not None:
        for node in _walk(parser.parse(cell_h).root_node):
            if node.type != "enum_specifier":
                continue
            name = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            key = (name.text or b"").decode() if name is not None else ""
            if body is None or key not in _ENUMS:
                continue
            # ponytail: enumerators assumed sequential from 0 (true for
            # these enums); explicit "= N" values are ignored.
            names[_ENUMS[key]] = [
                _cell_display(
                    (e.child_by_field_name("name").text or b"").decode()
                )
                for e in body.named_children
                if e.type == "enumerator"
            ]
    return names


def load_source_names(root: Path) -> dict[str, Any]:
    """Names parsed from a SWIFT tree, cached per profile.

    Raises:
        ValueError: If ``root`` has no parsable src/task.c.
    """
    root = root.resolve()
    task_c = root / "src" / "task.c"
    if not task_c.is_file():
        raise ValueError(f"{root} has no src/task.c")
    key = f"n{NAMES_VERSION}-{source_key(root)}"
    cache = get_profile_cache_dir() / "task_names"
    cache.mkdir(parents=True, exist_ok=True)
    path = (
        cache
        / f"tree-{hashlib.sha1(str(root).encode()).hexdigest()[:12]}.json"
    )
    payload = _load_cached(path, key)
    if payload is None:
        cell_h = root / "src" / "cell.h"
        payload = parse_source_names(
            task_c.read_bytes(),
            cell_h.read_bytes() if cell_h.is_file() else None,
        )
        _store(path, key, payload)
    return payload


def resolve_task_names(
    swift_src: Path | None, has_cells: bool, max_cell_type: int = 0
) -> TaskNames:
    """Pick the name tables for a dump.

    Args:
        swift_src: Explicit SWIFT tree (``--swift-src``); errors if it has
            no src/task.c.
        has_cells: Whether the dump has cell type/subtype columns.
        max_cell_type: Largest cell type index in the dump.

    Returns:
        Names from the source tree if one is available, else the built-in
        tables matching the dump layout (with a warning).
    """
    root = swift_src
    if root is None:
        try:
            root = load_swift_profile().swiftsim_dir
        except ProfileError:
            root = None
    fallback = builtin_names(has_cells, max_cell_type)
    if root is not None:
        try:
            payload = load_source_names(Path(root))
        except (OSError, ValueError) as error:
            if swift_src is not None:
                raise ValueError(
                    f"cannot read task names from --swift-src: {error}"
                ) from error
            warn(f"could not read task names from {root}: {error}")
        else:
            cells = "cell_types" in payload
            if has_cells and not cells:
                note(
                    f"{root} defines no cell types; using the "
                    f"{fallback.source} for the cell columns"
                )
            elif has_cells and len(
                payload["cell_types"]
            ) <= max_cell_type < len(LEGACY_ZOOM_CELL_TYPES):
                warn(
                    f"the dump has cell type {max_cell_type}, which {root} "
                    "does not define; assuming an older zoom dump with "
                    "cell types " + ", ".join(LEGACY_ZOOM_CELL_TYPES)
                )
                cells = False
            return TaskNames(
                tuple(payload["task_types"]),
                tuple(payload["subtypes"]),
                tuple(payload["cell_types"]) if cells else fallback.cell_types,
                tuple(payload["cell_subtypes"])
                if cells
                else fallback.cell_subtypes,
                f"names from {root}",
            )
    warn(
        "no SWIFT source tree to read task names from; using the "
        f"{fallback.source}, which may not match the SWIFT version that "
        "wrote the dump (pass --swift-src or set up a profile)"
    )
    return fallback
