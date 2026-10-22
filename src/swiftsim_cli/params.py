"""A module containing the parameter reader and parameter holder."""

from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.constructor import DuplicateKeyError

from swiftsim_cli.utilities.messages import warn

# Limits of SWIFT's parameter parser (src/parser.h); a longer line is
# split by fgets and the remainder rejected
SWIFT_MAX_LINE_LENGTH = 255
SWIFT_MAX_PARAMS = 700
SWIFT_MAX_SECTIONS = 64

# Values in SWIFT's parameter_example.yml that only document a default
PLACEHOLDER_VALUES = {"FLT_MAX", "DBL_MAX", "INT_MAX", "LONG_MAX"}

PARAMS: dict | None = None
PARAMS_SOURCE: Path | None = None
PARAMS_VERSION = 0


def round_trip_yaml() -> YAML:
    """Create a YAML instance for reading and writing SWIFT parameter files.

    The instance preserves comments on round trips and uses the
    formatting conventions of SWIFT's example parameter files.

    Returns:
        A configured round-trip YAML instance.
    """
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=4, sequence=4, offset=2)
    # SWIFT reads each "key: value" from a single line, so never wrap
    yaml.width = 4096
    yaml.allow_unicode = True
    return yaml


yaml = round_trip_yaml()


def _clean_yaml_text(text: str, spaces_per_tab: int = 4) -> str:
    """Replace all tab characters with spaces in the YAML text.

    Args:
        text:          Raw text from the YAML file.
        spaces_per_tab: Number of spaces to use for each tab.

    Returns:
        A new string where every tab character is replaced by the
        specified number of spaces.
    """
    return text.expandtabs(spaces_per_tab)


def _parse_parameters(param_file: Path) -> dict:
    """Parse parameters from a YAML file.

    Args:
        param_file: Path to the parameter YAML file.

    Returns:
        A dict of parsed parameters.

    Raises:
        IOError:   If the file cannot be read.
        ValueError: If YAML parsing fails, with context on filename.
    """
    # Read raw content using a context manager
    try:
        with param_file.open("r", encoding="utf-8") as file:
            raw = file.read()
    except Exception as e:
        raise IOError(f"Could not read parameter file '{param_file}': {e}")

    # Clean leading tabs
    cleaned = _clean_yaml_text(raw)

    # Parse the YAML. SWIFT rejects duplicate keys, but we may be reading
    # someone else's file (SWIFT's own parameter_example.yml has one), so
    # a duplicate is a warning and the first value is kept.
    try:
        try:
            params = yaml.load(cleaned)
        except DuplicateKeyError as e:
            line = e.problem_mark.line + 1 if e.problem_mark else "?"
            warn(
                f"{param_file}:{line}: {e.problem}; using the first value "
                "(SWIFT itself rejects duplicate keys)"
            )
            lenient = round_trip_yaml()
            lenient.allow_duplicate_keys = True
            params = lenient.load(cleaned)
    except Exception as e:
        raise ValueError(f"Error parsing YAML in '{param_file}': {e}") from e

    if params is None:
        return {}
    if not isinstance(params, dict):
        kind = "list" if isinstance(params, list) else "single value"
        raise ValueError(
            f"Parameter file '{param_file}' must contain a YAML mapping of "
            f"sections, not a {kind}"
        )
    return params


def load_parameters(param_file: Path | None = None) -> dict:
    """Load and cache parameters from a YAML file.

    Args:
        param_file: Optional Path to the parameter YAML file. If None, the
            currently cached parameters are returned (or an empty dict if
            nothing has been loaded).

    Returns:
        A dict of loaded parameters (cached after first load).

    Raises:
        FileNotFoundError: If the path is provided but does not exist.
        IOError:           If the file cannot be read.
        ValueError:        If parsing fails.
    """
    global PARAMS, PARAMS_SOURCE, PARAMS_VERSION

    # With no file requested, hand back whatever is already cached (the CLI
    # loads the user's file up-front, downstream callers just ask for it)
    if param_file is None:
        return PARAMS if PARAMS is not None else {}

    requested_source = param_file.resolve()
    if PARAMS is not None and PARAMS_SOURCE == requested_source:
        return PARAMS

    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    PARAMS = _parse_parameters(param_file)
    PARAMS_SOURCE = requested_source
    PARAMS_VERSION += 1
    return PARAMS


def clear_parameter_cache() -> None:
    """Clear the cached parameter state."""
    global PARAMS, PARAMS_SOURCE, PARAMS_VERSION
    PARAMS = None
    PARAMS_SOURCE = None
    PARAMS_VERSION += 1


def get_parameter_cache_version() -> int:
    """Return a monotonically increasing parameter cache version."""
    return PARAMS_VERSION


def check_swift_parameter_text(text: str) -> list[str]:
    """Check a parameter file against the limits of SWIFT's parser.

    SWIFT reads parameter files line by line (src/parser.c) and rejects
    long lines, duplicate sections or parameters, and more than 700
    parameters or 64 sections. Placeholder values such as FLT_MAX and
    empty values are reported too, since SWIFT cannot parse them as
    numbers.

    Args:
        text: The parameter file contents.

    Returns:
        One message per problem found (empty if the file is fine).
    """
    problems = []
    sections: set[str] = set()
    names: set[str] = set()
    section = ""
    for number, line in enumerate(text.splitlines(), start=1):
        if len(line) > SWIFT_MAX_LINE_LENGTH:
            problems.append(
                f"line {number} is {len(line)} characters long (SWIFT "
                f"reads at most {SWIFT_MAX_LINE_LENGTH})"
            )
        content = line.split("#", 1)[0].rstrip()
        if not content.strip() or content.strip() in ("---", "..."):
            continue
        key, _, value = content.partition(":")
        value = value.strip()
        if not line[0].isspace():
            if not value:
                section = key
                if key in sections:
                    problems.append(f"line {number}: duplicate section {key}")
                sections.add(key)
                continue
            section = ""
        name = f"{section}:{key.strip()}" if section else key.strip()
        if name in names:
            problems.append(f"line {number}: duplicate parameter {name}")
        names.add(name)
        if not value:
            problems.append(f"line {number}: {name} has no value")
        elif value in PLACEHOLDER_VALUES:
            problems.append(
                f"line {number}: {name} is the placeholder {value}; give "
                "a number or remove it"
            )
    if len(names) > SWIFT_MAX_PARAMS:
        problems.append(
            f"{len(names)} parameters (SWIFT reads at most {SWIFT_MAX_PARAMS})"
        )
    if len(sections) > SWIFT_MAX_SECTIONS:
        problems.append(
            f"{len(sections)} sections (SWIFT reads at most "
            f"{SWIFT_MAX_SECTIONS})"
        )
    return problems
