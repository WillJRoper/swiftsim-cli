"""A module containing the parameter reader and parameter holder."""

from pathlib import Path

from ruamel.yaml import YAML

PARAMS: dict | None = None
PARAMS_SOURCE: Path | None | object = None
PARAMS_VERSION = 0

# Configure YAML for round-trip comment preservation and consistent formatting
yaml = YAML()
yaml.default_flow_style = False
yaml.indent(mapping=4, sequence=4, offset=2)
yaml.width = 80
yaml.allow_unicode = True


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

    # Parse YAML safely
    try:
        return yaml.load(cleaned) or {}
    except Exception as e:
        # Add context and re-raise
        raise ValueError(f"Error parsing YAML in '{param_file}': {e}") from e


def load_parameters(param_file: Path | None = None) -> dict:
    """Load and cache parameters from a YAML file.

    Args:
        param_file: Optional Path to the parameter YAML file.

    Returns:
        A dict of loaded parameters (cached after first load).

    Raises:
        FileNotFoundError: If the path is provided but does not exist.
        IOError:           If the file cannot be read.
        ValueError:        If parsing fails.
    """
    global PARAMS, PARAMS_SOURCE, PARAMS_VERSION

    requested_source: Path | None | object
    if param_file is None:
        requested_source = None
    else:
        requested_source = param_file.resolve()

    if PARAMS is not None and PARAMS_SOURCE == requested_source:
        return PARAMS

    if param_file is None:
        clear_parameter_cache()
        return {}

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
