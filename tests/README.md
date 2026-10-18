# Test Suite

This directory contains the test suite for swiftsim-cli.

## Structure

```
tests/
├── conftest.py      # Shared fixtures (profile isolation, timer fixtures, ICs)
├── unit/
│   ├── conftest.py  # Git fixtures: a bare "remote", main clone, worktrees
│   └── test_*.py    # One file per module or mode (see below)
├── data/
│   ├── c_archetypes/  # Archetypal C syntax cases + expectations
│   ├── logs/          # Scrubbed excerpts of real SWIFT logs + synthetic logs
│   └── swift_src/     # Synthetic SWIFT-like C tree matching the logs
└── README.md
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -e ".[test]"
```

### Basic Usage

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src/swiftsim_cli --cov-report=html
```

Run specific test file:
```bash
pytest tests/unit/test_utilities.py
```

Run specific test:
```bash
pytest tests/unit/test_utilities.py::TestUtilityFunctions::test_create_ascii_table_basic
```

### Optional real-SWIFT smoke test

```bash
SWIFT_SRC=/path/to/swiftsim pytest tests/unit/test_timer_db_archetypes.py -k real
```

Parses a real SWIFT checkout (read-only) and checks sane invariants; it is
skipped unless `SWIFT_SRC` is set.

## Test Categories

### Unit Tests

- **test_cli.py**, **test_multi_mode_args_real.py**,
  **test_flag_conventions.py**: the entry point, multi-mode parsing, error
  handling, start-up imports and the flag conventions shared by every mode
- **test_cli_chains.py**: end-to-end chains of real modes through
  `cli.main` (outputs written, a failing mode stopping the chain)
- **test_readme.py**: the help text quoted in the top-level README matches
  the real output
- **test_profile.py**, **test_profile_worktrees.py**: profile storage and the
  worktree-based profile commands against temporary git repositories
- **test_config.py**, **test_modes_new.py**, **test_output_times*.py**,
  **test_show.py**, **test_utils_mode.py**, **test_params.py**,
  **test_cosmology.py**: the other modes and their helpers
- **test_analyse_*.py**, **test_gravity_*.py**, **test_mpiuse.py**,
  **test_task_plots.py**, **test_log_task_counts.py**,
  **test_step_table.py**: the analysis subcommands
- **test_timer_db.py**: Timer formats, function/operation rule, call graph,
  caching and revision matching (against `data/swift_src` and a temporary
  git repository)
- **test_timer_db_archetypes.py**: one parametrised check per expectation
  type over every case in `data/c_archetypes` (see its README); set
  `SWIFT_SRC=/path/to/swiftsim` to also run invariants on a real checkout
- **test_log_scan.py**, **test_timer_nesting.py**,
  **test_timing_hierarchy.py**, **test_log_timing.py**: the `analyse log`
  pipeline on the log fixtures in `data/logs`
- **test_utilities.py**: Tests utility functions including:
  - ASCII table creation
  - File system operations
  - Path handling

## Fixtures

The test suite uses pytest fixtures defined in `conftest.py`:

- `isolated_profile_file` (autouse): points the profile file at a temporary
  directory so no test touches `~/.swiftsim-utils`
- `temp_dir`: Temporary directory for file operations
- `in_inputs_dir`: runs the test in a directory holding empty input files
  (`a.txt`, `swift.log`, `ics.hdf5`, ...) and a `reports/` directory, for
  parse-only tests (input paths are checked at parse time)
- `write_tiny_ics`: writes a tiny gas-only HDF5 initial conditions file
- `gitsetup` (in `unit/conftest.py`): a fresh bare remote, main clone and
  worktree root, with git isolated from your global configuration
- `logs_dir`: The log fixtures (`data/logs`)
- `swift_src_dir`: The synthetic SWIFT source tree (`data/swift_src`)
- `synthetic_db`: Timer database parsed from `swift_src_dir`

The real-log fixtures are excerpts with job IDs, user names and host names
replaced by placeholders. `data/swift_src/src/generated.c` was generated
from the timers in those logs and their nesting at each log's SWIFT
revision; `special_cases.c` holds hand-written parser edge cases.

## Coverage

Test coverage reports are generated in multiple formats:
- Terminal output with missing lines
- HTML report in `htmlcov/` directory
- XML report for CI/CD integration

The run fails below 85% coverage (`--cov-fail-under` in `pyproject.toml`).

## CI/CD Integration

GitHub Actions (`.github/workflows/ci.yml`) runs ruff, mypy and the test
suite on Ubuntu for Python 3.10 to 3.14, plus macOS for Python 3.12, on
pushes and pull requests to `main`/`develop`. The release workflow runs
ruff and mypy, then the suite against the built wheel before publishing.

## Adding New Tests

When adding new functionality:

1. Create corresponding tests in the appropriate `test_*.py` file
2. Use the existing fixtures where possible
3. Follow the naming convention `test_<functionality>`
4. Add docstrings explaining what the test verifies

Example test structure:
```python
def test_new_functionality(tmp_path):
    """Test description of what this verifies."""
    # Arrange
    input_data = "test input"

    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

## Test Configuration

Test configuration is in `pyproject.toml`:
- Coverage settings
- Output formats
- Failure thresholds

Lint (ruff) and type checks (mypy) run alongside the tests in CI and in the
pre-commit hooks.