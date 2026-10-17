# Test Suite

This directory contains the test suite for swiftsim-cli.

## Structure

```
tests/
├── conftest.py           # Pytest configuration and shared fixtures
├── unit/                 # Unit tests
│   ├── test_cli.py       # CLI functionality tests  
│   ├── test_src_parser.py        # Source parser tests
│   └── test_utilities.py # Utility function tests
└── README.md            # This file
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

### Using Makefile

The project includes a Makefile with common test commands:

```bash
make test          # Run tests without coverage
make test-cov      # Run tests with coverage
make test-fast     # Run tests quickly (no coverage)
```

## Test Categories

### Unit Tests

- **test_cli.py**: Tests the main CLI entry point and argument parsing
- **test_src_parser.py**: Tests source code parsing functionality including:
  - Timer definition extraction
  - Log pattern compilation  
  - Timer instance scanning
  - Tree-sitter integration (when available)
- **test_utilities.py**: Tests utility functions including:
  - ASCII table creation
  - File system operations
  - Path handling

## Fixtures

The test suite uses pytest fixtures defined in `conftest.py`:

- `temp_dir`: Temporary directory for file operations
- `sample_timer_db`: Mock timer database for testing
- `sample_nesting_db`: Mock nesting database for testing  
- `sample_timer_instances`: Mock timer instances for testing
- `sample_log_content`: Sample SWIFT log content for testing
- `mock_swift_profile`: Mock SWIFT profile for testing

## Coverage

Test coverage reports are generated in multiple formats:
- Terminal output with missing lines
- HTML report in `htmlcov/` directory
- XML report for CI/CD integration

The current coverage target is 50% (configured in `pyproject.toml`).

## CI/CD Integration

Tests are automatically run in GitHub Actions on:
- Push to main/develop branches
- Pull requests
- Multiple Python versions (3.10, 3.11, 3.12)
- Multiple operating systems (Ubuntu, macOS)

## Adding New Tests

When adding new functionality:

1. Create corresponding tests in the appropriate `test_*.py` file
2. Use the existing fixtures where possible
3. Follow the naming convention `test_<functionality>`
4. Add docstrings explaining what the test verifies
5. Use appropriate test markers if needed (slow, integration, etc.)

Example test structure:
```python
def test_new_functionality(self, fixture_name):
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
- Test markers
- Output formats
- Failure thresholds

Security scanning and code quality checks are also integrated into the test pipeline.