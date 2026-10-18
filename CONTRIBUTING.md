# Contributing to swiftsim-cli

## Development setup

```bash
git clone https://github.com/WillJRoper/swiftsim-cli.git
cd swiftsim-cli
pip install -e ".[dev,test]"
pre-commit install
```

The pre-commit hooks run ruff (lint and format) and mypy. The mypy hook uses
the mypy installed in your environment and checks `src` and `tests` with the
`[tool.mypy]` settings in `pyproject.toml`, exactly as CI does.

## Before opening a pull request

```bash
ruff check src tests
ruff format --check src tests
mypy
pytest            # fails below 85% coverage
```

Optionally run the timer parser against a real SWIFT checkout:

```bash
SWIFT_SRC=/path/to/swiftsim pytest tests/unit/test_timer_db_archetypes.py -k real
```

Add tests for new behaviour (see [tests/README.md](tests/README.md)) and a
line under an "Unreleased" heading in [CHANGELOG.md](CHANGELOG.md) for
user-visible changes. Pull requests target `main`.

## Releasing

Versions come from git tags via setuptools_scm; there is no version number
to edit.

1. Move the "Unreleased" changelog entries under the new version and date,
   and merge that to `main`.
2. Tag and push:

   ```bash
   git tag -a v1.0.0 -m "swiftsim-cli 1.0.0"
   git push origin v1.0.0
   ```

3. The `Release` workflow (`.github/workflows/release.yml`) builds the sdist
   and wheel, checks them with twine, runs the test suite against the
   installed wheel, publishes to PyPI and creates a GitHub release with the
   distributions attached.

### One-time setup (maintainer)

PyPI publishing uses
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no API
token is stored in GitHub.

1. **PyPI trusted publisher.** On PyPI, open *Your account → Publishing* (or,
   once the project exists, *swiftsim-cli → Settings → Publishing*) and add
   a GitHub publisher with:
   - Owner: `WillJRoper`
   - Repository: `swiftsim-cli`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   Before the first release this is a "pending publisher", which creates the
   project on the first upload.
2. **GitHub environment.** In the repository, open *Settings → Environments*
   and create an environment named `pypi`. Optionally add required reviewers
   (to approve each release) and restrict it to tags matching `v*`.
3. **Codecov (optional).** Coverage uploads from CI work without a token for
   public repositories. If uploads are rejected or rate limited, add the
   repository's Codecov upload token as a secret named `CODECOV_TOKEN`
   (*Settings → Secrets and variables → Actions*). A failed upload never
   fails CI.
