# Changelog

All notable changes to swiftsim-cli are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [1.0.0] - Unreleased

First stable release. It overhauls the pre-1.0 development versions, so it
has **breaking changes** to profiles and to the command line (see below).

### Breaking changes

Profiles:

- Profiles are now git worktrees: each one has its own worktree of your main
  SWIFT clone, its own local branch (`swift-cli/<name>`) following an
  upstream branch, its own configure flags and its own build. Old profile
  files are detected and converted with `swift-cli profile --migrate` (a
  backup is kept as `profiles.yaml.bak`; old timer databases are deleted).
- `switch NAME` now switches the current **profile** (an alias of
  `profile --switch NAME`). It used to switch the git branch of the SWIFT
  checkout.
- `profile --clear` and `profile --override` were removed. Use
  `profile --remove NAME` and the global `--profile NAME` option instead.

Command-line flags (old → new):

| Old | New |
| --- | --- |
| `-p FILE` (parameter file) | `-P/--params FILE` (`-p` is now `--prefix`) |
| `--output-path DIR` | `-o/--out DIR` |
| `--show-plot` | `--show` |
| `--inic FILE` | `--ics FILE` |
| `swift-cli scaling ...` (top-level mode) | `swift-cli analyse scaling timers\|runtime ...` |
| `analyse gravity-error-map` | `analyse gravity-error-maps` |
| `analyse log` output directory `runtime_analysis/` | `log_analysis/` |
| `analyse timesteps --nthreads` | `analyse timesteps --threads-per-run` |
| `config --show/-s` (list configure options) | `config --list-options` |
| `profile -S` | `profile --show [NAME]` (long form only) |
| global `-v/--verbose` | removed (use `--debug` for tracebacks) |

- Every analysis now writes into `<out>/<analysis>_analysis/` (for example
  `timesteps_analysis/`, `scaling_timers_analysis/`) with `-p/--prefix`
  prepended once to each filename.
- `analyse scaling timers --min-percent-threshold` defaults to 5 (was 10)
  and applies to exclusive timer time.
- Python 3.10 or newer is required.

### Added

- Worktree-based profile commands: `profile --new NAME [--upstream]
  [--configure-flags] [--build]`, `--update [NAME]` (rebase with autostash,
  reconfigure only when needed, build), `--update-all` with a summary,
  `--edit`, `--remove [--force]`, `--adopt PATH --name NAME`, `--migrate`.
- Global options `--profile NAME` (run against another profile without
  switching), `--debug` (or `SWIFT_CLI_DEBUG=1`) and `--version`.
- Clean error handling: user errors print one `error: ...` line and exit 1;
  failed external commands exit with their own code.
- `config` validates every `--<option>` flag against `./configure --help`
  of the profile's worktree, with close-match suggestions.
- `-j/--nr-threads` for `make`, `update` and `profile`.
- `new --force` to replace an existing `params.yml`; `new --param` values are
  parsed as numbers where possible.
- `analyse log`: `--rank N|all`, `--swift-src DIR`,
  `--no-match-revision`, `--regenerate-db`.
- One inclusive `--steps START END` option shared by `analyse log`,
  `mpiuse`, `scaling timers` and `scaling runtime`.
- A release workflow publishing to PyPI with Trusted Publishing.

### Changed

- **Log timing analysis rework** (`analyse log`): timers are read from the
  SWIFT source with tree-sitter, at the revision printed in the log by
  default (extracted with `git archive`; the checkout is not touched), and
  cached per profile. Each step's timers are nested into the call tree that
  actually ran, so nothing is counted twice; all percentages are of the
  step-table wall-clock time with the untimed remainder shown; timers before
  step row N belong to step N and those before row 0 to initialisation;
  restarts and partial logs are handled; MPI ranks are never summed. Output
  is a short console summary plus `log_analysis/analysis_tables.txt` and
  charts.
- Step tables are located by their header, so older and newer SWIFT table
  layouts (with or without sink/BH columns, with MPI rank prefixes) parse.
- Shared helpers for plotting, tables and I/O replace duplicated code.
- Strict type checking (mypy with strict optional) across src and tests.

### Fixed

- Cosmology: a parameter file without a `Cosmology` section no longer
  crashes; ages before z=140 raise a clear error instead of silently
  becoming z=0; time/redshift conversions consistently use Gyr.
- `output-times`: a redshift start with a time interval uses the parameter
  file's cosmology; `a_end` from the parameter file is the default end.
- `new`: ICs with a `Units` group or without dark matter work; derived values
  are numbers and paths are absolute; scale-factor `Time` headers set
  `Cosmology:a_begin`; `params.yml` is written atomically.
- `gravity-checks`/`gravity-error-maps`: exact and SWIFT files are matched by
  particle ID, step numbers wider than four digits are found, several
  multipole orders give a warning, hexbin counts are autoscaled and
  distances are measured from the data centre; plot styling no longer leaks
  into other plots.
- `task-plots`: per-rank CPU frequencies, no fake task-0 records, unknown
  task/subtype IDs and threadpool dumps on the right rank.
- `scaling`: single-rank logs, skipped series no longer shift labels, and
  decreasing x columns are not interpolated.
- Timer parsing: `#if 0` blocks, pointer arguments, literal units
  (`ms`, `s`, `seconds`, ...) and same-named static functions in different
  files.

[1.0.0]: https://github.com/WillJRoper/swiftsim-cli/releases/tag/v1.0.0
