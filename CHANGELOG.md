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
| `output-times -o/--out FILE` | `output-times --output-list FILE` (`-o/--out` always names an output directory) |
| `output-times -fz`, `-ft`, `-fa`, `-dz`, `-dt`, `-da`, `-dla`, `-sdz`, `-sdt`, `-sda`, `-sdla`, `-fzf`, `-ftf`, `-faf` | the long options only (`--first-snap-z`, `--delta-time`, ...) |
| `output-times -u/--list-units` | `output-times --list-units` |
| `new --path DIR` | `new -o/--out DIR` |
| `new --param ROOT:KEY=VALUE` | `new --set ROOT:KEY=VALUE` (repeatable) |
| `config --debug/-d` (preset) | `config --debug-build` (`--debug` is only the global option) |
| `config -g`, `-e`, `-x` | `config --gravity`, `--eagle`, `--eaglexl` |
| `analyse gravity-error-maps -r N`, `--gridsize N` | `analyse gravity-error-maps --resolution N` |
| `analyse task-plots --task NAME` | `analyse task-plots --tasks NAME [NAME ...]` |
| `analyse task-counts -t NAME ...` | `analyse task-counts --tasks NAME [NAME ...]` |
| `analyse task-plots --rank N` | `analyse task-plots --rank N\|all` (default `all`) |
| `analyse timesteps --time-column 1\|2\|...` (column index) | `analyse timesteps --time-column step\|time\|scale-factor\|redshift` |
| required `-l/--labels` (`timesteps`, `gravity-checks`, `gravity-error-maps`) | optional everywhere, defaulting to the file stems |
| abbreviated long options (e.g. `--res` for `--resolution`) | rejected: spell options out in full |

- Every analysis now writes into `<out>/<analysis>_analysis/` (for example
  `timesteps_analysis/`, `scaling_timers_analysis/`) with `-p/--prefix`
  prepended once to each filename.
- `analyse scaling timers --min-percent-threshold` defaults to 5 (was 10)
  and applies to exclusive timer time.
- Analyses that find no usable data in any input (`gravity-checks`,
  `gravity-error-maps`, `mpiuse`, `task-counts`, `log` with no timer lines,
  `timesteps`) now fail with exit code 1 and stop the chain; they used to
  print a message and exit 0. An input that cannot be used while others can
  is skipped with a warning.
- Input files and directories are checked while parsing, so a missing input
  is a usage error (exit code 2) before any mode runs; invalid numbers
  (e.g. `-j 0`, `--steps 9 2`, `--rank -1`, `--width -5`) are rejected the
  same way.
- Warnings are printed to stderr as `warning: ...` (they used to go to
  stdout).
- Dependency floors: numpy >= 2.1 and swiftsimio >= 10.0 (swiftsimio needs
  numpy 2), with matching floors for astropy (6.1), scipy (1.13), h5py
  (3.11), matplotlib (3.9), pandas (2.2.2), ruamel.yaml (0.17.21) and
  prompt_toolkit (3.0.29).

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
- `--steps START END` for `analyse timesteps` and `analyse task-counts`.
- `--` ends a mode's arguments explicitly
  (`swift-cli config --enable-debug -- make -j 8`).
- Every `analyse`/`utils` subcommand's `--help` starts with a description
  (including what SWIFT must be built or run with).
- `new --force` to replace an existing `params.yml`; `new --set` values are
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
- The licence is GPL-3.0-or-later (it was GPL-3.0-only).
- Start-up is much faster (`swift-cli --help` takes ~0.4 s instead of
  ~2.2 s): mode modules and the heavy libraries (matplotlib, astropy, pandas,
  h5py, swiftsimio, tree-sitter) are imported only when a mode needs them.
- Error messages for parse errors in input files name the file, and
  one-line errors suggest `--debug` for the traceback; reading from a closed
  stdin at a prompt is reported as `error: no input`.

### Fixed

- `output-times`: cosmological `# Time` lists are now written as ages in
  SWIFT's internal time units (they were written in Gyr, which SWIFT
  misread). Times on the command line are still given in Gyr.
- Parameter files that are not a YAML mapping, or are not valid YAML, give a
  one-line error naming the file; duplicate keys (as in SWIFT's own
  `parameter_example.yml`) are a warning and the first value is used.
- Cosmology: a parameter file without a `Cosmology` section no longer
  crashes, and an incomplete one names the missing parameter; ages before z=140 raise a clear error instead of silently
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
