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
| `analyse timesteps --nthreads` | `analyse timesteps --cores-per-run` |
| `analyse timesteps --time-column 1\|2\|...` (column index, default 2) | `analyse timesteps --time-column auto\|step\|time\|scale-factor\|redshift` (default `auto`) |
| `analyse task-plots TASK_FILE THREADPOOL_FILE` | `analyse task-plots TASK_FILE [--threadpool FILE]` |
| `config --show/-s` (list configure options) | `config --list-options` |
| `profile -S` | `profile --show [NAME]` (long form only) |
| global `-v/--verbose` | removed (use `--debug` for tracebacks) |
| `output-times -o/--out FILE` | `output-times --output-list FILE` (`-o/--out` always names an output directory) |
| `output-times -fz`, `-ft`, `-fa`, `-dz`, `-dt`, `-da`, `-dla`, `-sdz`, `-sdt`, `-sda`, `-sdla`, `-fzf`, `-ftf`, `-faf` | the long options only (`--first-snap-z`, `--delta-time`, ...) |
| `output-times -u/--list-units` | `output-times --list-units` |
| `new --path DIR` | `new -o/--out DIR` |
| `new --param ROOT:KEY=VALUE` | `new --set SECTION:KEY=VALUE` (repeatable) |
| `config --debug/-d` (preset) | `config --debug-build` (`--debug` is only the global option) |
| `config -g`, `-e`, `-x` | `config --gravity`, `--eagle`, `--eaglexl` |
| `analyse gravity-error-maps -r N`, `--gridsize N` | `analyse gravity-error-maps --resolution N` |
| `analyse task-plots --task NAME` | `analyse task-plots --tasks NAME [NAME ...]` |
| `analyse task-counts -t NAME ...` | `analyse task-counts --tasks NAME [NAME ...]` |
| `analyse task-plots --rank N` | `analyse task-plots --rank N\|all` (default `all`) |
| `show` (profile only), `show --all` (profile and cosmology) | `show [NAME]` shows both; `--cosmo` only the cosmology; `--all` is a no-op |
| required `-l/--labels` (`timesteps`, `gravity-checks`, `gravity-error-maps`) | optional everywhere, defaulting to the file stems |
| abbreviated long options (e.g. `--res` for `--resolution`) | rejected: spell options out in full |

Behaviour:

- Every analysis now writes into `<out>/<analysis>_analysis/` (for example
  `timesteps_analysis/`, `scaling_timers_analysis/`) with `-p/--prefix`
  prepended once to each filename. The timesteps plot is now
  `timesteps_analysis.png`; `task_counts_cumulative.png` is no longer
  written; `analyse log` writes per-step coverage to
  `per_step_coverage.csv`.
- `new` starts from a bundled template chosen from the ICs (or `-P` /
  the profile's `parameter_file`) instead of SWIFT's
  `examples/parameter_example.yml`.
- `output-times` no longer overwrites an existing list without `--force`,
  and contradictory, zero or negative inputs are usage errors.
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
- A failed `configure` or `make` exits with that command's own exit code.
- Warnings are printed to stderr as `warning: ...` (they used to go to
  stdout).
- `Cosmology:Omega_lambda` is required in parameter files (as in SWIFT).
- `profiles.yaml` is written with normal permissions (umask), not 0600.
- Dependency floors: numpy >= 2.1 and swiftsimio >= 10.0 (swiftsimio needs
  numpy 2), with matching floors for astropy (6.1), scipy (1.13), h5py
  (3.11), matplotlib (3.9), pandas (2.2.2), ruamel.yaml (0.17.21) and
  prompt_toolkit (3.0.29).

### Added

General:

- Global options `--profile NAME` (run against another profile without
  switching; checked before any mode runs), `--debug` (or
  `SWIFT_CLI_DEBUG=1`) and `--version`.
- `--` ends a mode's arguments explicitly
  (`swift-cli config --enable-debug -- make -j 8`).
- Every `analyse`/`utils` subcommand's `--help` starts with a description
  (including what SWIFT must be built or run with).
- A release workflow publishing to PyPI with Trusted Publishing.

Profiles and builds:

- Worktree-based profile commands: `profile --new NAME [--upstream]
  [--configure-flags] [--build] [--reuse-branch]`, `--update [NAME]`
  (rebase with autostash, reconfigure only when needed, build),
  `--update-all` with a summary, `--edit`, `--remove [--force]`,
  `--adopt DIR --name NAME`, `--migrate [--worktree-root DIR]`.
- Non-interactive use: `profile --init --repo DIR` (with `--worktree-root`,
  `--name`, `--upstream`, `--configure-flags`, `--set KEY=VALUE`),
  `profile --edit NAME --set KEY=VALUE`, and `-y/--yes` for the
  confirmations of `--remove`, `--migrate` and `--init`.
- `update --no-build`; a chained `update ... config/make` builds once.
- `show [NAME]` shows any profile.
- `-j/--nr-threads` for `make`, `update` and `profile`.
- `config` validates every `--<option>` flag against `./configure --help`
  of the profile's worktree, with close-match suggestions.

Run setup:

- `new`: runnable parameter templates `cosmo-dmo`, `cosmo-hydro`,
  `isolated-gravity` and `isolated-hydro`, chosen from the ICs or with
  `--template`; `-P/--params` starts from your own file.
- `new` checks `params.yml` against SWIFT's parameter parser limits before
  writing, prints the `swift` command line (with `--stars` when the ICs
  contain stars), and `--check` runs `swift --dry-run` on the result.
- `new --force` replaces an existing `params.yml`.
- `output-times --no-cosmology` and `--force`.

Analysis:

- `analyse log`: `--rank N|all`, `--swift-src DIR`,
  `--no-match-revision`, `--regenerate-db`; a "BETWEEN STEPS" table for
  timers printed outside any step; recovery of interleaved or damaged log
  lines with a warning.
- One inclusive `--steps START END` option shared by `analyse log`,
  `timesteps`, `task-counts`, `mpiuse`, `scaling timers` and
  `scaling runtime`.
- `analyse scaling timers`: `--swift-src`, `--no-match-revision`,
  `--regenerate-db` and a "SCALING INPUTS" table.
- `analyse task-plots`: `--threadpool FILE` (optional), `--swift-src DIR`
  for task and cell names, and cell filters by name.
- `analyse timesteps`: `--time-column auto`, `--cores-per-run`.
- `analyse gravity-checks --all-orders`.
- `analyse gravity-error-maps --vmin`, `--vmax` and `--box-size`.

### Changed

Profiles and builds:

- Update fails up front if the branch has no upstream, reports what was
  pulled, and refuses to build when re-applying stashed changes conflicts;
  no build runs in a worktree that is mid-rebase or has conflicts.
- `profile --edit` with a new upstream moves a branch without commits of
  its own and otherwise keeps your commits, printing the `git rebase
  --onto` command.
- `profile --remove` deletes only worktrees swift-cli created; adopted
  worktrees, the main clone and shared worktrees are only unregistered.
- Prompts validate each answer; piped answers are read one per line.
- `config` rejects arguments that are neither `--opt[=value]` nor
  `VAR=value`; `--eagle` and `--eaglexl` are mutually exclusive.

Run setup:

- `new` sets zoom softenings from the high-resolution particle mass with
  `softening_ratio_background` (and a `ZoomRegion` section where SWIFT
  supports it), non-cosmological maximal physical softenings from the
  mean separation, and Gadget h-factor clean-up for cosmological ICs
  without a `Units` group.
- `new --set` values are written as numbers, `[lists]` or verbatim
  strings; new keys are allowed with a warning; empty values are rejected.
- `utils softenings`: `-P` is optional and the calculation is exactly
  `new`'s.
- Cosmology conversions follow SWIFT's cosmology (radiation, neutrinos,
  w0/wa) and time → redshift is vectorised.

Analysis:

- **Log timing rework** (`analyse log`): timers are read from the SWIFT
  source with tree-sitter, at the revision printed in the log by default
  (extracted with `git archive`; the checkout is not touched), and cached
  per profile. Each step's timers are nested into the call tree that
  actually ran, so nothing is counted twice; all percentages are of the
  step-table wall-clock time with the untimed remainder shown; timers
  before step row N belong to step N and those before row 0 to
  initialisation; restarts and partial logs are handled; MPI ranks are
  never summed. Output is a short console summary plus
  `log_analysis/analysis_tables.txt` (coverage outliers only) and charts.
- Step tables are located by their header, so older and newer SWIFT table
  layouts (with or without sink/BH columns, with MPI rank prefixes) parse.
- `analyse scaling` is a package; `--rank-source all` averages over the
  ranks that printed timers, and `scaling runtime` leaves out step 0 and
  restart rows like the other analyses.
- `analyse task-plots` reads every SWIFT task dump layout (11/13/17/19/21
  columns), takes task and cell names from the SWIFT source with built-in
  fallbacks, uses one step window for all ranks and stable colours, and is
  about 15x faster with a third of the memory on large dumps.
- `analyse task-counts` uses SWIFT's `System total` and warns when the log
  only has rank 0's counts.
- `analyse mpiuse` reports bytes sent and received separately.
- `analyse timesteps` handles restarts, gaps, malformed rows and a missing
  dead-time column.
- `analyse gravity-checks` prints one percentile table per run and plots
  autoscaled histograms with a summary panel; `gravity-error-maps` shows the
  median error per hexagon with a percentile colour scale, scatter points
  for small samples, a median centre and an error histogram.

General:

- Shared helpers for plotting, tables and I/O replace duplicated code.
- Strict type checking (mypy with strict optional) across src and tests.
- The licence is GPL-3.0-or-later (it was GPL-3.0-only).
- Start-up is much faster (`swift-cli --help` takes ~0.4 s instead of
  ~2.2 s): mode modules and the heavy libraries are imported only when a
  mode needs them.
- Errors print one `error: ...` line (suggesting `--debug` for the
  traceback); parse errors in input files name the file; reading from a
  closed stdin at a prompt is reported as `error: no input`.

### Fixed

Profiles and builds:

- Update building after an autostash conflict; `--remove` deleting adopted
  worktrees or the main clone; re-running `--init` re-pointing the clone of
  existing profiles; migration losing data when every profile failed;
  invalid profile names (`..`, trailing `.` or `.lock`, case clashes).
- `show` crashing on duplicate keys in the parameter file.

Run setup:

- `output-times`: cosmological `# Time` lists are written as ages in
  SWIFT's internal time units (they were written in Gyr, which SWIFT
  misread); no outputs past the final one, no duplicate snapshot/snipshot
  entries, no single-entry lists, no crash on `--delta-* 0`, no precision
  loss for small internal-unit times; a redshift start with a time
  interval uses the parameter file's cosmology.
- `new`: crashes on SWIFT's `parameter_example.yml` and files SWIFT rejects
  (wrapped long lines, too many parameters or sections, `FLT_MAX`);
  non-cosmological runs no longer keep a `Cosmology` section; ICs with a
  `Units` group or without dark matter work; scale-factor `Time` headers
  set `Cosmology:a_begin`; `params.yml` is written atomically.
- `utils softenings`: ICs without `Units`, scalar or 1-element header
  attributes and `NumPart_Total_HighWord`.
- Parameter files that are not a YAML mapping, or are not valid YAML, give a
  one-line error naming the file; duplicate keys are a warning and the
  first value is used.
- Cosmology: a missing `Cosmology` section no longer crashes, an incomplete
  one names the missing parameter, and ages before z=140 raise a clear
  error instead of silently becoming z=0.

Analysis:

- `analyse log`: coverage above 100% on task-dump steps; timers missing
  from the source revision double counted; `--swift-src` without C files
  caching an empty database; `--steps` outside the log silently empty.
- `analyse mpiuse`: volumes were counted twice (send and receive).
- `analyse timesteps`: restarted runs double counted repeated steps; bogus
  speedups for non-cosmological runs.
- `gravity-checks`/`gravity-error-maps`: files matched by particle ID
  (int64), step numbers wider than four digits, `step1000` no longer
  matching `step10000`, non-finite values dropped, plot styling no longer
  leaking into other plots.
- `task-plots`: per-rank CPU frequencies, no fake task-0 records,
  threadpool dumps on the right rank.
- `scaling`: single-rank logs, skipped series no longer shift labels, and
  decreasing x columns are not interpolated.
- Timer parsing: `#if 0` blocks, pointer arguments, literal units
  (`ms`, `s`, `seconds`, ...), same-named static functions in different
  files, and `main()` timers in `examples/main.c`.

[1.0.0]: https://github.com/WillJRoper/swiftsim-cli/releases/tag/v1.0.0
