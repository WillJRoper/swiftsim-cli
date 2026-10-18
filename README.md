```
    ______       _________________        _______   ____
   / ___/ |     / /  _/ ___/_  __/ ____  / ___/ /  /  _/
   \__ \| | /| / // // /_   / /   /___/ / /  / /   / /
  ___/ /| |/ |/ // // __/  / /         / /__/ /__ / /
 /____/ |__/|__/___/_/    /_/         /____/____/___/

```

A command-line tool for building, running and analysing
[SWIFT](https://gitlab.cosma.dur.ac.uk/swift/swiftsim) simulations.

`swift-cli` manages any number of independent SWIFT builds ("profiles"),
configures and compiles them, sets up run directories, generates output-time
lists and analyses the logs and diagnostics SWIFT writes.

## Features

- **Multi-mode execution**: chain modes in one command, e.g.
  `swift-cli config --debug-build make -j 32`.
- **Profiles**: every profile is its own git worktree, branch, set of
  configure flags and build, so switching never needs a rebuild.
- **Simulation setup**: create a run directory with a ready-to-run
  `params.yml` derived from your initial conditions.
- **Output times**: generate snapshot/snipshot output lists in redshift, time
  or scale factor.
- **Analysis**: per-step timer call trees from SWIFT logs, timestep plots,
  gravity accuracy checks, task plots, MPI communication and scaling studies.

## Installation

Requirements: Python >= 3.10 and git. Running SWIFT itself needs SWIFT's own
dependencies (see [SWIFT's documentation](https://swift.strw.leidenuniv.nl/docs/index.html)).

From PyPI:

```bash
pip install swiftsim-cli
```

From source:

```bash
git clone https://github.com/WillJRoper/swiftsim-cli.git
cd swiftsim-cli
pip install .            # or: pip install -e ".[dev,test]" for development
```

Check the installation with `swift-cli --version`.

## Quick start

```bash
# 1. Point swift-cli at your SWIFT clone and create the first profile
#    (interactive: clone path, worktree root, upstream, configure flags,
#    data directory, softening constants)
swift-cli profile --init

# 2. Add more profiles, each with its own branch, flags and build
swift-cli profile --new zoom --upstream origin/zoom \
    --configure-flags="--enable-debug --with-hydro=sphenix" --build -j 16

# 3. Pull, reconfigure if needed and rebuild the current profile
swift-cli profile --update -j 16        # same as: swift-cli update -j 16

# 4. Compile after editing the source
swift-cli make -j 16

# 5. Create a run directory with a params.yml for your ICs
swift-cli new -o runs/test --ics ics.hdf5 --set Snapshots:basename=snap

# 6. Generate an output list from the run's cosmology
swift-cli output-times -P runs/test/params.yml --first-snap-z 10 --delta-z 0.5 \
    --output-list runs/test/output_list.txt

# 7. After running SWIFT with -v 1, analyse where the time went
swift-cli analyse log runs/test/swift.log -o runs/test
```

## Profiles

A profile is a fully independent SWIFT build. Every profile has:

- its own **git worktree** of your main SWIFT clone (`<worktree_root>/<name>`),
- its own **local branch** (`swift-cli/<name>`) tracking an **upstream** branch (e.g. `origin/master`),
- its own **configure flags** and build products.

Because builds never share a directory, switching profiles is instant and can never mix up builds. All modes (`config`, `make`, `update`, `new`, ...) act on the *current* profile; use the global `--profile NAME` option to target another profile without switching.

```bash
swift-cli profile --new zoom --upstream origin/zoom --configure-flags="--enable-debug" --build -j 16
swift-cli profile --list              # all profiles (* marks the current one)
swift-cli profile --show [NAME]       # details of one profile
swift-cli profile --switch zoom       # or: swift-cli switch zoom
swift-cli profile --edit zoom         # change upstream, flags, paths, constants
swift-cli profile --update [NAME] -j 16   # or: swift-cli update -j 16
swift-cli profile --update-all -j 16  # update every profile, then print a summary
swift-cli profile --remove zoom       # removes the worktree (and optionally the branch)
swift-cli profile --remove zoom --force  # ...even if the worktree has untracked files
swift-cli --profile zoom make -j 16   # build another profile without switching
```

Updating a profile runs `git pull --rebase --autostash` in its worktree (your local commits are replayed on top of the upstream, and any uncommitted changes are stashed and re-applied afterwards). If the rebase hits conflicts nothing is built and you're told how to resolve them. Otherwise `./autogen.sh` is run if `./configure` is missing, `./configure` is (re)run only if the profile has never been configured or its configure flags changed, and then `make` is run.

`swift-cli config` sets the current profile's configure flags (from presets such as `--debug-build` and/or any `--<option>` flags) and reconfigures. Flags are validated against the options listed by `./configure --help` in that profile's worktree, with suggestions for typos.

Already have worktrees made by hand? Register them with:

```bash
swift-cli profile --adopt ../swiftsim-my-feature --name my-feature
```

Profiles are stored in `~/.swiftsim-utils/profiles.yaml` and per-profile caches (e.g. the timer database used by `analyse log`) in `~/.swiftsim-utils/cache/<name>/`. If you have a profile file from an older version of `swift-cli`, convert it (after reviewing the plan it prints) with:

```bash
swift-cli profile --migrate
```

The migration also deletes the old timer databases (`timers.yaml` and
`timer_nesting.yaml` in `~/.swiftsim-utils/` and its `cache/<name>/`
directories), which are no longer used; they are listed in the plan first.

## Global options

Global options must come before the first mode:

- `--profile NAME`: run every mode against profile `NAME` instead of the
  current one (e.g. `swift-cli --profile zoom make -j 16`).
- `--debug`: show full tracebacks instead of a one-line `error: ...` (or set
  `SWIFT_CLI_DEBUG=1`).
- `--version`, `-h/--help`.

`swift-cli --help` prints:

```text
swift-cli: Utilities for SWIFT development workflows

Usage:
  swift-cli [global options] <mode> [mode args] [<mode> [mode args]] ...

Global options (must come before the first mode):
  -h, --help      Show this help and exit.
  --version       Show the version and exit.
  --profile NAME  Run the modes against profile NAME.
  --debug         Show full tracebacks for errors (or set
                  SWIFT_CLI_DEBUG=1).

Available modes:
  profile         Manage SWIFT profiles (one worktree, branch and build each).
  config          Set the current profile's configure flags and reconfigure.
  output-times    Generate a list of snapshot (and snipshot) output times.
  update          Pull and rebuild the current profile.
  switch          Make another profile the current one.
  make            Compile the current profile.
  new             Create a new simulation run directory.
  analyse         Analyse SWIFT logs and diagnostic outputs.
  show            Show the current profile (and optionally its cosmology).
  utils           Run small helper calculations (e.g. softenings).

Modes run in order and the chain stops at the first error. A mode name
starts the next mode unless the current mode expects a value there; use
'--' to end a mode's arguments explicitly.

Examples:
  swift-cli config --enable-debug
  swift-cli make -j 8
  swift-cli config --enable-debug make -j 32
  swift-cli config --enable-debug -- make -j 32
  swift-cli --profile zoom make -j 16

For mode-specific help: swift-cli <mode> --help
```

Every mode has its own help: `swift-cli <mode> --help` (and
`swift-cli analyse <subcommand> --help`).

### Chaining modes

```bash
# Configure with the debug preset, then compile
swift-cli config --debug-build make -j 16

# Compile and create a new simulation in the test_run directory
swift-cli make -j 32 new -o test_run --ics initial_conditions.hdf5

# Analyse a log, then show the profile
swift-cli analyse log run/swift.log -o run show
```

Modes run in order and the chain stops at the first error. A word that
names a mode starts the next mode, unless the current mode expects it as a
value: `new -o make ...` creates a run directory called `make`, and
`switch show` switches to a profile called `show`. A `--` always ends the
current mode's arguments, so the next word must name a mode
(`config --enable-debug -- make -j 8`).

Every mode parses its arguments before the first one runs, so a typo in
the last mode fails before anything is built. Input files and directories
are checked at that point too, except the parameter file (`-P/--params`),
which is checked when its mode runs because an earlier `new` may create
it. Long options must be spelled out in full (no abbreviations).

### Errors

Errors are reported as a single `error: ...` line on stderr; use `--debug`
or `SWIFT_CLI_DEBUG=1` for the traceback. Exit codes:

- `0`: success (also for `--help` and `--version`).
- `1`: an error while running a mode (e.g. an unreadable input or an
  analysis that found no usable data), or no mode given (the help is
  printed).
- `2`: a usage error (unknown option, invalid value, missing input file).
- A failed external command (`configure`, `make`, `git`) exits with its own
  code; an interrupt exits with 130.

Warnings (e.g. one of several inputs that could not be used) go to stderr
as `warning: ...` lines and do not stop the chain.

## Modes

| Mode | What it does |
| --- | --- |
| `profile` | Create, list, show, switch, edit, update, remove, adopt and migrate profiles |
| `config` | Set the current profile's configure flags and reconfigure |
| `make` | Compile the current profile |
| `update` | Pull (rebase onto upstream) and rebuild the current profile |
| `switch` | Make another profile the current one |
| `show` | Show the current profile and/or the parameter file's cosmology |
| `new` | Create a run directory with a `params.yml` |
| `output-times` | Generate an output-time list |
| `analyse` | Analyse SWIFT logs and diagnostics (subcommands below) |
| `utils` | Small helper calculations (`softenings`) |

### `config`

```bash
swift-cli config --list-options                    # ./configure --help
swift-cli config --debug-build                     # preset flags
swift-cli config --eagle --with-cooling=COLIBRE    # preset + any --<option>
swift-cli config                                   # reconfigure with stored flags
```

Presets: `--debug-build`, `--gravity`, `--eagle`, `--eaglexl`. Any other
`--<option>` flag is validated against `./configure --help` in the profile's
worktree (with suggestions for typos) and passed on. The resulting flags
replace the profile's stored configure flags.

### `make`, `update`, `switch`, `show`

```bash
swift-cli make -j 16             # -j/--nr-threads, --clean runs make clean first
swift-cli update -j 16           # pull, reconfigure if needed, build
swift-cli switch zoom            # same as profile --switch zoom
swift-cli show --all             # profile and its parameter file's cosmology
```

### `new`

```bash
swift-cli new -o runs/L100 --ics ics/L100.hdf5 \
    --set Cosmology:a_end=1.0 --set Snapshots:basename=L100
```

Writes `runs/L100/params.yml` from the profile's
`examples/parameter_example.yml`, with absolute paths to the ICs and to
`restart/` and `snapshots/` subdirectories, the units, softenings and start
time (`a_begin` or `time_begin`) derived from the ICs, and your
`--set ROOT:KEY=VALUE` overrides. An existing `params.yml` is only replaced with `--force`.

### `output-times`

```bash
# Snapshots every 0.5 in redshift from z=10 to a_end in the params file
swift-cli output-times -P params.yml --first-snap-z 10 --delta-z 0.5

# Snapshots every 0.05 in scale factor plus snipshots every 0.01
swift-cli output-times -P params.yml --first-snap-scale-factor 0.1 \
    --delta-scale-factor 0.05 --snipshot-delta-scale-factor 0.01

# Non-cosmological run: times in internal units
swift-cli output-times --first-snap-time 0 --delta-time 0.1 --final-snap-time 2
```

The list is written to `--output-list FILE` (default: `output_list.txt` in
the current directory). The interval (`--delta-*`) chooses the units of the
list; `--list-units` overrides them. With snipshots the list gets a `Select Output` column.
With a cosmology, `--*-time` values are given as ages of the Universe in
Gyr, and the list is written as ages in SWIFT's *internal* time units
(UnitLength_in_cgs / UnitVelocity_in_cgs from the parameter file's
`InternalUnitSystem`), which is what SWIFT expects for a `# Time` list in a
cosmological run.

### `utils`

```bash
swift-cli utils softenings --ics ics.hdf5 -P params.yml
```

Prints the mean interparticle separation and the comoving/maximal physical
softening of each particle species (coefficient and pivot redshift default
to the profile's values).

## Analysis (`analyse`)

```text
usage: swift-cli analyse [-h] ANALYSIS ...

Analyse SWIFT logs and diagnostic outputs.

positional arguments:
  ANALYSIS            Type of analysis to perform.
    timesteps         Plot the timestep files of one or more runs.
    gravity-checks    Compare SWIFT gravity forces with exact gravity checks.
    gravity-error-maps
                      Create hexbin error maps for gravity check files.
    log               Analyse the timers in a SWIFT log file (run SWIFT with
                      -v 1 for rank 0 timers, -v 2 for every rank).
    task-counts       Plot the engine_print_task_counts output of SWIFT log
                      files per step and cumulatively.
    mpiuse            Analyse SWIFT mpiuse_report-rank*-step*.dat files and
                      compare communication volume across inputs.
    task-plots        Plot SWIFT task-debug and threadpool timings on a shared
                      wall-clock axis.
    scaling           Analyse how SWIFT scales across runs with different MPI
                      rank counts.

options:
  -h, --help          show this help message and exit

Each analysis writes its files into <out>/<analysis>_analysis/ (-o/--out,
default: the current directory), with -p/--prefix prepended to each filename.
```

### Outputs

All `analyse` subcommands share the same output options:

- `-o/--out DIR`: outputs go into `DIR/<analysis>_analysis/` (e.g.
  `timesteps_analysis/`, `log_analysis/`, `scaling_timers_analysis/`); `DIR`
  defaults to the current directory.
- `-p/--prefix PREFIX`: prepended once to every output filename
  (`PREFIX_<name>`).
- `--show`: also show the plots interactively.

Analyses that compare several inputs take optional `-l/--labels LABEL ...`
(one per input; default: the file stems, with `_1`, `_2`, ... added to
stems shared by several inputs). Put the inputs before `-l`. Step-based
analyses take `--steps START END`, and those that can pick an MPI rank take
`--rank N|all`.

An analysis that finds no usable data in any input fails (exit code 1);
inputs that cannot be used while others can are skipped with a warning.

Parameter files are always given with `-P/--params` and initial conditions
with `--ics`.

### Subcommands

```bash
# Timestep files of two runs, CPU time from the threads each run used
swift-cli analyse timesteps run1/timesteps.txt run2/timesteps.txt \
    -l run1 run2 --threads-per-run 64 128 --time-column redshift

# Gravity accuracy: SWIFT forces vs the exact gravity checks
swift-cli analyse gravity-checks gravity_checks_exact_step0000.dat
swift-cli analyse gravity-error-maps gravity_checks_exact_step0000.dat \
    --resolution 50

# Task counts printed by engine_print_task_counts
swift-cli analyse task-counts swift.log --tasks self pair --steps 10 100

# Task-debug (-y) and threadpool (-Y) timings on a shared axis
swift-cli analyse task-plots thread_info-step64.dat threadpool_info-step64.dat \
    --tasks pair/force self/force --activity-plot

# Scaling across runs with different MPI rank counts
swift-cli analyse scaling timers logs/*.log --steps 10 100 -p zoom
swift-cli analyse scaling runtime logs/*.log
```

`analyse scaling timers` plots the timers whose exclusive time (excluding
nested timers) is at least `--min-percent-threshold` percent of the step
time in at least one log (default: 5).

### Log timing analysis (`analyse log`)

`analyse log` shows where the time of a run goes, from the timers SWIFT
prints with `-v 1` (rank 0) or `-v 2` (every rank):

```bash
swift-cli analyse log swift.log                     # rank 0, all steps
swift-cli analyse log swift.log --steps 100 200 --top-n 10
swift-cli analyse log swift.log --rank all          # per rank + imbalance
swift-cli analyse log swift.log --hierarchy-functions engine_rebuild
```

- Timers are read from the SWIFT source with tree-sitter. By default the
  source is taken at the revision printed in the log (extracted with
  `git archive` from the profile's repository into the profile cache; the
  checkout is not touched); `--no-match-revision` uses the checkout as it
  is, and `--swift-src DIR` any other source tree. The parsed database is
  cached and rebuilt only when the source changes (`--regenerate-db`
  forces a rebuild). The share of timer lines matched to the source is
  always reported.
- The timers of each step are nested into the call tree that actually ran
  (a function called from several places is attributed to the caller that
  contained each call), so nothing is counted twice. All percentages are of
  the step-table wall-clock time: the top-level breakdown, including the
  untimed remainder, sums to 100%. Exclusive (self) times add up; inclusive
  times of nested timers do not.
- The table row SWIFT prints for step N comes at the start of step N+1, so
  the timers before row N are step N's; those before row 0 are
  initialisation.
- MPI ranks are never summed: `--rank N` picks one rank (default 0) and
  `--rank all` analyses each rank and compares them.

The console shows a short summary; `log_analysis/analysis_tables.txt` has
the call tree, the timers by exclusive time, the per-caller breakdown of the
`--hierarchy-functions`, per-step coverage, initialisation, task categories
and task counts, alongside the charts.

Interpreting the numbers:

- **Inclusive** time is the time spent inside a timer including everything
  nested in it; **exclusive** (self) time excludes nested timers. Exclusive
  times add up to the timed total; inclusive times of nested timers do not.
- **Denominator**: every percentage is a share of the summed step-table
  wall-clock time of the analysed steps (`--steps`), not of the sum of the
  timers. The difference is reported as the untimed remainder.
- **Ranks**: with `--rank N` the numbers are that rank's alone; `--rank all`
  repeats the analysis per rank and adds an imbalance table.
- **Revision matching**: the timer list comes from the source at the
  revision in the log's `Revision:` header line. If the log's revision is
  not in the profile's repository (e.g. an unpushed commit), use
  `--no-match-revision` or `--swift-src`; the reported match rate shows how
  many timer lines were recognised.

### MPI use analysis (`analyse mpiuse`)

The `analyse mpiuse` subcommand compares MPI communication recorded by
SWIFT when built with `--enable-mpiuse-reports`.

Each input should be a directory, glob, or single
`mpiuse_report-rank*-step*.dat` file set representing one run. Multiple
inputs are compared side by side.

```bash
# Compare MPI communication between two runs
swift-cli analyse mpiuse master_reports fewer_gpart_comms_reports \
  --labels master fewer_gpart_comms \
  --prefix gravity_cmp \
  --format both \
  --per-rank
```

This produces:

- Per-input ASCII summary tables
- A cross-input comparison table with byte deltas
- Per-step and per-subtype plots in `mpiuse_analysis/` (under `--out`)
- Optional per-step CSV output with `--format csv` or `--format both`

Useful filters include `--subtypes gpart grav_counts` and
`--steps START END`.

## Migrating from older versions

- **Profiles**: profile files from before the worktree-based profiles are
  detected automatically; run `swift-cli profile --migrate` to convert them
  (a backup is kept as `profiles.yaml.bak`; old timer databases are
  deleted).
- **Flags**: many flags were renamed for v1.0 (e.g. `-o/--out` is always
  an output directory, `new --path`/`--param` are now `-o/--out`/`--set`,
  and parameter files use `-P/--params`). See
  [CHANGELOG.md](https://github.com/WillJRoper/swiftsim-cli/blob/main/CHANGELOG.md) for the full list.

## Development

```bash
pip install -e ".[dev,test]"
pre-commit install

ruff check src tests && ruff format src tests
mypy                       # src and tests, config in pyproject.toml
pytest                     # with coverage (fails under 85%)
SWIFT_SRC=/path/to/swiftsim pytest tests/unit/test_timer_db_archetypes.py -k real
```

The last command is an optional smoke test of the timer parser against a
real SWIFT checkout (it is skipped unless `SWIFT_SRC` is set). See
[tests/README.md](https://github.com/WillJRoper/swiftsim-cli/blob/main/tests/README.md) for the test layout and
[CONTRIBUTING.md](https://github.com/WillJRoper/swiftsim-cli/blob/main/CONTRIBUTING.md) for the contribution and release
process.

Project layout:

- `src/swiftsim_cli/cli.py`: entry point; `multi_mode_args.py`: multi-mode
  parsing
- `src/swiftsim_cli/modes/`: one module per mode (`analyse/` holds the
  analysis subcommands)
- `src/swiftsim_cli/profile.py`: profile storage and git worktree management
- `src/swiftsim_cli/swiftsim_dir.py`: configure/build/update helpers
- `src/swiftsim_cli/timing/`: timer database, log scanning and the per-step
  call trees behind `analyse log`
- `src/swiftsim_cli/utilities/`: shared I/O, plotting and table helpers

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later); see
[LICENSE](https://github.com/WillJRoper/swiftsim-cli/blob/main/LICENSE).

## Citation

If you use swiftsim-cli in your research, please cite:

```bibtex
@software{swiftsim_cli,
  author = {Roper, W. J.},
  title = {swiftsim-cli: A command-line toolkit for SWIFT simulations},
  url = {https://github.com/WillJRoper/swiftsim-cli},
  version = {1.0.0},
  year = {2026}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/WillJRoper/swiftsim-cli/issues)
- **Contact**: w.roper@sussex.ac.uk
