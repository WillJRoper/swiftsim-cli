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
  `params.yml` from a bundled template, filled in from your initial
  conditions and checked against SWIFT's parameter parser.
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
# 1. Set up: clone SWIFT into ./swiftsim and create the "default" profile
swift-cli init
#    Or use an existing clone:  swift-cli init --swift-path ~/swiftsim

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
swift-cli profile --show zoom         # settings of one profile (default: current)
swift-cli show zoom                   # the same, plus its parameter file's cosmology
swift-cli profile --switch zoom       # or: swift-cli switch zoom
swift-cli profile --edit zoom         # change upstream, flags, paths, constants
swift-cli profile --edit zoom --set upstream=origin/zoom-v2 --set softening_coeff=0.04
swift-cli profile --update zoom -j 16 # or, for the current profile: swift-cli update -j 16
swift-cli profile --update-all -j 16  # update every profile, then print a summary
swift-cli profile --remove zoom       # removes the worktree (and optionally the branch)
swift-cli profile --remove zoom --force -y  # even with untracked files; keeps the branch
swift-cli profile --new zoom --reuse-branch  # recreate it on the kept swift-cli/zoom branch
swift-cli --profile zoom make -j 16   # build another profile without switching
```

`swift-cli init` sets everything up once, without prompts: it uses the SWIFT
clone given with `--swift-path` or clones SWIFT (`--swift-url`, default the
GitHub mirror) into `./swiftsim`, then creates the first profile (`--name`,
default `default`; `--upstream`; `--configure-flags`) with its worktree in
`--worktree-root` (default `~/.swiftsim-cli/worktrees`). Every later profile
is another worktree of that clone, added with `profile --new`.

`profile --edit NAME` opens the profile's settings (`upstream`,
`configure_flags`, `data_dir`, `parameter_file`, `softening_coeff`,
`softening_pivot_z`) as YAML in `$VISUAL`/`$EDITOR` (default `vi`); the
changes are checked and applied when you save and quit.
`profile --edit NAME --set KEY=VALUE` (repeatable) changes only the given
settings without opening an editor.
`-y/--yes` answers the confirmation of `--remove`
(`--remove --yes` keeps the profile's branch and prints the `git branch -D`
command).

Updating a profile runs `git pull --rebase --autostash` in its worktree (your local commits are replayed on top of the upstream, and any uncommitted changes are stashed and re-applied afterwards). It fails up front if the branch has no upstream, and reports "Already up to date" or how many commits were pulled. If the rebase, or re-applying your stashed changes, hits conflicts nothing is built and you're told how to resolve them. Otherwise `./autogen.sh` is run if `./configure` is missing, `./configure` is (re)run only if the profile has never been configured or its configure flags changed, and then `make` is run. No build starts in a worktree that is mid-rebase or has unresolved conflicts, and a failed `configure` or `make` exits with that command's own exit code.

`update --no-build` only pulls. When `update` is followed by `config` or `make` in the same command it skips its own build, so `swift-cli update config --debug-build make -j 8` builds once.

Changing a profile's upstream with `--edit` moves a branch with no commits of its own to the new upstream (refused if the worktree has uncommitted changes); a branch with its own commits keeps them, and the `git rebase --onto` command to move them is printed.

`swift-cli config` sets the current profile's configure flags (from presets such as `--debug-build` and/or any `--<option>` flags) and reconfigures. Flags are validated against the options listed by `./configure --help` in that profile's worktree, with suggestions for typos.

Already have worktrees made by hand? Register them with:

```bash
swift-cli profile --adopt ../swiftsim-my-feature --name my-feature
```

`--remove` deletes only the worktrees swift-cli created; adopted worktrees,
the main clone and worktrees shared with another profile are only
unregistered.

Profiles are stored in `~/.swiftsim-cli/profiles.yaml` and per-profile caches (e.g. the timer database used by `analyse log` and the task names used by `analyse task-plots`) in `~/.swiftsim-cli/cache/<name>/`.

## Global options

Global options must come before the first mode:

- `--profile NAME`: run every mode that acts on one profile against
  profile `NAME` instead of the current one (e.g.
  `swift-cli --profile zoom make -j 16`). The name is checked before any
  mode runs; `profile --list`, `--new` and `--switch` ignore it.
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
  init            Clone or reuse SWIFT and create the first profile.
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
| `profile` | Create, list, show, switch, edit, update, remove and adopt profiles |
| `config` | Set the current profile's configure flags and reconfigure |
| `make` | Compile the current profile |
| `update` | Pull (rebase onto upstream) and rebuild the current profile |
| `switch` | Make another profile the current one |
| `show` | Show a profile (default: the current one) and its parameter file's cosmology |
| `new` | Create a run directory with a `params.yml` |
| `output-times` | Generate an output-time list |
| `analyse` | Analyse SWIFT logs and diagnostics (subcommands below) |
| `utils` | Small helper calculations (`softenings`) |

### `config`

```bash
swift-cli config --list-options                    # ./configure --help
swift-cli config --debug-build                     # preset flags
swift-cli config --eagle --with-cooling=COLIBRE    # preset + any --<option>
swift-cli config --with-hydro=sphenix CC=icx       # options and VAR=value
swift-cli config                                   # reconfigure with stored flags
```

Presets: `--debug-build`, `--gravity`, `--eagle`, `--eaglexl` (`--eagle`
and `--eaglexl` are mutually exclusive). Any other `--<option>[=value]`
flag is validated against `./configure --help` in the profile's worktree
(with suggestions for typos) and passed on, as are `VAR=value`
assignments; anything else is rejected (`--with-hydro sphenix` suggests
`--with-hydro=sphenix`). A later value for an option overrides an earlier
one, with a warning. The resulting flags replace the profile's stored
configure flags.

### `make`, `update`, `switch`, `show`

```bash
swift-cli make -j 16             # -j/--nr-threads, --clean runs make clean first
swift-cli update -j 16           # pull, reconfigure if needed, build
swift-cli update --no-build      # only pull
swift-cli switch zoom            # same as profile --switch zoom
swift-cli show                   # current profile and its parameter file's cosmology
swift-cli show zoom --cosmo      # only the cosmology of profile zoom
```

### `new`

```bash
swift-cli new -o runs/L100 --ics ics/L100.hdf5 --set Snapshots:basename=L100
swift-cli new -o runs/L100 --ics ics/L100.hdf5 --template cosmo-hydro --check
swift-cli new -o runs/L100 --ics ics/L100.hdf5 -P my_params.yml --force
```

Writes `runs/L100/params.yml`, starting from one of swift-cli's bundled,
runnable parameter templates (derived from SWIFT's own examples):

| Template | For | SWIFT flags |
| --- | --- | --- |
| `cosmo-dmo` | cosmological, dark matter only | `--cosmology --self-gravity` |
| `cosmo-hydro` | cosmological, gravity + hydro | `--cosmology --self-gravity --hydro` |
| `isolated-gravity` | non-cosmological, non-periodic, gravity | `--self-gravity` |
| `isolated-hydro` | non-cosmological, non-periodic, gravity + hydro | `--self-gravity --hydro` |

The template is chosen from the ICs (cosmological if the header has a
positive `Redshift` or 0 < `Time` < 1; hydro if there is gas).
`--template NAME` picks one explicitly; `-P/--params FILE` (or the
profile's `parameter_file`, unless it is SWIFT's `parameter_example.yml`)
starts from your own file instead. `new` then fills in:

- absolute paths to the ICs and to `snapshots/` and `restart/` in the run
  directory;
- the unit system from the ICs' `Units` group; cosmological ICs without one
  are taken to be Gadget-style (h^-1 units, sqrt(a) velocities), so
  `cleanup_h_factors` and `cleanup_velocity_factors` are set to 1;
- `a_begin` (cosmological) or `time_begin`, with the first snapshot and
  statistics times to match; non-cosmological runs get no `Cosmology`
  section;
- softenings: the profile's `softening_coeff` times the mean
  inter-particle separation (comoving, with the maximal physical softening
  at the pivot redshift, for cosmological runs; maximal physical only, with
  a warning, for non-cosmological runs). Zoom ICs (a `PartType2`
  background) use the high-resolution particle mass and set
  `Gravity:softening_ratio_background`, plus a `ZoomRegion` section and the
  `--zoom` flag when the profile's SWIFT has zoom support
  (`src/zoom_region`);
- `generate_gas_in_ics` when a hydro template is used with DM-only ICs;
- your `--set SECTION:KEY=VALUE` overrides (repeatable): integers and
  floats are written as numbers, `[a, b]` as a list, anything else
  verbatim; new keys or sections are added with a warning (in case of a
  typo), and empty values are rejected.

Before anything is written the result is checked against SWIFT's parameter
parser (no duplicate keys or sections, lines of at most 255 characters, at
most 700 parameters and 64 sections, no placeholder values such as
`FLT_MAX`); the run directory is created only once it passes. `new` prints
the softenings it wrote and the `swift ...` command to run with, adding
`--stars` when the ICs contain stars (SWIFT drops them otherwise). An
existing `params.yml` is only replaced with `--force`.

`--check` runs the profile's built `swift` binary with `--dry-run` and the
right physics flags on the new file, from inside the run directory, and
reports SWIFT's verdict.

### `output-times`

```bash
# Snapshots every 0.5 in redshift from z=10 to a_end in the params file
swift-cli output-times -P params.yml --first-snap-z 10 --delta-z 0.5

# Snapshots every 0.05 in scale factor plus snipshots every 0.01
swift-cli output-times -P params.yml --first-snap-scale-factor 0.1 \
    --delta-scale-factor 0.05 --snipshot-delta-scale-factor 0.01 \
    --output-list output_list.txt --force

# Non-cosmological run: times in internal units
swift-cli output-times --first-snap-time 0 --delta-time 0.1 --final-snap-time 2
```

The first output, the final output, the snapshot interval and the snipshot
interval are each given in one unit (`--first-snap-{z,time,scale-factor}`,
`--final-snap-...`, `--delta-{z,time,scale-factor,log-scale-factor}`,
`--snipshot-delta-...`); contradictory, zero or negative values are usage
errors. `--delta-log-scale-factor` is the interval in log10(a). The final
output defaults to `Cosmology:a_end` from the parameter file.

The list is written to `--output-list FILE` (default: `output_list.txt` in
the current directory); an existing list is only replaced with `--force`.
The interval (`--delta-*`) chooses the units of the list; `--list-units`
overrides them. The list stops at the final output, snapshot and snipshot
entries are never duplicated (snapshots win), and it must hold at least two
outputs. With snipshots the list gets a `Select Output` column, and SWIFT
also needs `Snapshots:select_output_on: 1` and a select-output file (a note
is printed).

With a cosmology, `--*-time` values are given as ages of the Universe in
Gyr, and the list is written as ages in SWIFT's *internal* time units
(UnitLength_in_cgs / UnitVelocity_in_cgs from the parameter file's
`InternalUnitSystem`), which is what SWIFT expects for a `# Time` list in a
cosmological run. `--no-cosmology` ignores the parameter file's `Cosmology`
section. Redshifts and scale factors need no cosmology; only times do.
Conversions follow SWIFT's own cosmology (radiation, neutrinos, w0/wa).

### `utils`

```bash
swift-cli utils softenings --ics ics.hdf5
swift-cli utils softenings --ics ics.hdf5 -P params.yml --softening-coeff 0.04
```

Prints the mean inter-particle separation and the comoving/maximal physical
softening of each particle species (dark matter, baryons, neutrinos, and
the zoom background), using exactly the calculation `new` uses. `-P` is
optional (the default is the template `new` would pick); the coefficient
and pivot redshift default to the profile's values.

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
                      files per step.
    mpiuse            Analyse SWIFT mpiuse_report-rank*-step*.dat files and
                      compare communication volume across inputs.
    task-plots        Plot SWIFT task-debug timings (optionally with a
                      threadpool dump) per thread.
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

### Timesteps (`analyse timesteps`)

```bash
# Timestep files (or logs) of two runs, in core-hours
swift-cli analyse timesteps run1/timesteps.txt run2/timesteps.txt \
    -l run1 run2 --cores-per-run 64 128 --time-column redshift
```

Plots the cumulative wall-clock and dead time of each run against
`--time-column`: `auto` (default) uses the scale factor if it varies in
every input and otherwise time (non-cosmological runs), with a note when it
falls back; `step`, `time`, `scale-factor` and `redshift` choose
explicitly. `--cores-per-run N ...` (MPI ranks x threads per rank, one per
run) turns the times into core-hours.
`--match-runtimes` trims every run to the shortest x range. The runtime
comparison table is skipped, with a warning, when the x column is constant
or decreasing.

Steps repeated after a restart replace the earlier rows ("N rows
superseded by a restart"), malformed rows are counted and skipped, and gaps
in the step sequence are warned about and break the plotted lines. The
`Dead time [ms]` column is optional (older SWIFT versions). The plot is
`timesteps_analysis/timesteps_analysis.png`.

### Gravity accuracy (`analyse gravity-checks`, `analyse gravity-error-maps`)

```bash
swift-cli analyse gravity-checks gravity_checks_exact_step0000.dat --all-orders
swift-cli analyse gravity-error-maps gravity_checks_exact_step0000.dat \
    --resolution 50 --vmin 1e-6 --vmax 1e-2 --box-size 100
```

Both need SWIFT configured with `--enable-gravity-force-checks=N`. Give
either file of a pair (`gravity_checks_exact_*` or
`gravity_checks_swift_*`); the counterpart
(`..._step<N>[_order<K>].dat`) is found next to it and particles are
matched by ID. An exact file with SWIFT files of several multipole orders
is compared with the lowest order unless `--all-orders` is given.
Particles with non-finite values are dropped with a warning, errors are
undefined where the exact acceleration is zero (counted by
`gravity-checks`, dropped with a warning by `gravity-error-maps`), and a
warning is given when the exact and SWIFT positions differ.

`gravity-checks` prints one table per run (median, 90% and 99% error per
component, the shares below `--min-error` and above `--max-error`, and the
undefined count) and
plots error histograms with a percentile summary panel.
`gravity-error-maps` maps the median relative error per hexagon (or per
particle for samples of up to 5000) in three projections, plus a binary
map of the particles above `--thresh` with an error histogram. The colour
scale runs from the 1st to the 99th percentile of the errors unless
`--vmin`/`--vmax` are given; distances are measured from the median
particle position, wrapped periodically with `--box-size L`.

### Task counts (`analyse task-counts`)

```bash
swift-cli analyse task-counts swift.log --tasks self pair --steps 10 100
```

Plots the `engine_print_task_counts` output (SWIFT run with `-v 1`) per
step. The total uses SWIFT's `System total` (all ranks) when it is printed,
otherwise the sum of each rank's total; the per-type counts are the tasks
that run (skipped tasks are one `skipped (any type)` row). A log that only
has some ranks' counts (typically rank 0 of an MPI run) gives a warning
and the axis labels say "(rank 0 only)". Unknown `--tasks` names are
errors.

### Task plots (`analyse task-plots`)

```bash
# Task-debug (-y) timings, optionally with the threadpool (-Y) dump above
swift-cli analyse task-plots thread_info-step64.dat \
    --threadpool threadpool_info-step64.dat \
    --tasks pair/force self/force --activity-plot
swift-cli analyse task-plots thread_info_MPI-step64.dat --rank 0 --c-type zoom
```

Needs SWIFT configured with `--enable-task-debugging` and run with `-y N`
(and, for `--threadpool`, `--enable-threadpool-debugging` and `-Y N`).
Every `task_dump_all` layout is read: upstream 11-column (non-MPI) and
13-column (MPI) dumps, older zoom 17-column MPI rows, and zoom 19/21-column
dumps; the cell filters (`--c-type`, `--ci-subtype`, ..., by name or index)
need a 19/21-column dump.

Task, subtype and cell-type names are parsed from the SWIFT source
(`src/task.c`, `src/cell.h`) of the active profile's checkout, or of
`--swift-src DIR`, and cached per profile. Without a source tree built-in
tables are used with a warning; indices outside the tables warn and show
as `unknown_N`. Unknown `--tasks` labels or cell names are errors listing
the valid names, and ranks with no matching task are skipped with a
warning. All ranks share one time window (by default the longest step in
the dump headers; `--limit`, `--mintic`).

### MPI use (`analyse mpiuse`)

Compares MPI communication recorded by SWIFT when built with
`--enable-mpiuse-reports`. Each input is a directory, glob, or
`mpiuse_report-rank*-step*.dat` file set representing one run; several
inputs are compared side by side.

```bash
swift-cli analyse mpiuse master_reports fewer_gpart_comms_reports \
  --labels master fewer_gpart_comms \
  --prefix gravity_cmp \
  --format both \
  --per-rank
```

Each message is logged by SWIFT as a send and a receive; volumes are
counted once (from the send side), and bytes sent and received are reported
separately. The output has per-input summary tables (with `TOTAL send` and
`TOTAL recv` rows), a cross-input comparison table, per-step and
per-subtype plots, per-rank sent/received tables and plots with
`--per-rank`, and per-step CSV files with `--format csv|both`. `--subtypes
gpart grav_counts` restricts everything, including the in-flight
high-water mark, to those subtypes; `--steps START END` restricts the
steps.

### Scaling (`analyse scaling`)

```bash
swift-cli analyse scaling timers logs/*.log --steps 10 100 -p zoom
swift-cli analyse scaling timers logs/*.log --rank-source all --swift-src ~/swiftsim
swift-cli analyse scaling runtime logs/*.log
```

Both compare logs of runs with at least two different MPI rank counts
(and warn when their step counts differ by more than 1%).
`scaling timers` uses the same per-step timer trees as `analyse log` (and
the same timer database flags `--swift-src`, `--no-match-revision`,
`--regenerate-db`) and plots the timers whose exclusive time is at least
`--min-percent-threshold` percent of the step time in at least one log
(default: 5). `--rank-source root` (default) uses rank 0; `all` uses the
mean over the ranks that printed timers (`-v 2` logs). A "SCALING INPUTS"
table lists each log's ranks, steps, step time, timed share and unmatched
timer lines, and `rank_imbalance_summary.txt` gives each timer's
max/mean over ranks.
`scaling runtime` uses the summed step-table wall-clock time. Both leave
out step 0 and restart rows, like `analyse log`.

### Log timing analysis (`analyse log`)

`analyse log` shows where the time of a run goes, from the timers SWIFT
prints with `-v 1` (rank 0) or `-v 2` (every rank):

```bash
swift-cli analyse log swift.log                     # rank 0, all steps
swift-cli analyse log swift.log --steps 100 200 --top-n 10
swift-cli analyse log swift.log --rank all          # per rank + imbalance
swift-cli analyse log swift.log --hierarchy-functions engine_rebuild
swift-cli analyse log swift.log --swift-src ~/swiftsim --regenerate-db
```

- Timers are read from the SWIFT source with tree-sitter (including
  `examples/main.c`). By default the source is taken at the revision
  printed in the log (extracted with `git archive` from the profile's
  repository into the profile cache; the checkout is not touched);
  `--no-match-revision` uses the checkout as it is, and `--swift-src DIR`
  any other source tree (it must contain C files). The parsed database is
  cached and rebuilt only when the source changes (`--regenerate-db`
  forces a rebuild). The share of timer lines matched to the source is
  always reported, with a warning when more than 5% are unmatched.
- The timers of each step are nested into the call tree that actually ran
  (a function called from several places is attributed to the caller that
  contained each call), so nothing is counted twice. All percentages are of
  the step-table wall-clock time: the top-level breakdown, including the
  untimed remainder, sums to 100%. Exclusive (self) times add up; inclusive
  times of nested timers do not. A warning is given if any step's coverage
  exceeds 105%.
- The table row SWIFT prints for step N comes at the start of step N+1, so
  the timers before row N are step N's; those before row 0 are
  initialisation. Timers `main()` prints between steps (for example
  `task_dump_all`) are reported in a separate "BETWEEN STEPS" table and are
  not part of any step.
- MPI ranks are never summed: `--rank N` picks one rank (default 0) and
  `--rank all` analyses each rank and compares them.
- Interleaved or damaged lines (as written by several ranks at once) are
  split and rebuilt where possible; a warning reports how many lines were
  cut, recovered or dropped, and steps whose rows are missing are left out
  of the per-step analysis.
- `--steps` outside the log's steps is an error naming the available range.

The console shows a short summary; `log_analysis/analysis_tables.txt` has
the call tree, the timers by exclusive time, the per-caller breakdown of the
`--hierarchy-functions`, the steps whose coverage is outside 90-105%,
initialisation, the between-steps timers, task categories and task counts,
alongside the charts. The coverage of every step is in
`log_analysis/per_step_coverage.csv`. Outputs left from an earlier run in
the same directory are removed first.

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
  analysis subcommands, `analyse/scaling/` the scaling package)
- `src/swiftsim_cli/templates/`: the parameter templates used by `new`
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
