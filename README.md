```
    ______       _________________        _______   ____
   / ___/ |     / /  _/ ___/_  __/ ____  / ___/ /  /  _/
   \__ \| | /| / // // /_   / /   /___/ / /  / /   / /
  ___/ /| |/ |/ // // __/  / /         / /__/ /__ / /
 /____/ |__/|__/___/_/    /_/         /____/____/___/

```

A CLI tool for automating common operations when running [SWIFT](https://gitlab.cosma.dur.ac.uk/swift/swiftsim) simulations.

The `swift-cli` command provides a unified interface for managing SWIFT simulation directories, generating parameter files, analyzing outputs, and automating common workflows.

## Features

- **Multi-mode execution**: Run multiple commands in sequence (e.g., `swift-cli config --enable-debug make -j 32`).
- **SWIFT repository management**: Configure, compile, and update SWIFT installations.
- **Maintain multiple SWIFT configurations seamlessly**: Each SWIFT "profile" is its own git worktree on its own branch with its own configure flags and build, so switching never needs a rebuild.
- **Output analysis**: Analyze simulation logs and diagnostics.
- **Simulation setup**: Create new simulation directories containing everything needed to run a SWIFT simulation out of the box.
- **Automate common calculations**: Provides a suite of common operations such as generating output time sequences, managing softening lengths, and more.

## Installation

### Requirements

- Python ≥ 3.9
- Git (for SWIFT repository management)
- To run SWIFT you will also need all of SWIFT's dependencies (for more details see [SWIFT's documentation](https://swift.strw.leidenuniv.nl/docs/index.html))

### Install from source

```bash
# Clone the repository
git clone https://github.com/WillJRoper/swiftsim-utils.git
cd swiftsim-utils

# Install in development mode with all dependencies
pip install -e .
```

## Quick Start

Once installed you'll need to initialise your first profile (you can have an arbitrary number of different profiles). To do so just invoke:

```bash
swift-cli profile --init
```

This will prompt you for your main SWIFT clone, where profile worktrees should live (default: a `swiftsim-profiles` directory next to the clone), the first profile's name, the upstream branch it follows, its configure flags, your data directory and some constants. It then creates the profile's worktree.

To view the profile you just set up you can run:

```bash
swift-cli profile --show
```

### Profiles

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
swift-cli --profile zoom make -j 16   # build another profile without switching
```

Updating a profile runs `git pull --rebase` in its worktree (your local commits are replayed on top of the upstream). If the rebase hits conflicts nothing is built and you're told how to resolve them. Otherwise `./autogen.sh` is run if `./configure` is missing, `./configure` is (re)run only if the profile has never been configured or its configure flags changed, and then `make` is run.

`swift-cli config` sets the current profile's configure flags (from presets such as `--debug` and/or any `--<option>` flags) and reconfigures. Flags are validated against the options listed by `./configure --help` in that profile's worktree, with suggestions for typos.

Already have worktrees made by hand? Register them with:

```bash
swift-cli profile --adopt ../swiftsim-my-feature --name my-feature
```

Profiles are stored in `~/.swiftsim-utils/profiles.yaml` and per-profile caches (e.g. the timer database used by `analyse log`) in `~/.swiftsim-utils/cache/<name>/`. If you have a profile file from an older version of `swift-cli`, convert it (after reviewing the plan it prints) with:

```bash
swift-cli profile --migrate
```

## Core Modes

- **`profile`**: Manage SWIFT profiles (one git worktree, branch and build each)
- **`config`**: Configure SWIFT compilation options (including presets and arbitrary `--<option>` flags)
- **`make`**: Compile SWIFT with specified number of processes
- **`update`**: Pull (rebase onto upstream) and rebuild the current profile
- **`switch`**: Switch the current profile
- **`new`**: Create new simulation run directory
- **`analyse`**: Analyze SWIFT logs and diagnostic files
- **`output-times`**: Generate output time sequences for simulations

### Exploring Modes

Each mode has its own help message. To see the available modes simply run:

```bash
swift-cli --help
```

This will show you the available modes.

```bash
swift-cli: Utilities for Swift development workflows

Usage:
  swift-cli [--verbose] [--swift-dir DIR] [--data-dir DIR] <mode1> [mode1_args] [<mode2> [mode2_args]] ...

Global options:
  -v, --verbose     Enable verbose output
  --swift-dir DIR   Path to SWIFT directory
  --data-dir DIR    Path to data directory

Available modes:
  profile         Profile mode for configuring SWIFTsim-CLI settings and profiles
  config          A module containing tools for configuring SWIFT
  output-times    Output-times mode for generating output time lists
  update          Update mode for pulling latest SWIFT changes
  switch          Switch mode for switching SWIFT branches
  make            Make mode for compiling SWIFT
  new             New mode for creating new SWIFT run directories
  analyse         Analyse mode for analysing SWIFT runs

Examples:
  swift-cli config --enable-debug
  swift-cli make -j 8
  swift-cli config --enable-debug make -j 32
  swift-cli update config --enable-debug make -j 8

For mode-specific help: swift-cli <mode> --help
```

For mode specific help just run:

```bash
swift-cli <mode> --help
```

e.g. for the analyse mode:

```bash
swift-cli analyse --help
```

Which yields:

```bash
usage: swift-cli analyse [-h] {timesteps,gravity-checks,gravity-error-maps,log,task-counts,mpiuse} ...

positional arguments:
  {timesteps,gravity-checks,gravity-error-maps,log,task-counts,mpiuse}
                        Type of analysis to perform
    timesteps           Analyse timestep files
    gravity-checks      Analyse gravity check files
    gravity-error-maps  Create hexbin error maps for gravity check files
    log                 Analyse timing information from SWIFT log files. To get the most from this mode SWIFT should be run with -v 1 for verbose output.
    task-counts         Analyse engine_print_task_counts output from SWIFT log files.
    mpiuse              Analyse SWIFT mpiuse_report-rank*-step*.dat files.

options:
  -h, --help            show this help message and exit
```

### Multi-mode Usage

Commands can be chained together for complex workflows:

```bash
# Update SWIFT, configure with debug mode, and compile
swift-cli update config --debug make -j 16

# Update SWIFT, compile, and create new simulation in the test_run directory
swift-cli update make -j 32 new test_run --ics initial_conditions.hdf5
```

### MPI Use Analysis

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
- Per-step and per-subtype plots in `<prefix>_mpiuse_analysis/`
- Optional per-step CSV output with `--format csv` or `--format both`

Useful filters include `--subtypes gpart grav_counts` and
`--steps START END`.

## Development

### Code Quality

```bash
# Run linting and formatting
ruff check --fix
ruff format

# Type checking
mypy src/swiftsim_cli/

# Run tests
pytest
```

### Project Structure

- `src/swiftsim_cli/cli.py`: Main CLI entry point
- `src/swiftsim_cli/modes/`: Individual command mode implementations
- `src/swiftsim_cli/multi_mode_args.py`: Multi-mode argument parsing
- `src/swiftsim_cli/profile.py`: Profile storage and git worktree management
- `src/swiftsim_cli/swiftsim_dir.py`: Configure/build/update helpers for a profile's worktree

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e ".[dev,test]"`)
4. Install pre-commit hooks (`pre-commit install`)
5. Make your changes and add tests
6. Run the test suite (`pytest`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use swiftsim-utils in your research, please cite:

```bibtex
@software{swiftsim_cli,
  author = {Roper, W. J.},
  title = {swiftsim-utils: A CLI tool for SWIFT simulations},
  url = {https://github.com/WillJRoper/swiftsim-utils},
  version = {1.0.0}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/WillJRoper/swiftsim-utils/issues)
- **Documentation**: Coming Soon
- **Contact**: w.roper@sussex.ac.uk
