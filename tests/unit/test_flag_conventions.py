"""Tests for the CLI flag conventions shared by every mode."""

from pathlib import Path

import pytest

from swiftsim_cli.modes.analyse.scaling import analyse_swift_runtime_scaling
from swiftsim_cli.multi_mode_args import MultiModeCLIArgs

# argv (after "analyse") for each analysis subcommand, without output flags
_ANALYSE_ARGV = {
    "timesteps": ["timesteps", "a.txt", "-l", "a"],
    "gravity-checks": ["gravity-checks", "a.dat", "-l", "a"],
    "gravity-error-maps": ["gravity-error-maps", "a.dat", "-l", "a"],
    "log": ["log", "swift.log"],
    "task-counts": ["task-counts", "swift.log"],
    "mpiuse": ["mpiuse", "reports"],
    "task-plots": ["task-plots", "tasks.dat", "threadpool.dat"],
    "scaling timers": ["scaling", "timers", "a.log", "b.log"],
    "scaling runtime": ["scaling", "runtime", "a.log", "b.log"],
}


def _parse(argv):
    """Parse a single-mode command line and return its namespace."""
    ((_, args),) = MultiModeCLIArgs(argv).modes
    return args


@pytest.mark.parametrize("name", sorted(_ANALYSE_ARGV))
def test_analyse_output_flags(name):
    """Every analysis takes -o/--out, -p/--prefix and --show."""
    base = ["analyse", *_ANALYSE_ARGV[name]]

    args = _parse(base)
    assert (args.out, args.prefix, args.show) == (None, None, False)

    args = _parse([*base, "-o", "res", "-p", "run1", "--show"])
    assert (args.out, args.prefix, args.show) == (Path("res"), "run1", True)

    args = _parse([*base, "--out", "res", "--prefix", "run1"])
    assert (args.out, args.prefix) == (Path("res"), "run1")


@pytest.mark.parametrize("name", sorted(_ANALYSE_ARGV))
@pytest.mark.parametrize("old_flag", ["--output-path", "--show-plot"])
def test_analyse_old_output_flags_are_gone(name, old_flag):
    """The old output flag spellings are rejected."""
    argv = ["analyse", *_ANALYSE_ARGV[name], old_flag]
    if old_flag == "--output-path":
        argv.append("res")
    with pytest.raises(SystemExit) as excinfo:
        MultiModeCLIArgs(argv)
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "name", ["timesteps", "gravity-checks", "gravity-error-maps"]
)
def test_file_positionals_are_paths(name):
    """File positional arguments are parsed as Paths."""
    args = _parse(["analyse", *_ANALYSE_ARGV[name]])
    assert all(isinstance(f, Path) for f in args.files)


@pytest.mark.parametrize(
    "argv",
    [
        ["output-times", "-P", "p.yml"],
        ["output-times", "--params", "p.yml"],
        ["utils", "softenings", "--ics", "ics.hdf5", "-P", "p.yml"],
        ["utils", "softenings", "--ics", "ics.hdf5", "--params", "p.yml"],
    ],
)
def test_params_flag(argv):
    """The parameter file is -P/--params everywhere."""
    assert _parse(argv).params == Path("p.yml")


@pytest.mark.parametrize(
    "argv",
    [
        ["output-times", "-p", "p.yml"],
        ["utils", "softenings", "--ics", "ics.hdf5", "-p", "p.yml"],
        ["utils", "softenings", "--inic", "ics.hdf5", "-P", "p.yml"],
        ["new", "--path", "run", "--inic", "ics.hdf5"],
    ],
)
def test_old_params_and_ics_flags_are_gone(argv):
    """-p (for params) and --inic are no longer accepted."""
    with pytest.raises(SystemExit) as excinfo:
        MultiModeCLIArgs(argv)
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["new", "--path", "run", "--ics", "ics.hdf5"],
        ["utils", "softenings", "--ics", "ics.hdf5", "-P", "p.yml"],
    ],
)
def test_ics_flag(argv):
    """The initial conditions file is --ics everywhere."""
    assert _parse(argv).ics == Path("ics.hdf5")


@pytest.mark.parametrize("mode", ["make", "update", "profile"])
@pytest.mark.parametrize("flag", ["-j", "--nr-threads"])
def test_nr_threads_flag(mode, flag):
    """Compilation threads are -j/--nr-threads, defaulting to 1."""
    assert _parse([mode]).nr_threads == 1
    assert _parse([mode, flag, "8"]).nr_threads == 8


@pytest.mark.parametrize("name", ["mpiuse", "scaling timers"])
def test_steps_flag(name):
    """--steps takes an inclusive START END range."""
    base = ["analyse", *_ANALYSE_ARGV[name]]
    assert _parse(base).steps is None
    assert _parse([*base, "--steps", "2", "5"]).steps == [2, 5]


def _write_runtime_log(path: Path, ranks: int, wallclock: float) -> None:
    """Write a minimal SWIFT log with a step table."""
    path.write_text(
        "\n".join(
            [
                f"[0000] main: MPI is up and running with {ranks} node(s).",
                "#   Step  Time Scale-factor Redshift Time-step Time-bins "
                "Updates g-Updates s-Updates Wall-clock time [ms] Props "
                "Dead time [ms]",
                f"  1 0.0 0.0 0.0 0.0 1 56 0 0 0 {wallclock} 265 0.0",
                f"  2 0.0 0.0 0.0 0.0 1 56 0 0 0 {wallclock} 265 0.0",
            ]
        ),
        encoding="utf-8",
    )


def test_scaling_prefix_applied_once(tmp_path):
    """Scaling outputs go to <out>/<subdir>/ with the prefix used once."""
    logs = [tmp_path / "a.log", tmp_path / "b.log"]
    _write_runtime_log(logs[0], 1, 100.0)
    _write_runtime_log(logs[1], 2, 60.0)

    analyse_swift_runtime_scaling(
        [str(log) for log in logs],
        output_path=tmp_path / "res",
        prefix="p",
    )

    out_dir = tmp_path / "res" / "scaling_runtime_analysis"
    assert sorted(p.name for p in (tmp_path / "res").iterdir()) == [
        "scaling_runtime_analysis"
    ]
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "p_runtime_scaling.png",
        "p_runtime_scaling_summary.txt",
    ]
