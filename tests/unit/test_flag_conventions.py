"""Tests for the CLI flag conventions shared by every mode."""

import argparse
from pathlib import Path

import pytest

from swiftsim_cli.modes import AVAILABLE_MODES
from swiftsim_cli.modes.analyse.scaling import analyse_swift_runtime_scaling
from swiftsim_cli.modes.common_args import resolve_labels
from swiftsim_cli.multi_mode_args import MultiModeCLIArgs

# Input paths are checked at parse time; run in a directory that has them
pytestmark = pytest.mark.usefixtures("in_inputs_dir")

# argv (after "analyse") for each analysis subcommand, without output flags
_ANALYSE_ARGV = {
    "timesteps": ["timesteps", "a.txt", "-l", "a"],
    "gravity-checks": ["gravity-checks", "a.dat", "-l", "a"],
    "gravity-error-maps": ["gravity-error-maps", "a.dat", "-l", "a"],
    "log": ["log", "swift.log"],
    "task-counts": ["task-counts", "swift.log"],
    "mpiuse": ["mpiuse", "reports"],
    "task-plots": [
        "task-plots",
        "tasks.dat",
        "--threadpool",
        "threadpool.dat",
    ],
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
        ["new", "-o", "run", "--inic", "ics.hdf5"],
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
        ["new", "-o", "run", "--ics", "ics.hdf5"],
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


@pytest.mark.parametrize(
    "name",
    [
        "mpiuse",
        "scaling timers",
        "scaling runtime",
        "log",
        "timesteps",
        "task-counts",
    ],
)
def test_steps_flag(name):
    """--steps takes an inclusive START END range."""
    base = ["analyse", *_ANALYSE_ARGV[name]]
    assert _parse(base).steps is None
    assert _parse([*base, "--steps", "2", "5"]).steps == [2, 5]


def test_renamed_v1_flags():
    """Flags renamed for v1.0 parse to the expected destinations."""
    assert _parse(["config", "--list-options"]).list_options is True
    # config passes unknown flags on to ./configure, so --show is not ours
    assert _parse(["config", "--show"]).options == ["--show"]
    assert _parse(["profile", "--show", "zoom"]).show == "zoom"
    assert _parse(["profile", "--show"]).show == ""
    base = ["analyse", *_ANALYSE_ARGV["timesteps"]]
    assert _parse(base).threads_per_run is None
    args = _parse([*base, "--threads-per-run", "4"])
    assert args.threads_per_run == [4]


@pytest.mark.parametrize(
    "argv",
    [
        ["profile", "-S"],
        ["analyse", *_ANALYSE_ARGV["timesteps"], "--nthreads", "4"],
    ],
)
def test_removed_v1_flags(argv):
    """Flags removed for v1.0 (profile -S, --nthreads) are rejected."""
    with pytest.raises(SystemExit) as excinfo:
        MultiModeCLIArgs(argv)
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        # -o/--out is always a directory; output-times uses --output-list
        ["output-times", "-o", "list.txt"],
        ["output-times", "--out", "list.txt"],
        # no multi-character single-dash shorts, no -u
        ["output-times", "-fz", "10"],
        ["output-times", "-dla", "0.1"],
        ["output-times", "-u", "redshift"],
        # new: --path -> -o/--out, --param -> --set
        ["new", "--path", "run", "--ics", "ics.hdf5"],
        ["new", "-o", "run", "--ics", "ics.hdf5", "--param", "A:b=1"],
        # gravity-error-maps: --resolution only
        ["analyse", *_ANALYSE_ARGV["gravity-error-maps"], "-r", "50"],
        ["analyse", *_ANALYSE_ARGV["gravity-error-maps"], "--gridsize", "5"],
        # task-plots --task -> --tasks; task-counts -t dropped
        ["analyse", *_ANALYSE_ARGV["task-plots"], "--task", "sort"],
        ["analyse", *_ANALYSE_ARGV["task-counts"], "-t", "sort"],
    ],
)
def test_flags_removed_in_cleanup(argv):
    """Flags replaced in the v1.0 flag cleanup are rejected."""
    with pytest.raises(SystemExit) as excinfo:
        MultiModeCLIArgs(argv)
    assert excinfo.value.code == 2


def test_cleanup_replacement_flags():
    """The replacement flags parse to the expected destinations."""
    args = _parse(["output-times", "--output-list", "list.txt"])
    assert args.output_list == Path("list.txt")
    assert _parse(["output-times", "--list-units", "time"]).list_units == (
        "time"
    )
    args = _parse(
        [
            "new",
            "--out",
            "run",
            "--ics",
            "ics.hdf5",
            "--set",
            "A:b=1",
            "--set",
            "C:d=e",
        ]
    )
    assert (args.out, args.set) == (Path("run"), [("A:b", "1"), ("C:d", "e")])
    args = _parse(
        ["analyse", *_ANALYSE_ARGV["gravity-error-maps"], "--resolution", "50"]
    )
    assert args.resolution == 50
    for name in ("task-plots", "task-counts"):
        args = _parse(["analyse", *_ANALYSE_ARGV[name], "--tasks", "a", "b"])
        assert args.tasks == ["a", "b"]
    assert _parse(["config", "--debug-build"]).debug_build is True


@pytest.mark.parametrize(
    "name",
    [
        "timesteps",
        "gravity-checks",
        "gravity-error-maps",
        "task-counts",
        "mpiuse",
    ],
)
def test_labels_are_optional(name):
    """-l/--labels is optional wherever it exists."""
    argv = _ANALYSE_ARGV[name][:2]  # the subcommand and its input
    assert _parse(["analyse", *argv]).labels is None


def test_default_labels_are_deduplicated_stems():
    """Default labels are file stems, with an index for shared stems."""
    assert resolve_labels(
        ["a/timesteps.txt", "b/timesteps.txt", "c.txt"], None
    ) == ["timesteps_1", "timesteps_2", "c"]
    assert resolve_labels(["x.txt"], ["mine"]) == ["mine"]
    with pytest.raises(ValueError, match="one label per input"):
        resolve_labels(["x.txt", "y.txt"], ["mine"])


@pytest.mark.parametrize(
    ("name", "default"), [("log", 0), ("task-plots", "all")]
)
def test_rank_syntax(name, default):
    """--rank takes N or 'all' everywhere, with per-mode defaults."""
    base = ["analyse", *_ANALYSE_ARGV[name]]
    assert _parse(base).rank == default
    assert _parse([*base, "--rank", "3"]).rank == 3
    assert _parse([*base, "--rank", "all"]).rank == "all"


@pytest.mark.parametrize(
    "argv",
    [
        ["make", "-j", "0"],
        ["make", "-j", "-3"],
        ["analyse", *_ANALYSE_ARGV["log"], "--steps", "9", "2"],
        ["analyse", *_ANALYSE_ARGV["log"], "--steps", "-1", "2"],
        ["analyse", *_ANALYSE_ARGV["log"], "--rank", "-1"],
        ["analyse", *_ANALYSE_ARGV["log"], "--rank", "some"],
        ["analyse", *_ANALYSE_ARGV["log"], "--top-n", "0"],
        ["analyse", *_ANALYSE_ARGV["task-plots"], "--width", "-5"],
        ["analyse", *_ANALYSE_ARGV["task-plots"], "--rank", "-2"],
        ["analyse", *_ANALYSE_ARGV["gravity-checks"], "--num-bins", "0"],
        ["analyse", *_ANALYSE_ARGV["gravity-error-maps"], "--resolution", "0"],
        ["analyse", *_ANALYSE_ARGV["timesteps"], "--threads-per-run", "0"],
        ["analyse", "timesteps", "missing.txt"],
        ["analyse", "log", "swift.log", "--swift-src", "nowhere"],
        ["new", "-o", "run", "--ics", "missing.hdf5"],
        ["profile", "--adopt", "nowhere", "--name", "x"],
    ],
)
def test_invalid_values_are_usage_errors(argv, capsys):
    """Bad numbers, ranges and missing inputs fail while parsing."""
    with pytest.raises(SystemExit) as excinfo:
        MultiModeCLIArgs(argv)
    assert excinfo.value.code == 2
    assert "error:" in capsys.readouterr().err


def _all_parsers():
    """Yield (name, parser) for every mode parser and subparser."""
    stack: list[tuple[str, argparse.ArgumentParser]] = [
        (mode, MultiModeCLIArgs._mode_parser(mode)) for mode in AVAILABLE_MODES
    ]
    while stack:
        name, parser = stack.pop()
        yield name, parser
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                stack.extend(
                    (f"{name} {sub}", p) for sub, p in action.choices.items()
                )


def test_parser_conventions():
    """Conventions every mode and subcommand parser follows."""
    for name, parser in _all_parsers():
        assert parser.allow_abbrev is False, name
        assert parser.description, name
        options = parser._option_string_actions
        # No clash with the global options
        assert not {"--debug", "--profile", "--version"} & set(options), name
        # -o/--out is always an output directory
        if "-o" in options:
            assert options["-o"].option_strings == ["-o", "--out"], name
            assert options["-o"].metavar == "DIR", name
        # Short options are single letters
        for flag in options:
            assert flag.startswith("--") or len(flag) == 2, (name, flag)


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
