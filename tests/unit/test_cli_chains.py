"""End-to-end multi-mode chains through cli.main, with real modes."""

import pytest

from swiftsim_cli.cli import main


def test_analyse_chain_runs_every_mode(logs_dir, tmp_path, capsys):
    """Two analyses chained in one command both write their outputs."""
    log = str(logs_dir / "sample.txt")
    out = str(tmp_path / "out")

    main(
        [
            "analyse",
            "timesteps",
            log,
            "-o",
            out,
            "analyse",
            "task-counts",
            log,
            "--steps",
            "1",
            "3",
            "-o",
            out,
        ]
    )

    assert (tmp_path / "out/timesteps_analysis/timestep_analysis.png").exists()
    counts = tmp_path / "out" / "task_counts_analysis"
    assert (counts / "task_counts_per_step.png").exists()
    assert "Found task counts for 3 steps" in capsys.readouterr().out


def test_failing_mode_stops_the_chain(logs_dir, tmp_path, capsys):
    """A mode that fails exits 1 before the next mode runs."""
    quiet = tmp_path / "quiet.log"
    quiet.write_text("no task counts here\n")
    out = str(tmp_path / "out")

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "analyse",
                "task-counts",
                str(quiet),
                "-o",
                out,
                "--",
                "analyse",
                "timesteps",
                str(logs_dir / "sample.txt"),
                "-o",
                out,
            ]
        )

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "error: no engine_print_task_counts output" in err
    assert "(run with --debug for the full traceback)" in err
    assert not (tmp_path / "out" / "timesteps_analysis").exists()


def test_usage_error_in_a_later_mode_runs_nothing(logs_dir, tmp_path):
    """Every mode is parsed before any runs, so a bad flag runs nothing."""
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "analyse",
                "timesteps",
                str(logs_dir / "sample.txt"),
                "-o",
                str(out),
                "make",
                "-j",
                "0",
            ]
        )
    assert excinfo.value.code == 2
    assert not out.exists()
