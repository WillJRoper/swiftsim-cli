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

    assert (
        tmp_path / "out/timesteps_analysis/timesteps_analysis.png"
    ).exists()
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


def test_new_then_output_times(tmp_path, monkeypatch, write_tiny_ics):
    """output-times later in the chain reads the params.yml new writes.

    The tiny ICs (Time = 0.1, a Units group) give a cosmological run from
    the cosmo-hydro template, whose Cosmology and a_end the list uses.
    """
    monkeypatch.chdir(tmp_path)
    write_tiny_ics(tmp_path / "cosmo.hdf5", time=0.1)

    main(
        [
            "new",
            "-o",
            "run",
            "--ics",
            "cosmo.hdf5",
            "output-times",
            "-P",
            "run/params.yml",
            "--first-snap-scale-factor",
            "0.2",
            "--delta-scale-factor",
            "0.2",
            "--output-list",
            "run/output_list.txt",
        ]
    )

    lines = (tmp_path / "run" / "output_list.txt").read_text().splitlines()
    assert lines[0] == "# Scale Factor"
    assert [float(x) for x in lines[1:]] == pytest.approx(
        [0.2, 0.4, 0.6, 0.8, 1.0]
    )
