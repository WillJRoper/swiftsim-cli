"""Tests for the mpiuse analysis module."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from swiftsim_cli.modes.analyse.mpiuse import (
    _aggregate,
    _discover_files,
    _read_mpiuse_file,
    add_mpiuse_arguments,
    analyse_swift_mpiuse,
    run_swift_mpiuse,
)


def _write_mpiuse_file(
    directory: Path,
    rank: int,
    step: int,
    rows: list[str],
) -> Path:
    path = directory / f"mpiuse_report-rank{rank}-step{step}.dat"
    content = [
        (
            "# stic etic dtic step rank otherrank type itype subtype "
            "isubtype activation tag size sum"
        ),
        *rows,
        "##",
        "## Number of requests: 2",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def _make_sample_input(base: Path, scale: int = 1) -> Path:
    run_dir = base
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_mpiuse_file(
        run_dir,
        rank=0,
        step=1,
        rows=[
            (
                f"10 100 0 1 0 1 send 1 gpart 4 1 10 {100 * scale} "
                f"{100 * scale}"
            ),
            "11 101 5 1 0 1 send 1 gpart 4 0 10 -100 0",
            (f"12 102 0 1 0 1 recv 2 rho 5 1 11 {50 * scale} {50 * scale}"),
            "13 103 4 1 0 1 recv 2 rho 5 0 11 -50 0",
        ],
    )
    _write_mpiuse_file(
        run_dir,
        rank=1,
        step=1,
        rows=[
            (
                f"15 105 0 1 1 0 send 1 gpart 4 1 12 {120 * scale} "
                f"{120 * scale}"
            ),
            "16 106 6 1 1 0 send 1 gpart 4 0 12 -120 0",
        ],
    )
    _write_mpiuse_file(
        run_dir,
        rank=0,
        step=2,
        rows=[
            (
                f"20 200 0 2 0 1 send 1 grav_counts 6 1 20 {80 * scale} "
                f"{80 * scale}"
            ),
            "21 201 5 2 0 1 send 1 grav_counts 6 0 20 -80 0",
            (f"22 202 0 2 0 1 recv 2 xv 7 1 21 {40 * scale} {40 * scale}"),
            "23 203 4 2 0 1 recv 2 xv 7 0 21 -40 0",
        ],
    )

    return run_dir


class TestMpiuseArguments:
    """Tests for mpiuse CLI argument setup."""

    def test_add_mpiuse_arguments_basic(self):
        """Test positional inputs and prefix parsing."""
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()

        add_mpiuse_arguments(subparsers)

        args = parser.parse_args(["mpiuse", "run_a", "--prefix", "cmp"])

        assert args.inputs == ["run_a"]
        assert args.prefix == "cmp"
        assert args.show is False
        assert args.subtypes is None
        assert args.steps is None
        assert args.per_rank is False
        assert args.format == "ascii"

    def test_add_mpiuse_arguments_all_options(self):
        """Test labels, steps, subtype filters, and flags."""
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()

        add_mpiuse_arguments(subparsers)

        args = parser.parse_args(
            [
                "mpiuse",
                "run_a",
                "run_b",
                "--labels",
                "master",
                "branch",
                "--steps",
                "10",
                "20",
                "--subtypes",
                "gpart",
                "rho",
                "--per-rank",
                "--format",
                "both",
                "--show",
            ]
        )

        assert args.inputs == ["run_a", "run_b"]
        assert args.labels == ["master", "branch"]
        assert args.steps == [10, 20]
        assert args.subtypes == ["gpart", "rho"]
        assert args.per_rank is True
        assert args.format == "both"
        assert args.show is True


class TestRunSwiftMpiuse:
    """Tests for the mpiuse CLI entry point."""

    @patch("swiftsim_cli.modes.analyse.mpiuse.analyse_swift_mpiuse")
    def test_run_swift_mpiuse(self, mock_analyse):
        """Test the CLI entry point forwards arguments correctly."""
        args = Mock()
        args.inputs = ["/tmp/run_a", "/tmp/run_b"]
        args.labels = ["a", "b"]
        args.output_path = Path("/output")
        args.prefix = "cmp"
        args.show = True
        args.subtypes = ["gpart"]
        args.steps = [1, 4]
        args.per_rank = True
        args.format = "both"

        run_swift_mpiuse(args)

        mock_analyse.assert_called_once_with(
            inputs=["/tmp/run_a", "/tmp/run_b"],
            labels=["a", "b"],
            output_path="/output",
            prefix="cmp",
            show_plot=True,
            subtype_filter=["gpart"],
            step_range=(1, 4),
            per_rank=True,
            output_format="both",
        )


class TestMpiuseHelpers:
    """Tests for mpiuse parsing and aggregation helpers."""

    def test_discover_files_directory_and_glob(self, tmp_path):
        """Test discovering files from a directory and glob pattern."""
        run_dir = _make_sample_input(tmp_path / "run")

        from_dir = _discover_files(run_dir)
        from_glob = _discover_files(
            str(run_dir / "mpiuse_report-rank*-step*.dat")
        )

        assert len(from_dir) == 3
        assert from_dir == from_glob

    def test_read_mpiuse_file_filters_activation_rows(self, tmp_path):
        """Test that only activation rows are retained."""
        path = _write_mpiuse_file(
            tmp_path,
            rank=0,
            step=7,
            rows=[
                "1 10 0 7 0 1 send 1 gpart 4 1 1 64 64",
                "2 11 5 7 0 1 send 1 gpart 4 0 1 -64 0",
                "3 12 0 7 0 1 recv 2 rho 5 1 2 32 32",
                "4 13 4 7 0 1 recv 2 rho 5 0 2 -32 0",
            ],
        )

        data = _read_mpiuse_file(path)

        assert list(data.columns) == [
            "step",
            "rank",
            "otherrank",
            "type",
            "subtype",
            "tag",
            "size",
            "dtic",
        ]
        assert len(data) == 2
        assert set(data["subtype"].tolist()) == {"gpart", "rho"}
        assert data["size"].tolist() == [64, 32]
        assert np.issubdtype(data["step"].dtype, np.integer)
        assert np.issubdtype(data["size"].dtype, np.integer)

    def test_aggregate_totals(self, tmp_path):
        """Test aggregation across ranks and steps."""
        run_dir = _make_sample_input(tmp_path / "run")
        aggregated = _aggregate(
            files=_discover_files(run_dir),
            step_range=None,
            subtype_filter=None,
            per_rank=True,
        )

        assert aggregated is not None
        assert aggregated["total_bytes"] == 390
        assert aggregated["total_count"] == 5
        assert aggregated["steps"] == [1, 2]
        assert aggregated["ranks"] == [0, 1]

        per_subtype = aggregated["per_subtype"].set_index(["type", "subtype"])
        assert int(per_subtype.loc[("send", "gpart"), "bytes"]) == 220
        assert int(per_subtype.loc[("recv", "rho"), "bytes"]) == 50
        assert int(per_subtype.loc[("send", "grav_counts"), "bytes"]) == 80
        assert int(per_subtype.loc[("recv", "xv"), "bytes"]) == 40

        per_step = aggregated["per_step_totals"].set_index("step")
        assert int(per_step.loc[1, "bytes"]) == 270
        assert int(per_step.loc[1, "count"]) == 3
        assert int(per_step.loc[2, "bytes"]) == 120
        assert int(per_step.loc[2, "count"]) == 2

        inflight = aggregated["inflight_per_step"].set_index("step")
        assert int(inflight.loc[1, "inflight_max"]) == 120
        assert int(inflight.loc[2, "inflight_max"]) == 80


class TestAnalyseSwiftMpiuse:
    """Tests for the core mpiuse analysis function."""

    def test_label_validation(self):
        """Test that labels must match the number of inputs."""
        import pytest

        with pytest.raises(ValueError, match="must match"):
            analyse_swift_mpiuse(
                inputs=["a", "b"],
                labels=["only-one"],
                show_plot=False,
            )

    def test_step_range_validation(self):
        """Test invalid step ranges are rejected."""
        import pytest

        with pytest.raises(ValueError, match="Step range"):
            analyse_swift_mpiuse(
                inputs=["a"],
                step_range=(5, 1),
                show_plot=False,
            )

    def test_analyse_swift_mpiuse_smoke(self, tmp_path):
        """Test end-to-end output generation for a single input."""
        run_dir = _make_sample_input(tmp_path / "run_a")

        analyse_swift_mpiuse(
            inputs=[str(run_dir)],
            labels=["a"],
            output_path=str(tmp_path),
            prefix="t",
            show_plot=False,
            per_rank=True,
            output_format="both",
        )

        out_dir = tmp_path / "t_mpiuse_analysis"
        assert (out_dir / "t_mpiuse_total_bytes_per_step.png").exists()
        assert (out_dir / "t_mpiuse_total_count_per_step.png").exists()
        assert (out_dir / "t_mpiuse_bytes_by_subtype.png").exists()
        assert (out_dir / "t_mpiuse_count_by_subtype.png").exists()
        assert (out_dir / "t_mpiuse_gpart_per_step.png").exists()
        assert (out_dir / "t_mpiuse_inflight_max.png").exists()
        assert (out_dir / "t_mpiuse_bytes_per_rank.png").exists()
        assert (out_dir / "t_mpiuse_summary_a.txt").exists()
        assert (out_dir / "t_mpiuse_per_rank_a.txt").exists()
        assert (out_dir / "t_mpiuse_per_step.csv").exists()

        csv_data = pd.read_csv(out_dir / "t_mpiuse_per_step.csv")
        assert set(csv_data["label"].tolist()) == {"a"}
        assert {1, 2} == set(csv_data["step"].tolist())

    def test_comparison_table_contains_labels_and_delta(
        self, tmp_path, capsys
    ):
        """Test comparison output includes both labels and delta columns."""
        run_a = _make_sample_input(tmp_path / "run_a", scale=1)
        run_b = _make_sample_input(tmp_path / "run_b", scale=2)

        analyse_swift_mpiuse(
            inputs=[str(run_a), str(run_b)],
            labels=["master", "branch"],
            output_path=str(tmp_path),
            prefix="cmp",
            show_plot=False,
        )

        captured = capsys.readouterr()
        assert "master bytes" in captured.out
        assert "branch bytes" in captured.out
        assert "Δ bytes (branch)" in captured.out

        comparison = (
            tmp_path / "cmp_mpiuse_analysis" / "cmp_mpiuse_comparison.txt"
        ).read_text(encoding="utf-8")
        assert "master bytes" in comparison
        assert "Δ % (branch)" in comparison

    def test_step_filter_excludes_everything(self, tmp_path, capsys):
        """Test clean exit when the step filter removes all data."""
        run_dir = _make_sample_input(tmp_path / "run_a")

        analyse_swift_mpiuse(
            inputs=[str(run_dir)],
            labels=["a"],
            output_path=str(tmp_path),
            prefix="empty",
            show_plot=False,
            step_range=(100, 200),
        )

        captured = capsys.readouterr()
        assert "no usable mpiuse data remained" in captured.out.lower()
        assert "No usable mpiuse data found in any input" in captured.out
