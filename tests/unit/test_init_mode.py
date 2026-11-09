"""Tests for 'swift-cli init' (clone or use SWIFT, first profile)."""

import shutil

import pytest

from swiftsim_cli import profile as profile_mod
from swiftsim_cli.modes import init as init_mode
from swiftsim_cli.multi_mode_args import MultiModeCLIArgs
from swiftsim_cli.profile import ProfileError, load_store

from .conftest import run_git
from .test_profile_workflows import _main


def _run(argv):
    """Run init mode directly so errors raise instead of exiting."""
    ((_, args),) = MultiModeCLIArgs(["init", *argv]).modes
    init_mode.run(args)


def test_init_clones_swift(gitsetup, tmp_path, monkeypatch, capsys):
    """Without --swift-path SWIFT is cloned into ./swiftsim."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert _main(["init", "--swift-url", str(gitsetup.remote)]) == 0
    store = load_store()
    assert store.repo == (work / "swiftsim").resolve()
    assert store.worktree_root == profile_mod.default_worktree_root()
    assert store.current == "default"
    p = store.profiles["default"]
    assert p.worktree is not None and (p.worktree / "src.c").exists()
    assert p.data_dir == work.resolve()
    assert "swift-cli make -j" in capsys.readouterr().out


def test_init_uses_swift_path(gitsetup):
    """--swift-path uses an existing clone; the options set the profile."""
    _run(
        [
            "--swift-path",
            str(gitsetup.repo),
            "--worktree-root",
            str(gitsetup.tmp / "wts"),
            "--name",
            "zoom",
            "--upstream",
            "origin/feature",
            "--configure-flags=--enable-debug",
        ]
    )
    store = load_store()
    p = store.profiles["zoom"]
    assert store.repo == gitsetup.repo
    assert store.current == "zoom"
    assert p.worktree == (gitsetup.tmp / "wts" / "zoom").resolve()
    assert p.worktree is not None and (p.worktree / "feature.c").exists()
    assert p.configure_flags == ["--enable-debug"]


def test_init_only_once(gitsetup):
    """Once profiles exist, init points at profile --new."""
    _run(["--swift-path", str(gitsetup.repo)])
    with pytest.raises(ProfileError, match="profile --new"):
        _run(["--swift-path", str(gitsetup.repo), "--name", "b"])
    assert list(load_store().profiles) == ["default"]


def test_init_refusals(tmp_path, monkeypatch):
    """Bad names, bad paths and an existing ./swiftsim never clone."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ProfileError, match="Invalid profile name"):
        _run(["--name", "bad name"])
    with pytest.raises(ProfileError, match="not found"):
        _run(["--swift-path", str(tmp_path / "nope")])
    (tmp_path / "plain").mkdir()
    with pytest.raises(ProfileError, match="Not a git repository"):
        _run(["--swift-path", str(tmp_path / "plain")])
    (tmp_path / "swiftsim").mkdir()
    with pytest.raises(ProfileError, match="--swift-path"):
        _run([])


def test_init_resolves_the_main_clone(gitsetup):
    """A subdirectory or another worktree of the clone means the clone."""
    (gitsetup.repo / "src").mkdir()
    _run(["--swift-path", str(gitsetup.repo / "src")])
    assert load_store().repo == gitsetup.repo
    profile_mod.PROFILE_FILE.unlink()
    other = gitsetup.tmp / "other-worktree"
    run_git(
        "worktree", "add", "-q", str(other), "-b", "side", cwd=gitsetup.repo
    )
    _run(["--swift-path", str(other), "--name", "second"])
    assert load_store().repo == gitsetup.repo


def test_init_after_starting_over(gitsetup):
    """Deleting ~/.swiftsim-cli (worktrees and all) and re-running works.

    The clone still has the profile's branch and the deleted worktree
    registered; init picks the branch up again.
    """
    argv = ["--swift-path", str(gitsetup.repo)]
    _run(argv)
    shutil.rmtree(profile_mod.PROFILE_FILE.parent)
    _run(argv)
    p = load_store().profiles["default"]
    assert p.worktree is not None and (p.worktree / "src.c").exists()
