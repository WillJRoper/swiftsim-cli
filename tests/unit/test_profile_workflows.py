"""Profile/build workflow tests (update chains, edits, migration, ...).

These use real git repositories from the ``gitsetup`` fixture.
"""

import argparse
import shutil

import pytest

from swiftsim_cli import cli
from swiftsim_cli import profile as profile_mod
from swiftsim_cli.modes import config as config_mode
from swiftsim_cli.modes import make as make_mode
from swiftsim_cli.modes import profile as pm
from swiftsim_cli.modes import show as show_mode
from swiftsim_cli.modes import update as update_mode
from swiftsim_cli.multi_mode_args import MultiModeCLIArgs
from swiftsim_cli.profile import (
    ProfileError,
    get_profile,
    load_store,
    save_store,
)
from swiftsim_cli.swiftsim_dir import get_swiftsim_dir, update_profile

from .conftest import run_git
from .test_profile_worktrees import _answers, _args, _configure_runs, _confirm


def _main(argv):
    """Run the CLI, returning its exit code (0 if it did not exit)."""
    try:
        cli.main(argv)
    except SystemExit as e:
        return e.code
    return 0


def _conflicting_dirty_change(gitsetup, wt):
    """Leave an uncommitted change to src.c that upstream also changes."""
    (wt / "src.c").write_text("int x = 42;\n")
    gitsetup.push("master", "src.c", "int x = 7;\n")


class TestUpdateChain:
    """update --no-build and chains with config/make (P1, P14, P20)."""

    def test_no_build(self, gitsetup, capsys):
        """Update --no-build only pulls."""
        gitsetup.store()
        pm.new_profile("dev")
        gitsetup.push("master", "src.c", "int x = 2;\n")
        args = MultiModeCLIArgs(["update", "--no-build"]).modes[0][1]
        update_mode.run(args)
        wt = gitsetup.root / "dev"
        assert (wt / "src.c").read_text() == "int x = 2;\n"
        assert not (wt / "swift").exists()
        assert "Pulled 1 new commit(s) from origin/master" in (
            capsys.readouterr().out
        )

    def test_chain_builds_once(self, gitsetup, capsys):
        """Update before make/config leaves the build to them."""
        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        assert _main(["update", "config", "--enable-debug", "make"]) == 0
        captured = capsys.readouterr()
        assert "not building here: the later 'config'" in captured.err
        assert "Already up to date with origin/master" in captured.out
        assert captured.out.count("Building profile 'dev'") == 1
        assert "Configuring profile 'dev' with: --enable-debug" in (
            captured.out
        )
        assert _configure_runs(wt) == ["--enable-debug"]
        assert (wt / "swift").exists()

    def test_alone_still_builds(self, gitsetup, capsys):
        """Update on its own (or before other modes) builds."""
        gitsetup.store()
        pm.new_profile("dev")
        assert _main(["update", "-j", "2", "show"]) == 0
        assert "Building profile 'dev' with make -j 2" in (
            capsys.readouterr().out
        )
        assert (gitsetup.root / "dev" / "swift").exists()

    def test_bad_thread_count_before_pull(self, gitsetup, capsys):
        """-j 0 is rejected before anything is pulled."""
        gitsetup.store()
        pm.new_profile("dev")
        gitsetup.push("master", "src.c", "int x = 2;\n")
        assert _main(["profile", "--update", "-j", "0"]) == 2
        with pytest.raises(ValueError, match="at least 1"):
            update_profile(get_profile("dev"), nr_threads=0)
        assert (gitsetup.root / "dev" / "src.c").read_text() == "int x = 1;\n"


class TestUpdateSafety:
    """Autostash conflicts, missing upstreams and conflicted builds."""

    def test_autostash_conflict_is_an_error(self, gitsetup):
        """Re-applying the stash with conflicts stops before building (P2)."""
        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        _conflicting_dirty_change(gitsetup, wt)
        with pytest.raises(ProfileError, match="git stash drop") as e:
            update_profile(get_profile("dev"))
        assert "src.c" in str(e.value)
        assert not (wt / "swift").exists()
        assert _configure_runs(wt) == []
        # git kept the changes in the stash
        assert run_git("stash", "list", cwd=wt)

        # ... and a plain make refuses the conflicted tree (P19)
        with pytest.raises(ProfileError, match="unresolved conflicts"):
            make_mode.make_swift()

    def test_make_refuses_mid_rebase(self, gitsetup):
        """Make refuses a worktree stopped part-way through a rebase."""
        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        (wt / "src.c").write_text("int x = 42;\n")
        run_git("commit", "-qam", "local", cwd=wt)
        gitsetup.push("master", "src.c", "int x = 7;\n")
        with pytest.raises(ProfileError, match="rebase --continue"):
            update_profile(get_profile("dev"))
        with pytest.raises(ProfileError, match="rebase is in progress"):
            make_mode.make_swift()

    def test_no_upstream_checked_first(self, gitsetup):
        """A branch without upstream fails cleanly before pulling (P15)."""
        gitsetup.store()
        pm.new_profile("dev")
        run_git("branch", "--unset-upstream", cwd=gitsetup.root / "dev")
        with pytest.raises(ProfileError, match="has no upstream.*--set"):
            update_profile(get_profile("dev"))

    def test_failed_make_exit_code(self, gitsetup, capsys):
        """A failed build exits with make's own code (P21)."""
        gitsetup.store()
        pm.new_profile("dev")
        update_profile(get_profile("dev"))
        (gitsetup.root / "dev" / "Makefile").write_text("all:\n\tfalse\n")
        assert _main(["make"]) == 2
        assert "'make -j 1' failed (exit code 2)" in capsys.readouterr().err


class TestCosmology:
    """profile --show / show and the parameter file (P3, P22)."""

    def test_duplicate_key_parameter_file(self, gitsetup, capsys):
        """SWIFT's example file has a duplicate key: warn, don't crash."""
        gitsetup.store()
        pm.new_profile("dev")
        param = gitsetup.tmp / "dup.yml"
        param.write_text(
            "Cosmology:\n"
            "  h: 0.6777\n"
            "  Omega_cdm: 0.2587\n"
            "Hydrogen:\n"
            "  cmb_temperature_floor: 2.7\n"
            "  cmb_temperature_floor: 3.0\n"
        )
        profile_mod.update_profile_fields("dev", parameter_file=param)
        pm.run(_args(show=""))
        captured = capsys.readouterr()
        assert "PARAMETER FILE COSMOLOGY" in captured.out
        assert "Omega_cdm" in captured.out
        assert "duplicate" in captured.err

    def test_show_matches_profile_show(self, gitsetup, capsys):
        """Show NAME prints the profile and cosmology like profile --show."""
        gitsetup.store()
        pm.new_profile("a")
        pm.new_profile("b")
        assert _main(["show", "b"]) == 0
        out = capsys.readouterr().out
        assert "PROFILE: b" in out and "Omega_cdm" in out

    def test_cosmo_without_parameter_file(self, gitsetup, capsys):
        """Show --cosmo with a missing parameter file is an error."""
        gitsetup.store()
        pm.new_profile("dev")
        missing = gitsetup.tmp / "missing.yml"
        profile_mod.update_profile_fields("dev", parameter_file=missing)
        with pytest.raises(ProfileError, match="parameter file not found"):
            show_mode.run(argparse.Namespace(name=None, cosmo=True, all=False))
        pm.display_profile(name="dev")  # with the table it's just a note
        assert "no cosmology to show" in capsys.readouterr().err


class TestInitAndOverride:
    """Non-interactive init, clone changes and --profile (P4, P10, P18)."""

    def test_init_non_interactive(self, gitsetup, capsys):
        """--repo makes init prompt-free; --set fills the settings."""
        pm.run(
            _args(
                init=True,
                repo=str(gitsetup.repo),
                worktree_root=str(gitsetup.tmp / "wts"),
                name="zoom",
                upstream="origin/feature",
                configure_flags="--enable-debug",
                set=["softening_coeff=0.1", f"data_dir={gitsetup.tmp}"],
            )
        )
        store = load_store()
        p = store.profiles["zoom"]
        assert store.current == "zoom"
        assert p.worktree == (gitsetup.tmp / "wts" / "zoom").resolve()
        assert (p.worktree / "feature.c").exists()
        assert p.configure_flags == ["--enable-debug"]
        assert p.softening_coeff == 0.1
        assert p.data_dir == gitsetup.tmp.resolve()
        assert "swift-cli make -j" in capsys.readouterr().out

    def test_init_cannot_change_clone(self, gitsetup, tmp_path):
        """Re-running init can't re-point existing profiles' clone."""
        gitsetup.store()
        pm.new_profile("a")
        other = tmp_path / "other"
        run_git("clone", "-q", str(gitsetup.remote), str(other))
        with pytest.raises(ProfileError, match="can't change"):
            pm.init_profiles(repo=str(other), name="b", yes=True)
        assert load_store().repo == gitsetup.repo
        # The same clone is fine
        pm.init_profiles(repo=str(gitsetup.repo), name="b", yes=True)
        assert set(load_store().profiles) == {"a", "b"}

    def test_init_rejects_bad_answers(self, gitsetup, monkeypatch):
        """Prompt answers are checked as they are given (P11)."""
        _answers(monkeypatch, str(gitsetup.repo), None, "bad name")
        with pytest.raises(ProfileError, match="Invalid profile name"):
            pm.init_profiles()
        _answers(
            monkeypatch,
            str(gitsetup.repo),
            None,
            "ok",
            None,
            None,
            None,
            None,
            "soft",
        )
        with pytest.raises(ProfileError, match="must be a number"):
            pm.init_profiles()
        assert not profile_mod.PROFILE_FILE.exists()

    def test_global_profile_checked_up_front(self, gitsetup, capsys):
        """An unknown --profile fails before the first mode runs."""
        gitsetup.store()
        pm.new_profile("dev")
        assert _main(["--profile", "nope", "make"]) == 1
        assert "Unknown profile 'nope'" in capsys.readouterr().err
        assert not (gitsetup.root / "dev" / "swift").exists()

    def test_remove_uses_worktree_clone(self, gitsetup, monkeypatch):
        """Removal runs git in the worktree's own clone (P4)."""
        gitsetup.store()
        pm.new_profile("a")
        pm.new_profile("b")
        store = load_store()
        store.repo = gitsetup.tmp / "moved"
        save_store(store)
        _confirm(monkeypatch, True, True)
        pm.remove_profile("b")
        assert not (gitsetup.root / "b").exists()
        assert "swift-cli/b" not in run_git("branch", cwd=gitsetup.repo)


class TestEditUpstream:
    """--edit upstream changes and --set (P6, P10)."""

    def test_set_moves_clean_branch(self, gitsetup, capsys):
        """--set upstream=... moves a branch with no commits of its own."""
        gitsetup.store()
        pm.new_profile("dev")
        pm.run(
            _args(
                edit="dev",
                set=["upstream=origin/feature", "softening-pivot-z=3"],
            )
        )
        p = get_profile("dev")
        assert p.upstream == "origin/feature"
        assert p.softening_pivot_z == 3.0
        assert p.softening_coeff == 0.04  # untouched
        assert (gitsetup.root / "dev" / "feature.c").exists()
        assert "Moved branch swift-cli/dev" in capsys.readouterr().out

    def test_own_commits_kept_with_warning(self, gitsetup, capsys):
        """A branch with its own commits keeps them and gets a hint."""
        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        (wt / "mine.c").write_text("x\n")
        run_git("add", "mine.c", cwd=wt)
        run_git("commit", "-qm", "mine", cwd=wt)
        pm.edit_profile("dev", {"upstream": "origin/feature"})
        assert (wt / "mine.c").exists()
        assert not (wt / "feature.c").exists()
        err = capsys.readouterr().err
        assert "1 commit(s) of its own" in err
        assert "rebase --onto origin/feature origin/master" in err

    def test_dirty_tree_refused(self, gitsetup):
        """Moving a branch with uncommitted changes is refused."""
        gitsetup.store()
        pm.new_profile("dev")
        (gitsetup.root / "dev" / "src.c").write_text("dirty\n")
        with pytest.raises(ProfileError, match="uncommitted changes"):
            pm.edit_profile("dev", {"upstream": "origin/feature"})
        assert get_profile("dev").upstream == "origin/master"

    @pytest.mark.parametrize(
        "item, match",
        [
            ("colour=red", "KEY one of"),
            ("upstream", "KEY one of"),
            ("softening_coeff=x", "must be a number"),
            ("data_dir=/no/such/dir", "does not exist"),
            ("configure_flags='--a", "configure_flags"),
        ],
    )
    def test_bad_set(self, item, match):
        """Bad --set values are rejected before anything changes."""
        with pytest.raises(ProfileError, match=match):
            pm.parse_settings([item])


class TestRemoveAndAdopt:
    """Adopted worktrees, --yes and branch reuse (P5, P9, P10)."""

    def _handmade(self, gitsetup):
        hand = gitsetup.tmp / "handmade"
        run_git(
            "worktree", "add", "-q", "-b", "mine", str(hand), cwd=gitsetup.repo
        )
        return hand

    def test_adopt_twice_rejected(self, gitsetup):
        """A worktree can back only one profile."""
        gitsetup.store()
        hand = self._handmade(gitsetup)
        pm.adopt_worktree(hand, "one")
        with pytest.raises(ProfileError, match="already the worktree"):
            pm.adopt_worktree(hand, "two")

    def test_adopted_and_main_clone_only_unregistered(self, gitsetup):
        """Removing adopted profiles never deletes their directories."""
        gitsetup.store()
        pm.new_profile("keep")
        hand = self._handmade(gitsetup)
        pm.adopt_worktree(hand, "hand")
        pm.adopt_worktree(gitsetup.repo, "main")
        pm.run(_args(remove="hand", yes=True))
        pm.run(_args(remove="main", yes=True))
        assert set(load_store().profiles) == {"keep"}
        assert hand.exists() and (gitsetup.repo / "src.c").exists()
        assert "mine" in run_git("branch", cwd=gitsetup.repo)

    def test_yes_keeps_branch(self, gitsetup, capsys):
        """--yes removes without asking but keeps the branch."""
        gitsetup.store()
        pm.new_profile("a")
        pm.new_profile("b")
        pm.run(_args(remove="b", yes=True))
        assert not (gitsetup.root / "b").exists()
        assert "swift-cli/b" in run_git("branch", cwd=gitsetup.repo)
        assert "kept branch 'swift-cli/b'" in capsys.readouterr().err

    def test_reuse_branch(self, gitsetup):
        """A kept branch blocks the name until reused explicitly."""
        gitsetup.store()
        pm.new_profile("a")
        pm.new_profile("b")
        wt = gitsetup.root / "b"
        (wt / "mine.c").write_text("x\n")
        run_git("add", "mine.c", cwd=wt)
        run_git("commit", "-qm", "mine", cwd=wt)
        pm.remove_profile("b", yes=True)
        with pytest.raises(ProfileError, match="--reuse-branch"):
            pm.new_profile("b")
        pm.run(_args(new="b", reuse_branch=True, upstream="origin/feature"))
        assert (wt / "mine.c").exists()
        assert (
            run_git("rev-parse", "--abbrev-ref", "swift-cli/b@{u}", cwd=wt)
            == "origin/feature"
        )


class TestMigrateSafety:
    """Migration edge cases (P7, P8, P27)."""

    def _legacy(self, gitsetup, text):
        profile_mod.PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
        profile_mod.PROFILE_FILE.write_text(text)

    def test_all_failed_changes_nothing(self, gitsetup):
        """If nothing migrates, the old file and timer files stay."""
        self._legacy(
            gitsetup,
            f"Default:\n  swiftsim_dir: {gitsetup.repo}\n  branch: gone\n",
        )
        timers = profile_mod.PROFILE_FILE.parent / "timers.yaml"
        timers.write_text("x")
        old = profile_mod.PROFILE_FILE.read_text()
        with pytest.raises(ProfileError, match="nothing was changed"):
            pm.migrate_profiles(yes=True)
        assert profile_mod.PROFILE_FILE.read_text() == old
        assert timers.exists()
        assert not profile_mod.PROFILE_FILE.with_suffix(".yaml.bak").exists()

    def test_current_dropped_keys_root_and_rerun(self, gitsetup, capsys):
        """Unmatched Current becomes 'current'; dropped keys are listed."""
        run_git("branch", "local-only", cwd=gitsetup.repo)
        self._legacy(
            gitsetup,
            f"Current:\n  swiftsim_dir: {gitsetup.repo}\n"
            f"  branch: local-only\n"
            f"Default:\n  swiftsim_dir: {gitsetup.repo}\n  branch: master\n"
            f"  h: 0.7\n  Omega_m: 0.3\n",
        )
        (profile_mod.PROFILE_FILE.parent / "notes.txt").write_text("x")
        root = gitsetup.tmp / "elsewhere"
        # A worktree left by an interrupted earlier attempt is reused
        run_git(
            "worktree",
            "add",
            "-q",
            "--track",
            "-b",
            "swift-cli/default",
            str(root / "default"),
            "origin/master",
            cwd=gitsetup.repo,
        )
        pm.run(_args(migrate=True, yes=True, worktree_root=str(root)))
        captured = capsys.readouterr()
        assert "becomes the profile 'current'" in captured.err
        assert "default: Omega_m, h" in captured.err
        assert "notes.txt" in captured.err
        assert "Reusing existing worktree" in captured.out
        store = load_store()
        assert set(store.profiles) == {"default", "current"}
        assert store.current == "current"
        assert store.profiles["current"].upstream == "local-only"
        assert store.worktree_root == root.resolve()


class TestMessages:
    """Error text (P12, P13)."""

    def test_missing_worktree_hint(self, gitsetup):
        """The hint names commands that can actually fix the worktree."""
        gitsetup.store()
        pm.new_profile("dev")
        shutil.rmtree(gitsetup.root / "dev")
        with pytest.raises(FileNotFoundError) as e:
            get_swiftsim_dir()
        assert "--adopt" in str(e.value) and "--edit" not in str(e.value)

    def test_config_value_without_equals(self, gitsetup):
        """Config --with-hydro sphenix suggests --with-hydro=sphenix."""
        gitsetup.store()
        pm.new_profile("dev")
        args = MultiModeCLIArgs(["config", "--with-hydro", "sphenix"])
        with pytest.raises(
            ProfileError, match="did you mean --with-hydro=sphenix"
        ):
            config_mode.run(args.modes[0][1])
        assert _configure_runs(gitsetup.root / "dev") == []
