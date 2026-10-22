"""Integration tests for worktree-based profiles using real git repos."""

import argparse
import sys

import pytest

from swiftsim_cli import profile as profile_mod
from swiftsim_cli.modes import config as config_mode
from swiftsim_cli.modes import make as make_mode
from swiftsim_cli.modes import profile as pm
from swiftsim_cli.modes import show as show_mode
from swiftsim_cli.modes import switch as switch_mode
from swiftsim_cli.modes import update as update_mode
from swiftsim_cli.multi_mode_args import MultiModeCLIArgs
from swiftsim_cli.profile import (
    ProfileError,
    get_profile,
    load_store,
    set_profile_override,
)
from swiftsim_cli.swiftsim_dir import (
    configure_profile,
    get_swiftsim_dir,
    update_profile,
)

from .conftest import run_git


def _editor(monkeypatch, tmp_path, **changes):
    """Point $EDITOR at a script that applies ``changes`` to the file."""
    script = tmp_path / "fake_editor.py"
    script.write_text(
        "import sys\n"
        "from ruamel.yaml import YAML\n"
        "yaml = YAML(typ='safe')\n"
        "data = yaml.load(open(sys.argv[1]))\n"
        f"data.update({changes!r})\n"
        "yaml.dump(data, open(sys.argv[1], 'w'))\n"
    )
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", f"{sys.executable} {script}")


def _confirm(monkeypatch, *replies):
    """Script the yes/no prompts."""
    queue = list(replies)
    monkeypatch.setattr(pm, "confirm", lambda question: queue.pop(0))
    return queue


def _upstream(worktree, branch):
    return run_git(
        "rev-parse", "--abbrev-ref", f"{branch}@{{u}}", cwd=worktree
    )


def _configure_runs(worktree):
    log = worktree / "configure.log"
    return log.read_text().splitlines() if log.exists() else []


def _args(**kwargs):
    """Parse profile-mode arguments, overriding with kwargs."""
    parser = argparse.ArgumentParser()
    pm.add_arguments(parser)
    args = parser.parse_args([])
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


class TestNew:
    """profile --new."""

    def test_creates_worktree_and_tracking_branch(self, gitsetup, capsys):
        """A new profile gets its own worktree and tracking branch."""
        gitsetup.store()
        pm.run(_args(new="dev"))
        p = get_profile("dev")
        assert p.worktree is not None
        assert p.worktree == gitsetup.root / "dev"
        assert (p.worktree / "src.c").exists()
        assert p.branch == "swift-cli/dev"
        assert run_git("branch", "--show-current", cwd=p.worktree) == p.branch
        assert _upstream(p.worktree, p.branch) == "origin/master"
        assert p.parameter_file == (
            p.worktree / "examples" / "parameter_example.yml"
        )
        assert p.configured_flags is None
        assert load_store().current == "dev"  # first profile becomes current
        assert not p.is_built

    def test_other_upstream_inherits_settings(self, gitsetup):
        """New profiles follow --upstream and inherit user settings."""
        gitsetup.store()
        pm.new_profile("a")
        profile_mod.update_profile_fields(
            "a", data_dir=gitsetup.tmp, softening_coeff=0.1
        )
        pm.run(
            _args(
                new="feat",
                upstream="origin/feature",
                configure_flags="--enable-debug --with-hydro=sphenix",
            )
        )
        p = get_profile("feat")
        assert p.worktree is not None
        assert (p.worktree / "feature.c").exists()
        assert _upstream(p.worktree, p.branch) == "origin/feature"
        assert p.configure_flags == ["--enable-debug", "--with-hydro=sphenix"]
        assert p.data_dir == gitsetup.tmp
        assert p.softening_coeff == 0.1
        assert load_store().current == "a"

    def test_errors(self, gitsetup):
        """Bad names, duplicates and git failures give clean errors."""
        gitsetup.store()
        pm.new_profile("dev")
        with pytest.raises(ProfileError, match="already exists"):
            pm.new_profile("dev")
        with pytest.raises(ProfileError, match="Invalid profile name"):
            pm.new_profile("a/b")
        with pytest.raises(ProfileError, match="origin/nope"):
            pm.new_profile("x", upstream="origin/nope")
        (gitsetup.root / "taken").mkdir()
        with pytest.raises(ProfileError, match="already exists"):
            pm.new_profile("taken")
        run_git("branch", "swift-cli/clash", cwd=gitsetup.repo)
        with pytest.raises(ProfileError, match="swift-cli/clash"):
            pm.new_profile("clash")
        assert list(load_store().profiles) == ["dev"]

    def test_build(self, gitsetup):
        """--build runs autogen, configure with the flags, and make."""
        gitsetup.store()
        pm.run(_args(new="dev", configure_flags="--enable-debug", build=True))
        p = get_profile("dev")
        assert p.is_built
        assert _configure_runs(p.worktree) == ["--enable-debug"]
        assert p.configured_flags == ["--enable-debug"]


class TestSwitchListShow:
    """profile --switch / --list / --show and the switch/show modes."""

    def test_switch_list_show(self, gitsetup, capsys):
        """Switching only moves the pointer; list/show report it."""
        gitsetup.store()
        pm.new_profile("a")
        pm.new_profile("b", upstream="origin/feature")
        a_head = run_git("rev-parse", "HEAD", cwd=gitsetup.root / "a")

        switch_mode.run(argparse.Namespace(name="b"))
        assert load_store().current == "b"
        assert run_git("rev-parse", "HEAD", cwd=gitsetup.root / "a") == a_head
        capsys.readouterr()

        pm.run(_args())  # default action lists
        out = capsys.readouterr().out
        assert "* b" in out and "  a" in out
        assert "origin/feature" in out

        before = profile_mod.PROFILE_FILE.read_bytes()
        pm.run(_args(show=""))
        out = capsys.readouterr().out
        assert "PROFILE: b" in out
        assert "swift-cli/b" in out
        assert "(never)" in out
        pm.run(_args(show="a"))
        assert "PROFILE: a" in capsys.readouterr().out
        assert profile_mod.PROFILE_FILE.read_bytes() == before

    def test_switch_unknown(self, gitsetup):
        """Unknown profiles suggest close matches."""
        gitsetup.store()
        pm.new_profile("dev")
        with pytest.raises(ProfileError, match="Did you mean: dev"):
            pm.switch_profile("dve")

    def test_list_empty(self, gitsetup, capsys):
        """An empty store says so."""
        gitsetup.store()
        pm.list_profiles()
        assert "No profiles found" in capsys.readouterr().out

    def test_show_mode_cosmology(self, gitsetup, capsys):
        """The show mode can print the parameter file's cosmology."""
        gitsetup.store()
        pm.new_profile("dev")
        show_mode.run(argparse.Namespace(name=None, cosmo=True, all=False))
        out = capsys.readouterr().out
        assert "PARAMETER FILE COSMOLOGY" in out
        assert "Omega_cdm" in out
        assert "PROFILE:" not in out
        show_mode.run(argparse.Namespace(name=None, cosmo=False))
        out = capsys.readouterr().out
        assert "PROFILE: dev" in out and "Omega_cdm" in out


class TestEdit:
    """profile --edit."""

    def test_edit_upstream_and_flags(
        self, gitsetup, monkeypatch, tmp_path, capsys
    ):
        """Upstream changes move the branch; flags are just stored."""
        gitsetup.store()
        pm.new_profile("dev")
        _editor(
            monkeypatch,
            tmp_path,
            upstream="origin/feature",
            configure_flags="--enable-ipo --with-hydro=sphenix",
            data_dir=str(gitsetup.tmp),
            softening_coeff=0.05,
        )
        pm.run(_args(edit="dev"))
        p = get_profile("dev")
        assert p.upstream == "origin/feature"
        assert _upstream(p.worktree, p.branch) == "origin/feature"
        # No commits of its own, so the branch moved to the new upstream
        assert (gitsetup.root / "dev" / "feature.c").exists()
        assert p.configure_flags == ["--enable-ipo", "--with-hydro=sphenix"]
        assert p.configured_flags is None
        assert p.data_dir == gitsetup.tmp.resolve()
        assert p.softening_coeff == 0.05
        assert p.softening_pivot_z == 2.7
        assert "reconfigure" in capsys.readouterr().out

    def test_edit_unchanged(self, gitsetup, monkeypatch, tmp_path, capsys):
        """Quitting without changes saves nothing."""
        gitsetup.store()
        pm.new_profile("dev")
        _editor(monkeypatch, tmp_path)
        before = profile_mod.PROFILE_FILE.read_bytes()
        pm.edit_profile("dev")
        assert profile_mod.PROFILE_FILE.read_bytes() == before
        assert "unchanged" in capsys.readouterr().out

    def test_edit_bad_values(self, gitsetup, monkeypatch, tmp_path):
        """Bad edits are clean errors and nothing is saved."""
        gitsetup.store()
        pm.new_profile("dev")
        for changes, match in (
            ({"upstream": "origin/nope"}, "origin/nope"),
            ({"softening_coeff": "soft"}, "must be a number"),
            ({"colour": "red"}, "may only contain"),
        ):
            _editor(monkeypatch, tmp_path, **changes)
            with pytest.raises(ProfileError, match=match):
                pm.edit_profile("dev")
        assert get_profile("dev").upstream == "origin/master"


class TestRemove:
    """profile --remove."""

    def test_remove(self, gitsetup, monkeypatch):
        """Removing deletes the worktree, then the branch if confirmed."""
        gitsetup.store()
        pm.new_profile("a")
        pm.new_profile("b")
        pm.new_profile("c")
        with pytest.raises(ProfileError, match="current profile"):
            pm.remove_profile("a")

        _confirm(monkeypatch, True, True)
        pm.run(_args(remove="b"))
        assert not (gitsetup.root / "b").exists()
        assert "b" not in load_store().profiles
        assert "swift-cli/b" not in run_git("branch", cwd=gitsetup.repo)

        _confirm(monkeypatch, True, False)
        pm.remove_profile("c")
        assert "c" not in load_store().profiles
        assert "swift-cli/c" in run_git("branch", cwd=gitsetup.repo)

    def test_remove_declined_and_dirty(self, gitsetup, monkeypatch):
        """Declining does nothing; git refuses to remove dirty trees."""
        gitsetup.store()
        pm.new_profile("a")
        pm.new_profile("b")
        _confirm(monkeypatch, False)
        pm.remove_profile("b")
        assert "b" in load_store().profiles

        (gitsetup.root / "b" / "untracked.c").write_text("x")
        _confirm(monkeypatch, True)
        with pytest.raises(ProfileError, match="--force"):
            pm.remove_profile("b")
        assert "b" in load_store().profiles
        assert (gitsetup.root / "b").exists()

        _confirm(monkeypatch, True, False)
        pm.run(_args(remove="b", force=True))
        assert "b" not in load_store().profiles
        assert not (gitsetup.root / "b").exists()

    def test_remove_missing_worktree(self, gitsetup, monkeypatch):
        """A worktree deleted by hand is pruned."""
        import shutil

        gitsetup.store()
        pm.new_profile("a")
        pm.new_profile("b")
        shutil.rmtree(gitsetup.root / "b")
        _confirm(monkeypatch, True, True)
        pm.remove_profile("b")
        assert "b" not in load_store().profiles
        assert "swift-cli/b" not in run_git("branch", cwd=gitsetup.repo)


class TestUpdate:
    """profile --update / --update-all and the update/make modes."""

    def test_reconfigure_only_when_flags_change(self, gitsetup):
        """Configure runs first time and when flags change, not otherwise."""
        gitsetup.store()
        pm.new_profile("dev", configure_flags=["--enable-debug"])
        wt = gitsetup.root / "dev"

        pm.run(_args(update="", nr_threads=2))
        assert (wt / "swift").exists()
        assert _configure_runs(wt) == ["--enable-debug"]

        gitsetup.push("master", "src.c", "int x = 2;\n")
        update_mode.run(argparse.Namespace(nr_threads=1, no_build=False))
        assert (wt / "src.c").read_text() == "int x = 2;\n"
        assert len(_configure_runs(wt)) == 1

        profile_mod.update_profile_fields(
            "dev", configure_flags=["--with-tbbmalloc"]
        )
        make_mode.run(argparse.Namespace(nr_threads=1, clean=True))
        assert _configure_runs(wt) == ["--enable-debug", "--with-tbbmalloc"]
        assert get_profile("dev").configured_flags == ["--with-tbbmalloc"]

    def test_local_commits_rebased(self, gitsetup):
        """Local commits are replayed on top of upstream."""
        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        (wt / "feature.c").write_text("local\n")
        run_git("add", "feature.c", cwd=wt)
        run_git("commit", "-qm", "local work", cwd=wt)
        gitsetup.push("master", "src.c", "int x = 3;\n")
        update_profile(get_profile("dev"))
        log = run_git("log", "--format=%s", cwd=wt).splitlines()
        assert log[:2] == ["local work", "change src.c"]

    def test_rebase_conflict(self, gitsetup):
        """Conflicts give instructions and nothing is built."""
        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        (wt / "src.c").write_text("int x = 42;\n")
        run_git("commit", "-qam", "local", cwd=wt)
        gitsetup.push("master", "src.c", "int x = 7;\n")
        with pytest.raises(ProfileError, match="git rebase --continue") as e:
            pm.run(_args(update="dev"))
        assert f"cd {wt}" in str(e.value)
        assert not (wt / "swift").exists()
        assert _configure_runs(wt) == []

    def test_dirty_tree_is_autostashed(self, gitsetup, capsys):
        """Uncommitted changes are stashed, rebased over and re-applied."""
        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        (wt / "feature.c").write_text("tracked\n")
        run_git("add", "feature.c", cwd=wt)
        run_git("commit", "-qm", "local work", cwd=wt)
        (wt / "feature.c").write_text("dirty\n")
        gitsetup.push("master", "src.c", "int x = 5;\n")
        update_profile(get_profile("dev"))
        assert "will be stashed" in capsys.readouterr().out
        assert (wt / "src.c").read_text() == "int x = 5;\n"
        assert (wt / "feature.c").read_text() == "dirty\n"

    def test_pull_failure_surfaces_git_error(self, gitsetup):
        """Non-conflict pull failures surface git's message as is."""
        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        run_git("branch", "--unset-upstream", cwd=wt)
        with pytest.raises(ProfileError) as e:
            update_profile(get_profile("dev"))
        assert "rebase --continue" not in str(e.value)

    def test_update_all_summary(self, gitsetup, capsys):
        """--update-all continues past failures and summarises."""
        gitsetup.store()
        pm.new_profile("good")
        pm.new_profile("bad")
        wt = gitsetup.root / "bad"
        (wt / "src.c").write_text("int x = 42;\n")
        run_git("commit", "-qam", "local", cwd=wt)
        gitsetup.push("master", "src.c", "int x = 7;\n")
        with pytest.raises(ProfileError, match="1 profile.*bad"):
            pm.run(_args(update_all=True))
        out = capsys.readouterr().out
        assert "UPDATE" in out
        assert "| good" in out and "ok" in out
        assert "FAILED" in out
        assert (gitsetup.root / "good" / "swift").exists()

    def test_build_failure_is_clean(self, gitsetup):
        """A failing build command raises ProfileError."""
        gitsetup.store()
        pm.new_profile("dev")
        update_profile(get_profile("dev"))
        (gitsetup.root / "dev" / "Makefile").write_text("all:\n\tfalse\n")
        with pytest.raises(ProfileError, match="make -j 1.*failed"):
            make_mode.make_swift()
        with pytest.raises(ValueError, match="at least 1"):
            make_mode.make_swift(nr_threads=0)


class TestConfig:
    """Config mode and flag validation."""

    def test_validation_and_suggestions(self, gitsetup):
        """Unknown flags are rejected with suggestions before configure."""
        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        with pytest.raises(ProfileError) as e:
            configure_profile(
                get_profile("dev"), ["--enable-debg", "--with-hydra=x"]
            )
        msg = str(e.value)
        assert "--enable-debg (did you mean --enable-debug" in msg
        assert "--with-hydro" in msg
        assert _configure_runs(wt) == []

        cache = profile_mod.PROFILE_FILE.parent / "cache" / "dev"
        assert (cache / "configure_options.json").exists()

        flags = [
            "--enable-debug=yes",
            "--disable-optimization",
            "--without-tbbmalloc",
            "--with-hydro=sphenix",
            "CC=gcc",
        ]
        configure_profile(get_profile("dev"), flags)
        assert _configure_runs(wt) == [" ".join(flags)]
        assert get_profile("dev").configured_flags == flags

    def test_cache_invalidated_by_configure_mtime(self, gitsetup):
        """A newer ./configure reparses the option list."""
        import json
        import os

        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        configure_profile(get_profile("dev"), ["--enable-debug"])
        cache = (
            profile_mod.PROFILE_FILE.parent
            / "cache"
            / "dev"
            / "configure_options.json"
        )
        data = json.loads(cache.read_text())
        data["options"] = ["--only-this"]
        cache.write_text(json.dumps(data))
        with pytest.raises(ProfileError):
            configure_profile(get_profile("dev"), ["--enable-debug"])
        st = (wt / "configure").stat()
        os.utime(wt / "configure", (st.st_atime, st.st_mtime + 10))
        configure_profile(get_profile("dev"), ["--enable-debug"])

    def test_config_mode(self, gitsetup, capsys):
        """Presets and forwarded flags become the profile's flags."""
        gitsetup.store()
        pm.new_profile("dev")
        wt = gitsetup.root / "dev"
        args = MultiModeCLIArgs(
            ["config", "--debug-build", "--with-hydro=sphenix"]
        ).modes[0][1]
        config_mode.run(args)
        expected = [
            "--enable-debug",
            "--enable-debugging-checks",
            "--disable-optimization",
            "--with-hydro=sphenix",
        ]
        assert get_profile("dev").configure_flags == expected
        assert _configure_runs(wt) == [" ".join(expected)]

        # No flags: reconfigure with the stored ones
        config_mode.run(MultiModeCLIArgs(["config"]).modes[0][1])
        assert _configure_runs(wt)[-1] == " ".join(expected)

        argv = ["config", "--list-options"]
        config_mode.run(MultiModeCLIArgs(argv).modes[0][1])

        with pytest.raises(ProfileError, match="did you mean"):
            config_mode.run(
                MultiModeCLIArgs(["config", "--enable-ip"]).modes[0][1]
            )


class TestOverride:
    """The global --profile option."""

    def test_profile_override(self, gitsetup):
        """--profile NAME redirects modes without switching."""
        gitsetup.store()
        pm.new_profile("a")
        pm.new_profile("b")
        assert get_swiftsim_dir() == gitsetup.root / "a"
        parsed = MultiModeCLIArgs(["--profile", "b", "make", "-j", "2"])
        assert parsed.modes[0][1].nr_threads == 2
        # Parsing has no side effects; cli.main applies the override
        assert get_swiftsim_dir() == gitsetup.root / "a"
        set_profile_override(parsed.global_args.profile)
        assert get_swiftsim_dir() == gitsetup.root / "b"
        make_mode.run(parsed.modes[0][1])
        assert (gitsetup.root / "b" / "swift").exists()
        assert not (gitsetup.root / "a" / "swift").exists()
        assert load_store().current == "a"
        set_profile_override(MultiModeCLIArgs(["make"]).global_args.profile)
        assert get_swiftsim_dir() == gitsetup.root / "a"

    def test_unknown_override(self, gitsetup):
        """An unknown --profile is a clean error."""
        gitsetup.store()
        pm.new_profile("a")
        # Fails as soon as the override is set, before any mode runs
        with pytest.raises(ProfileError, match="Unknown profile 'zzz'"):
            set_profile_override(
                MultiModeCLIArgs(
                    ["--profile", "zzz", "make"]
                ).global_args.profile
            )
        assert get_swiftsim_dir() == gitsetup.root / "a"


class TestAdopt:
    """profile --adopt."""

    def test_adopt(self, gitsetup, capsys):
        """An existing hand-made worktree becomes a profile."""
        gitsetup.store()
        hand = gitsetup.tmp / "handmade"
        run_git(
            "worktree",
            "add",
            "-q",
            "--track",
            "-b",
            "my-branch",
            str(hand),
            "origin/feature",
            cwd=gitsetup.repo,
        )
        run_git("checkout", "-q", "--", ".", cwd=hand)
        import subprocess

        subprocess.run(["./autogen.sh"], cwd=hand, check=True)
        subprocess.run(
            ["./configure", "--enable-debug", "--with-hydro=sphenix"],
            cwd=hand,
            check=True,
        )
        pm.run(_args(adopt=str(hand), name="hand"))
        store = load_store()
        p = store.profiles["hand"]
        assert p.worktree == hand.resolve()
        assert p.branch == "my-branch"
        assert p.upstream == "origin/feature"
        assert p.configured_flags == ["--enable-debug", "--with-hydro=sphenix"]
        assert p.configure_flags == p.configured_flags
        assert store.current == "hand"

        # Already configured with those flags: building does not reconfigure
        make_mode.make_swift()
        assert _configure_runs(hand) == ["--enable-debug --with-hydro=sphenix"]

    def test_adopt_errors(self, gitsetup):
        """Non-worktrees, detached HEADs and missing --name are rejected."""
        gitsetup.store()
        with pytest.raises(ProfileError, match="requires --name"):
            pm.run(_args(adopt=str(gitsetup.tmp)))
        with pytest.raises(ProfileError, match="not a worktree"):
            pm.adopt_worktree(gitsetup.tmp, "x")
        det = gitsetup.tmp / "det"
        run_git(
            "worktree", "add", "-q", "--detach", str(det), cwd=gitsetup.repo
        )
        with pytest.raises(ProfileError, match="detached"):
            pm.adopt_worktree(det, "det")

    def test_adopt_main_clone_without_upstream(self, gitsetup):
        """The main clone can be adopted; branches without upstream work."""
        gitsetup.store()
        run_git("checkout", "-q", "-b", "solo", cwd=gitsetup.repo)
        pm.adopt_worktree(gitsetup.repo, "main")
        p = get_profile("main")
        assert p.branch == "solo"
        assert p.upstream is None
        assert p.configured_flags is None
        with pytest.raises(ProfileError, match="already exists"):
            pm.adopt_worktree(gitsetup.repo, "main")
