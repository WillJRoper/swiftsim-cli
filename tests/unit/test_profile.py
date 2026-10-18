"""Unit tests for profile storage, resolution and configure parsing."""

import pytest

from swiftsim_cli import profile as profile_mod
from swiftsim_cli.profile import (
    ProfileError,
    ProfileStore,
    SWIFTCLIProfile,
    get_profile,
    get_profile_cache_dir,
    git,
    load_store,
    load_swift_profile,
    save_store,
    set_profile_override,
    update_profile_fields,
    validate_profile_name,
)
from swiftsim_cli.swiftsim_dir import parse_configure_options

from .conftest import CONFIGURE_HELP


def _store(tmp_path, current="a"):
    store = ProfileStore(
        repo=tmp_path / "repo",
        worktree_root=tmp_path / "wt",
        current=current,
        profiles={
            "a": SWIFTCLIProfile(
                name="a",
                worktree=tmp_path / "wt" / "a",
                branch="swift-cli/a",
                upstream="origin/master",
                configure_flags=["--enable-debug"],
            ),
            "b": SWIFTCLIProfile(name="b", worktree=tmp_path / "wt" / "b"),
        },
    )
    save_store(store)
    return store


class TestStore:
    """Reading and writing the profile file."""

    def test_round_trip(self, tmp_path):
        """A saved store loads back identically."""
        store = _store(tmp_path)
        assert load_store() == store
        text = profile_mod.PROFILE_FILE.read_text()
        assert "version: 2" in text
        assert "configured_flags:" in text

    def test_atomic_save_leaves_no_temp_files(self, tmp_path):
        """Only the profile file remains after saving."""
        _store(tmp_path)
        assert [p.name for p in profile_mod.PROFILE_FILE.parent.iterdir()] == [
            "profiles.yaml"
        ]

    def test_load_returns_copy_and_cache_cleared_on_save(self, tmp_path):
        """Mutating a loaded store does not leak; saves are visible."""
        _store(tmp_path)
        store = load_store()
        store.profiles["a"].upstream = "mutated"
        assert load_store().profiles["a"].upstream == "origin/master"
        update_profile_fields("a", upstream="origin/feature")
        assert get_profile("a").upstream == "origin/feature"

    def test_missing_and_empty_file(self):
        """A missing or empty file asks the user to run --init."""
        with pytest.raises(ProfileError, match="--init"):
            load_store()
        profile_mod.PROFILE_FILE.parent.mkdir(parents=True)
        profile_mod.PROFILE_FILE.write_text("")
        with pytest.raises(ProfileError, match="--init"):
            load_store()

    def test_legacy_file(self):
        """An old-format file asks the user to run --migrate."""
        profile_mod.PROFILE_FILE.parent.mkdir(parents=True)
        profile_mod.PROFILE_FILE.write_text(
            "Current:\n  swiftsim_dir: /x\nDefault:\n  swiftsim_dir: /x\n"
        )
        with pytest.raises(ProfileError, match="--migrate"):
            load_store()
        with pytest.raises(ProfileError, match="--migrate"):
            load_swift_profile()

    def test_bad_version_and_malformed(self):
        """Unknown versions and non-mapping files are rejected."""
        profile_mod.PROFILE_FILE.parent.mkdir(parents=True)
        profile_mod.PROFILE_FILE.write_text("version: 3\n")
        with pytest.raises(ProfileError, match="version 3"):
            load_store()
        profile_mod._load_store_cached.cache_clear()
        profile_mod.PROFILE_FILE.write_text("- a\n- b\n")
        with pytest.raises(ProfileError, match="Malformed"):
            load_store()


class TestResolution:
    """Resolving the active profile."""

    def test_placeholder_without_file(self):
        """Without a profile file a placeholder profile is returned."""
        profile = load_swift_profile()
        assert profile.swiftsim_dir is None
        assert profile.softening_coeff == 0.04

    def test_current_and_override(self, tmp_path):
        """The override beats the current pointer."""
        _store(tmp_path)
        assert load_swift_profile().name == "a"
        assert load_swift_profile().swiftsim_dir == tmp_path / "wt" / "a"
        set_profile_override("b")
        assert load_swift_profile().name == "b"

    def test_unknown_profile_suggests(self, tmp_path):
        """Unknown names give a clean error with close matches."""
        _store(tmp_path)
        set_profile_override("aa")
        with pytest.raises(ProfileError, match="Did you mean: a"):
            get_profile()

    def test_no_current(self, tmp_path):
        """A store with no current profile errors cleanly."""
        _store(tmp_path, current=None)
        with pytest.raises(ProfileError, match="--switch"):
            get_profile()

    def test_cache_dir(self, tmp_path):
        """Cache dirs are per profile and fall back to the root."""
        root = profile_mod.PROFILE_FILE.parent / "cache"
        assert get_profile_cache_dir() == root
        _store(tmp_path)
        assert get_profile_cache_dir() == root / "a"
        assert get_profile_cache_dir("b") == root / "b"
        assert (root / "b").is_dir()

    def test_timer_files_use_profile_cache(self, tmp_path):
        """Timer databases live in the active profile's cache dir."""
        from swiftsim_cli.timing import timer_db

        _store(tmp_path)
        expected = profile_mod.PROFILE_FILE.parent / "cache" / "a"
        assert timer_db._db_cache_dir() == expected / "timer_db"


class TestHelpers:
    """Small helpers."""

    @pytest.mark.parametrize("name", ["default", "zoom-v2", "a.b_c"])
    def test_valid_names(self, name):
        """Reasonable names pass."""
        validate_profile_name(name)

    @pytest.mark.parametrize("name", ["", "-x", "a/b", "has space", "../x"])
    def test_invalid_names(self, name):
        """Names unusable as branch/directory names fail."""
        with pytest.raises(ProfileError, match="Invalid profile name"):
            validate_profile_name(name)

    def test_git_error_message(self, tmp_path):
        """Git failures surface git's stderr."""
        with pytest.raises(ProfileError, match="not a git repository"):
            git("status", cwd=tmp_path)

    def test_parse_configure_options(self):
        """Real options are found, placeholders skipped, complements added."""
        opts = parse_configure_options(CONFIGURE_HELP)
        for opt in (
            "--help",
            "--quiet",
            "--prefix",
            "--enable-debug",
            "--enable-debugging-checks",
            "--disable-debug",
            "--disable-optimization",
            "--enable-option-checking",
            "--enable-gravity-force-checks",
            "--enable-shared",
            "--with-hydro",
            "--without-hydro",
            "--with-parmetis",
        ):
            assert opt in opts, opt
        assert not any("FEATURE" in o or "PACKAGE" in o for o in opts)
        assert "--enable-" not in opts


class TestPrompts:
    """The interactive prompt helpers."""

    def _ask(self, keys, *args, **kwargs):
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        with create_pipe_input() as inp:
            inp.send_text(keys)
            with create_app_session(input=inp, output=DummyOutput()):
                return profile_mod.ask(*args, **kwargs)

    def test_ask_text_and_paths(self, tmp_path):
        """Text and path prompts return the (default) answer."""
        assert self._ask(" hello \r", "Q: ") == "hello"
        assert self._ask("\r", "Q: ", default="dflt") == "dflt"
        answer = self._ask(
            "\r", "D: ", default=str(tmp_path), kind="dir", must_exist=True
        )
        assert answer == str(tmp_path)

    def test_confirm(self, monkeypatch):
        """Only an explicit yes confirms."""
        replies = iter(["y", "YES", "", "n"])
        monkeypatch.setattr("builtins.input", lambda q: next(replies))
        assert [profile_mod.confirm("ok?") for _ in range(4)] == [
            True,
            True,
            False,
            False,
        ]
