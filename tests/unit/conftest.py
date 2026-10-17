"""Fixtures providing a real git SWIFT-like repository for profile tests."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from swiftsim_cli.profile import ProfileStore, save_store

CONFIGURE_HELP = """\
`configure' configures SWIFT 2025.10 to adapt to many kinds of systems.

Usage: ./configure [OPTION]... [VAR=VALUE]...

Configuration:
  -h, --help              display this help and exit
  -q, --quiet, --silent   do not print `checking ...' messages
  --prefix=PREFIX         install architecture-independent files in PREFIX

Optional Features:
  --disable-option-checking  ignore unrecognized --enable/--with options
  --disable-FEATURE       do not include FEATURE (same as --enable-FEATURE=no)
  --enable-FEATURE[=ARG]  include FEATURE [ARG=yes]
  --enable-debug=[yes/info/profile/no]
                          compile with debugging
  --enable-debugging-checks
                          Activate expensive consistency checks [yes/no]
  --enable-ipo            Enable interprocedural optimization [default=yes]
  --enable-optimization   Enable compile time optimization flags for host
  --enable-gravity-force-checks=<N>
                          Activate the direct summation gravity checks
  --enable-shared[=PKGS]  build shared libraries [default=no]

Optional Packages:
  --with-PACKAGE[=ARG]    use PACKAGE [ARG=yes]
  --without-PACKAGE       do not use PACKAGE (same as --with-PACKAGE=no)
  --with-hydro=<scheme>   Hydro dynamics to use [gadget2, sphenix, ...]
  --with-subgrid=<subgrid>
                          Master switch for subgrid methods
  --with-kernel=<kernel>  Kernel function to use [cubic-spline, ...]
  --with-tbbmalloc        use tbbmalloc for memory allocations
  --with-parmetis=PATH    root directory where ParMETIS is installed

Some influential environment variables:
  CC          C compiler command
"""

# A fake configure: prints CONFIGURE_HELP for --help, otherwise records its
# arguments, writes a config.status supporting --config and a Makefile.
CONFIGURE_SCRIPT = f"""#!/bin/sh
if [ "$1" = "--help" ]; then
cat <<'HELP'
{CONFIGURE_HELP}HELP
exit 0
fi
echo "$*" >> configure.log
quoted=""
for a in "$@"; do quoted="$quoted '$a'"; done
printf '#!/bin/sh\\necho "%s"\\n' "$quoted" > config.status
chmod +x config.status
printf 'all:\\n\\ttouch swift\\nclean:\\n\\trm -f swift\\n' > Makefile
"""

AUTOGEN_SCRIPT = """#!/bin/sh
cp configure.ac configure
chmod +x configure
"""

PARAMS = """\
Cosmology:
  h: 0.6777
  a_begin: 0.0078125
  Omega_cdm: 0.2587
"""


def run_git(*args, cwd=None):
    """Run git in a directory and return its stdout."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture(scope="session")
def git_env(tmp_path_factory):
    """Isolate git from the user's global/system configuration."""
    cfg = tmp_path_factory.mktemp("gitcfg") / "gitconfig"
    cfg.write_text(
        "[user]\n\tname = Test\n\temail = test@example.com\n"
        "[init]\n\tdefaultBranch = master\n"
        "[advice]\n\tdetachedHead = false\n"
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_CONFIG_GLOBAL", str(cfg))
        mp.setenv("GIT_CONFIG_NOSYSTEM", "1")
        yield


@pytest.fixture(scope="session")
def seed_remote(git_env, tmp_path_factory):
    """Build a bare 'remote' SWIFT-like repo with master + feature branches."""
    base = tmp_path_factory.mktemp("seed")
    work = base / "work"
    work.mkdir()
    run_git("init", "-q", "-b", "master", cwd=work)
    (work / "autogen.sh").write_text(AUTOGEN_SCRIPT)
    (work / "autogen.sh").chmod(0o755)
    (work / "configure.ac").write_text(CONFIGURE_SCRIPT)
    (work / ".gitignore").write_text(
        "configure\nconfig.status\nMakefile\nswift\nconfigure.log\n"
    )
    (work / "src.c").write_text("int x = 1;\n")
    (work / "examples").mkdir()
    (work / "examples" / "parameter_example.yml").write_text(PARAMS)
    run_git("add", ".", cwd=work)
    run_git("commit", "-q", "-m", "initial", cwd=work)
    run_git("checkout", "-q", "-b", "feature", cwd=work)
    (work / "feature.c").write_text("int feature = 1;\n")
    run_git("add", ".", cwd=work)
    run_git("commit", "-q", "-m", "feature", cwd=work)
    run_git("checkout", "-q", "master", cwd=work)
    remote = base / "remote.git"
    run_git("clone", "-q", "--bare", str(work), str(remote))
    return remote


@dataclass
class GitSetup:
    """Paths of a per-test remote, main clone and worktree root."""

    tmp: Path
    remote: Path
    repo: Path
    root: Path

    def store(self) -> ProfileStore:
        """Write and return an empty version-2 store for this setup."""
        store = ProfileStore(repo=self.repo, worktree_root=self.root)
        save_store(store)
        return store

    def push(self, branch: str, filename: str, content: str) -> None:
        """Commit a change to ``branch`` of the remote."""
        pusher = self.tmp / "pusher"
        if not pusher.exists():
            run_git("clone", "-q", str(self.remote), str(pusher))
        run_git("checkout", "-q", branch, cwd=pusher)
        run_git("pull", "-q", "--rebase", cwd=pusher)
        (pusher / filename).write_text(content)
        run_git("commit", "-q", "-am", f"change {filename}", cwd=pusher)
        run_git("push", "-q", "origin", branch, cwd=pusher)


@pytest.fixture
def gitsetup(seed_remote, tmp_path):
    """A fresh remote and main clone for one test."""
    remote = tmp_path / "remote.git"
    shutil.copytree(seed_remote, remote)
    repo = tmp_path / "swiftsim"
    run_git("clone", "-q", str(remote), str(repo))
    return GitSetup(
        tmp=tmp_path,
        remote=remote,
        repo=repo.resolve(),
        root=(tmp_path / "swiftsim-profiles").resolve(),
    )
