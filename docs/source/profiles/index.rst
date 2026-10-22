Profiles and builds
===================

A profile is a fully independent SWIFT build. Every profile has:

+ its own **git worktree** of the SWIFT clone
  (``~/.swiftsim-cli/worktrees/<name>`` by default);
+ its own **local branch**, ``swift-cli/<name>``, tracking an **upstream**
  branch (e.g. ``origin/master``);
+ its own **configure flags** and build products.

Builds never share a directory, so switching profiles is instant and can
never mix builds. ``config``, ``make``, ``update``, ``new`` and the other
modes act on the *current* profile; the global ``--profile NAME`` option
targets another one without switching.

The first profile is created by ``swift-cli init`` (see
:doc:`../getting_started/index`).

Managing profiles
-----------------

.. code-block:: bash

   swift-cli profile --new zoom --upstream origin/zoom \
       --configure-flags="--enable-debug" --build -j 16
   swift-cli profile --list              # all profiles (* marks the current one)
   swift-cli profile --show zoom         # one profile's settings (default: current)
   swift-cli profile --switch zoom       # or: swift-cli switch zoom
   swift-cli profile --edit zoom         # edit its settings in $EDITOR
   swift-cli profile --edit zoom --set upstream=origin/zoom-v2
   swift-cli profile --update zoom -j 16 # pull and rebuild
   swift-cli profile --update-all -j 16  # every profile, then a summary
   swift-cli profile --remove zoom       # remove the worktree (and optionally the branch)
   swift-cli --profile zoom make -j 16   # build another profile without switching

Editing
^^^^^^^

``profile --edit NAME`` opens the profile's settings as YAML in
``$VISUAL``/``$EDITOR`` (default: ``vi``); the changes are checked and
applied when you save and quit. The settings are ``upstream``,
``configure_flags``, ``data_dir``, ``parameter_file``, ``softening_coeff``
and ``softening_pivot_z``. ``--set KEY=VALUE`` (repeatable) changes only the
given settings, without an editor.

Changing the upstream moves a branch with no commits of its own to the new
upstream (refused if the worktree has uncommitted changes); a branch with
its own commits keeps them, and the ``git rebase --onto`` command to move
them is printed. Changing the configure flags makes the next ``update``,
``config`` or ``make`` reconfigure.

Removing and adopting
^^^^^^^^^^^^^^^^^^^^^

``--remove`` deletes only worktrees ``swift-cli`` created; adopted
worktrees, the main clone and worktrees shared with another profile are
only unregistered. It asks before deleting the branch; ``-y/--yes`` keeps
the branch and prints the ``git branch -D`` command, and ``--force``
removes a worktree with untracked files. ``profile --new NAME
--reuse-branch`` recreates a profile on its kept branch.

Worktrees made by hand can be registered as profiles:

.. code-block:: bash

   swift-cli profile --adopt ../swiftsim-my-feature --name my-feature

Building
--------

.. code-block:: bash

   swift-cli config --list-options                    # ./configure --help
   swift-cli config --debug-build                     # preset flags
   swift-cli config --eagle --with-cooling=COLIBRE    # preset + any --<option>
   swift-cli config --with-hydro=sphenix CC=icx       # options and VAR=value
   swift-cli config                                   # reconfigure with stored flags
   swift-cli make -j 16                               # --clean runs make clean first
   swift-cli update -j 16                             # pull, reconfigure if needed, build
   swift-cli update --no-build                        # only pull

``config`` sets the current profile's configure flags and reconfigures. The
presets are ``--debug-build``, ``--gravity``, ``--eagle`` and ``--eaglexl``
(the last two are mutually exclusive). Any other ``--<option>[=value]`` is
checked against ``./configure --help`` in the profile's worktree, with
suggestions for typos, and passed on, as are ``VAR=value`` assignments;
anything else is rejected (``--with-hydro sphenix`` suggests
``--with-hydro=sphenix``). A later value for an option overrides an earlier
one, with a warning. The result replaces the profile's stored flags.

``make`` compiles the current profile, reconfiguring first if its flags
changed. A failed ``configure`` or ``make`` exits with that command's own
exit code.

Updating
^^^^^^^^

Updating a profile runs ``git pull --rebase --autostash`` in its worktree:
your local commits are replayed on top of the upstream, and uncommitted
changes are stashed and re-applied. It fails up front if the branch has no
upstream, and reports "Already up to date" or how many commits were pulled.
If the rebase, or re-applying your changes, hits conflicts, nothing is
built and you are told how to resolve them. Otherwise ``./autogen.sh`` runs
if ``./configure`` is missing, ``./configure`` runs only if the profile has
never been configured or its flags changed, and then ``make`` runs. No
build starts in a worktree that is mid-rebase or has unresolved conflicts.

When ``update`` is followed by ``config`` or ``make`` in the same command it
skips its own build, so ``swift-cli update config --debug-build make -j 8``
builds once.

Showing a profile
-----------------

.. code-block:: bash

   swift-cli show            # current profile and its parameter file's cosmology
   swift-cli show zoom --cosmo   # only the cosmology of profile zoom
