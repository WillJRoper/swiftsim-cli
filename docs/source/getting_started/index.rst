Getting Started
===============

Requirements
------------

.. include:: ../../../README.rst
   :start-after: REQS_START_LABEL
   :end-before: REQS_END_LABEL

Installing
----------

.. include:: ../../../README.rst
   :start-after: INSTALL_START_LABEL
   :end-before: INSTALL_END_LABEL

Setting up
----------

.. include:: ../../../README.rst
   :start-after: SETUP_START_LABEL
   :end-before: SETUP_END_LABEL

``init`` runs without prompts. It clones SWIFT from ``--swift-url`` (default: the GitHub mirror,
``https://github.com/SWIFTSIM/SWIFT.git``) unless ``--swift-path`` is
given (any directory inside the clone, or another worktree of it, means
the clone itself), then creates the first profile:

+ ``--name``: the profile's name (default: ``default``).
+ ``--upstream``: the branch it follows (default: ``origin/master``).
+ ``--configure-flags``: its configure flags, as one string.
+ ``-P/--params``: its parameter file (default: the worktree's
  ``examples/parameter_example.yml``).
+ ``--worktree-root``: where profile worktrees are created (default:
  ``~/.swiftsim-cli/worktrees``).

``init`` refuses to run again once profiles exist; add more with
``swift-cli profile --new NAME`` (see :doc:`../profiles/index`). To start
over, delete ``~/.swiftsim-cli`` and run ``init`` again: the profile's
branch (``swift-cli/<name>``) is still in the clone, with any commits on
it, and is picked up again.

Everything ``swift-cli`` stores lives in ``~/.swiftsim-cli``: the profiles in
``profiles.yaml``, the worktrees in ``worktrees/`` and per-profile caches
(the timer database used by ``analyse log``, the task names used by
``analyse task-plots`` and the configure options) in ``cache/<name>/``.

Usage example
-------------

.. include:: ../../../README.rst
   :start-after: USAGE_START_LABEL
   :end-before: USAGE_END_LABEL
