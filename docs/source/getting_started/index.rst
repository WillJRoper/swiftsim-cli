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

``swift-cli init`` sets everything up once, without prompts:

.. code-block:: bash

   swift-cli init                           # clone SWIFT into ./swiftsim
   swift-cli init --swift-path ~/swiftsim   # or use an existing clone

It clones SWIFT from ``--swift-url`` (default: the GitHub mirror,
``https://github.com/SWIFTSIM/SWIFT.git``) unless ``--swift-path`` is
given, then creates the first profile:

+ ``--name``: the profile's name (default: ``default``).
+ ``--upstream``: the branch it follows (default: ``origin/master``).
+ ``--configure-flags``: its configure flags, as one string.
+ ``--worktree-root``: where profile worktrees are created (default:
  ``~/.swiftsim-cli/worktrees``).

``init`` refuses to run again once profiles exist; add more with
``swift-cli profile --new NAME`` (see :doc:`../profiles/index`).

Everything ``swift-cli`` stores lives in ``~/.swiftsim-cli``: the profiles in
``profiles.yaml``, the worktrees in ``worktrees/`` and per-profile caches
(the timer database used by ``analyse log``, the task names used by
``analyse task-plots`` and the configure options) in ``cache/<name>/``.

Usage example
-------------

.. include:: ../../../README.rst
   :start-after: USAGE_START_LABEL
   :end-before: USAGE_END_LABEL
