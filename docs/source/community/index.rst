Community
=========

.. include:: ../../../README.rst
   :start-after: COMMUNITY_START_LABEL
   :end-before: COMMUNITY_END_LABEL

Development environment
-----------------------

Clone the repository and install it in editable mode with the development
and test dependencies:

.. code-block:: bash

   git clone https://github.com/WillJRoper/swiftsim-cli.git
   cd swiftsim-cli
   pip install -e ".[dev,test]"
   pre-commit install

Then:

.. code-block:: bash

   ruff check src tests && ruff format src tests
   mypy                       # src and tests, config in pyproject.toml
   pytest                     # with coverage (fails under 85%)
   SWIFT_SRC=/path/to/swiftsim pytest tests/unit/test_timer_db_archetypes.py -k real

The last command is an optional smoke test of the timer parser against a
real SWIFT checkout (skipped unless ``SWIFT_SRC`` is set). See
``tests/README.md`` for the test layout and ``CONTRIBUTING.md`` for the
contribution and release process.

Project layout
--------------

+ ``src/swiftsim_cli/cli.py``: the entry point; ``multi_mode_args.py``:
  multi-mode parsing.
+ ``src/swiftsim_cli/modes/``: one module per mode (``analyse/`` holds the
  analyses, ``analyse/scaling/`` the scaling package).
+ ``src/swiftsim_cli/templates/``: the parameter templates used by
  ``new``.
+ ``src/swiftsim_cli/profile.py``: profile storage and git worktree
  management.
+ ``src/swiftsim_cli/swiftsim_dir.py``: configure, build and update
  helpers.
+ ``src/swiftsim_cli/timing/``: the timer database, log scanning and the
  per-step call trees behind ``analyse log``.
+ ``src/swiftsim_cli/utilities/``: shared I/O, plotting and table helpers.
