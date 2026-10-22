Log timing (``analyse log``)
============================

``analyse log`` shows where the time of a run goes, from the timers SWIFT
prints with ``-v 1`` (rank 0) or ``-v 2`` (every rank):

.. code-block:: bash

   swift-cli analyse log swift.log                     # rank 0, all steps
   swift-cli analyse log swift.log --steps 100 200 --top-n 10
   swift-cli analyse log swift.log --rank all          # per rank + imbalance
   swift-cli analyse log swift.log --hierarchy-functions engine_rebuild
   swift-cli analyse log swift.log --swift-src ~/swiftsim --regenerate-db

How it works
------------

+ **Timers come from the source.** The timer calls are read from the SWIFT
  source with tree-sitter (including ``examples/main.c``). By default the
  source is taken at the revision printed in the log, extracted with
  ``git archive`` from the profile's repository into the profile cache (the
  checkout is not touched). ``--no-match-revision`` uses the checkout as it
  is, and ``--swift-src DIR`` any other source tree. The parsed database is
  cached and rebuilt only when the source changes (``--regenerate-db``
  forces a rebuild). The share of timer lines matched to the source is
  always reported, with a warning when more than 5% are unmatched.
+ **Each step is a call tree.** The timers of each step are nested into
  the call tree that actually ran: a function called from several places is
  attributed to the caller that contained each call, so nothing is counted
  twice.
+ **Steps.** The row SWIFT prints for step N comes at the start of step
  N+1, so the timers before row N are step N's, and those before row 0 are
  initialisation. Timers ``main()`` prints between steps (for example
  ``task_dump_all``) go to a separate "BETWEEN STEPS" table. ``--steps``
  outside the log's steps is an error naming the available range.
+ **Ranks are never summed.** ``--rank N`` picks one rank (default 0);
  ``--rank all`` analyses each rank and adds an imbalance table.
+ **Damaged lines** (written by several ranks at once) are split and
  rebuilt where possible; a warning reports how many lines were cut,
  recovered or dropped, and steps whose rows are missing are left out.

Reading the numbers
-------------------

+ **Inclusive** time is the time inside a timer including everything
  nested in it; **exclusive** (self) time excludes nested timers.
  Exclusive times add up to the timed total; inclusive times of nested
  timers do not.
+ **Every percentage** is a share of the summed step wall-clock time of the
  analysed steps, not of the sum of the timers. The top-level breakdown,
  including the untimed remainder, sums to 100%; a warning is given if any
  step's coverage exceeds 105%.
+ **Revision matching**: if the log's revision is not in the profile's
  repository (e.g. an unpushed commit), use ``--no-match-revision`` or
  ``--swift-src``; the match rate shows how many timer lines were
  recognised.

Outputs
-------

The console shows a short summary. ``log_analysis/analysis_tables.txt`` has
the call tree, the timers by exclusive time, the per-caller breakdown of
the ``--hierarchy-functions``, the steps whose coverage is outside 90-105%,
initialisation, the between-steps timers, task categories and task counts,
alongside the charts. The coverage of every step is in
``log_analysis/per_step_coverage.csv``. Outputs left from an earlier run in
the same directory are removed first.
