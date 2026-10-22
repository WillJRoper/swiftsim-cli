Scaling (``analyse scaling``)
=============================

.. code-block:: bash

   swift-cli analyse scaling timers logs/*.log --steps 10 100 -p zoom
   swift-cli analyse scaling timers logs/*.log --rank-source all --swift-src ~/swiftsim
   swift-cli analyse scaling runtime logs/*.log

Both compare logs of runs with at least two different MPI rank counts, and
warn when their step counts differ by more than 1%. Both leave out step 0
and restart rows, like ``analyse log``.

``scaling timers`` uses the same per-step timer trees as
:doc:`analyse log <log>` (and the same timer database flags
``--swift-src``, ``--no-match-revision`` and ``--regenerate-db``), and plots
the timers whose exclusive time is at least ``--min-percent-threshold``
percent of the step time in at least one log (default: 5).
``--rank-source root`` (the default) uses rank 0; ``all`` uses the mean over
the ranks that printed timers (``-v 2`` logs). A "SCALING INPUTS" table
lists each log's ranks, steps, step time, timed share and unmatched timer
lines, and ``rank_imbalance_summary.txt`` gives each timer's max/mean over
ranks.

``scaling runtime`` uses the summed step-table wall-clock time.
