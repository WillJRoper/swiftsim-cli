Tasks (``analyse task-counts``, ``analyse task-plots``)
=======================================================

Task counts
-----------

.. code-block:: bash

   swift-cli analyse task-counts swift.log --tasks self pair --steps 10 100

Plots the ``engine_print_task_counts`` output (SWIFT run with ``-v 1``) per
step. The total is SWIFT's ``System total`` (all ranks) when it is printed,
otherwise the sum of each rank's total; the per-type counts are the tasks
that run (skipped tasks are one ``skipped (any type)`` row). A log with
only some ranks' counts (typically rank 0 of an MPI run) gives a warning,
and the axis labels say "(rank 0 only)". Unknown ``--tasks`` names are
errors.

Task plots
----------

.. code-block:: bash

   # Task-debug (-y) timings, optionally with the threadpool (-Y) dump above
   swift-cli analyse task-plots thread_info-step64.dat \
       --threadpool threadpool_info-step64.dat \
       --tasks pair/force self/force --activity-plot
   swift-cli analyse task-plots thread_info_MPI-step64.dat --rank 0 --c-type zoom

Needs SWIFT configured with ``--enable-task-debugging`` and run with
``-y N`` (and, for ``--threadpool``, ``--enable-threadpool-debugging`` and
``-Y N``). Every ``task_dump_all`` layout is read: upstream 11-column
(non-MPI) and 13-column (MPI) dumps, older zoom 17-column MPI rows, and
zoom 19/21-column dumps. The cell filters (``--c-type``, ``--ci-subtype``,
..., by name or index) need a 19/21-column dump.

Task, subtype and cell-type names are parsed from the SWIFT source
(``src/task.c``, ``src/cell.h``) of the active profile's checkout, or of
``--swift-src DIR``, and cached per profile. Without a source tree,
built-in tables are used with a warning; indices outside the tables warn
and show as ``unknown_N``. Unknown ``--tasks`` labels or cell names are
errors listing the valid names, and ranks with no matching task are
skipped with a warning. All ranks share one time window (by default the
longest step in the dump headers; ``--limit``, ``--mintic``).
