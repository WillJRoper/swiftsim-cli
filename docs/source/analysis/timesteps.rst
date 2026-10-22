Timesteps (``analyse timesteps``)
=================================

.. code-block:: bash

   # Timestep files (or logs) of two runs, in core-hours
   swift-cli analyse timesteps run1/timesteps.txt run2/timesteps.txt \
       -l run1 run2 --cores-per-run 64 128 --time-column redshift

Plots the cumulative wall-clock and dead time of each run against
``--time-column``. ``auto`` (the default) uses the scale factor if it varies
in every input and time otherwise (non-cosmological runs), with a note when
it falls back; ``step``, ``time``, ``scale-factor`` and ``redshift`` choose
explicitly. ``--cores-per-run N ...`` (MPI ranks x threads per rank, one
per run) turns the times into core-hours, and ``--match-runtimes`` trims
every run to the shortest x range. The runtime comparison table is skipped,
with a warning, when the x column is constant or decreasing.

Steps repeated after a restart replace the earlier rows ("N rows superseded
by a restart"), malformed rows are counted and skipped, and gaps in the step
sequence are warned about and break the plotted lines. The
``Dead time [ms]`` column is optional (older SWIFT versions). The plot is
``timesteps_analysis/timesteps_analysis.png``.
