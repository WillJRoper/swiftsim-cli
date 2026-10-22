Gravity accuracy (``analyse gravity-checks``, ``analyse gravity-error-maps``)
=============================================================================

.. code-block:: bash

   swift-cli analyse gravity-checks gravity_checks_exact_step0000.dat --all-orders
   swift-cli analyse gravity-error-maps gravity_checks_exact_step0000.dat \
       --resolution 50 --vmin 1e-6 --vmax 1e-2 --box-size 100

Both need SWIFT configured with ``--enable-gravity-force-checks=N``. Give
either file of a pair (``gravity_checks_exact_*`` or
``gravity_checks_swift_*``); the counterpart
(``..._step<N>[_order<K>].dat``) is found next to it and particles are
matched by ID. An exact file with SWIFT files of several multipole orders
is compared with the lowest order unless ``--all-orders`` is given.

Particles with non-finite values are dropped with a warning. Errors are
undefined where the exact acceleration is zero (counted by
``gravity-checks``, dropped with a warning by ``gravity-error-maps``), and a
warning is given when the exact and SWIFT positions differ.

``gravity-checks`` prints one table per run (the median, 90% and 99% error
per component, the shares below ``--min-error`` and above ``--max-error``,
and the undefined count) and plots error histograms with a percentile
summary panel.

``gravity-error-maps`` maps the median relative error per hexagon (or per
particle, for samples of up to 5000) in three projections, plus a map of
the particles above ``--thresh`` with an error histogram. The colour scale
runs from the 1st to the 99th percentile of the errors unless
``--vmin``/``--vmax`` are given; distances are measured from the median
particle position, wrapped periodically with ``--box-size L``.
