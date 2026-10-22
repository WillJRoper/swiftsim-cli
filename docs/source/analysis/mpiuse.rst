MPI communication (``analyse mpiuse``)
======================================

.. code-block:: bash

   swift-cli analyse mpiuse master_reports fewer_gpart_comms_reports \
       --labels master fewer_gpart_comms --prefix gravity_cmp \
       --format both --per-rank

Compares the MPI communication SWIFT records when built with
``--enable-mpiuse-reports``. Each input is a directory, a glob or a set of
``mpiuse_report-rank*-step*.dat`` files for one run; several inputs are
compared side by side.

Each message is logged by SWIFT as a send and a receive; volumes are
counted once (from the send side), and bytes sent and received are reported
separately. The output has per-input summary tables (with ``TOTAL send``
and ``TOTAL recv`` rows), a cross-input comparison table, per-step and
per-subtype plots, per-rank tables and plots with ``--per-rank``, and
per-step CSV files with ``--format csv|both``. ``--subtypes gpart
grav_counts`` restricts everything, including the in-flight high-water
mark, to those subtypes; ``--steps START END`` restricts the steps.
