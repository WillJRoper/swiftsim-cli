Analysis
========

``swift-cli analyse <analysis>`` reads the logs and diagnostic files SWIFT
writes. ``swift-cli analyse --help`` prints:

.. code-block:: text

   usage: swift-cli analyse [-h] ANALYSIS ...

   Analyse SWIFT logs and diagnostic outputs.

   positional arguments:
     ANALYSIS            Type of analysis to perform.
       timesteps         Plot the timestep files of one or more runs.
       gravity-checks    Compare SWIFT gravity forces with exact gravity checks.
       gravity-error-maps
                         Create hexbin error maps for gravity check files.
       log               Analyse the timers in a SWIFT log file (run SWIFT with
                         -v 1 for rank 0 timers, -v 2 for every rank).
       task-counts       Plot the engine_print_task_counts output of SWIFT log
                         files per step.
       mpiuse            Analyse SWIFT mpiuse_report-rank*-step*.dat files and
                         compare communication volume across inputs.
       task-plots        Plot SWIFT task-debug timings (optionally with a
                         threadpool dump) per thread.
       scaling           Analyse how SWIFT scales across runs with different MPI
                         rank counts.

   options:
     -h, --help          show this help message and exit

   Each analysis writes its files into <out>/<analysis>_analysis/ (-o/--out,
   default: the current directory), with -p/--prefix prepended to each filename.

Outputs
-------

All analyses share the same output options:

+ ``-o/--out DIR``: outputs go into ``DIR/<analysis>_analysis/`` (e.g.
  ``timesteps_analysis/``, ``log_analysis/``,
  ``scaling_timers_analysis/``); ``DIR`` defaults to the current directory.
+ ``-p/--prefix PREFIX``: prepended once to every output filename
  (``PREFIX_<name>``).
+ ``--show``: also show the plots interactively.

Analyses that compare several inputs take optional ``-l/--labels LABEL
...``, one per input (default: the file stems, with ``_1``, ``_2``, ...
added to stems shared by several inputs); put the inputs before ``-l``.
Step-based analyses take ``--steps START END``, and those that can pick an
MPI rank take ``--rank N|all``.

An analysis that finds no usable data in any input fails (exit code 1);
inputs that cannot be used while others can are skipped with a warning.

.. toctree::
   :maxdepth: 1

   log
   timesteps
   gravity
   tasks
   mpiuse
   scaling
