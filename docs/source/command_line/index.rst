Command line
============

``swift-cli`` runs one or more *modes*, in order:

.. code-block:: text

   swift-cli [global options] <mode> [mode args] [<mode> [mode args]] ...

``swift-cli --help`` prints:

.. code-block:: text

   swift-cli: Utilities for SWIFT development workflows

   Usage:
     swift-cli [global options] <mode> [mode args] [<mode> [mode args]] ...

   Global options (must come before the first mode):
     -h, --help      Show this help and exit.
     --version       Show the version and exit.
     --profile NAME  Run the modes against profile NAME.
     --debug         Show full tracebacks for errors (or set
                     SWIFT_CLI_DEBUG=1).

   Available modes:
     init            Clone or reuse SWIFT and create the first profile.
     profile         Manage SWIFT profiles (one worktree, branch and build each).
     config          Set the current profile's configure flags and reconfigure.
     output-times    Generate a list of snapshot (and snipshot) output times.
     update          Pull and rebuild the current profile.
     switch          Make another profile the current one.
     make            Compile the current profile.
     new             Create a new simulation run directory.
     analyse         Analyse SWIFT logs and diagnostic outputs.
     show            Show the current profile (and optionally its cosmology).
     utils           Run small helper calculations (e.g. softenings).

   Modes run in order and the chain stops at the first error. A mode name
   starts the next mode unless the current mode expects a value there; use
   '--' to end a mode's arguments explicitly.

   Examples:
     swift-cli config --enable-debug
     swift-cli make -j 8
     swift-cli config --enable-debug make -j 32
     swift-cli config --enable-debug -- make -j 32
     swift-cli --profile zoom make -j 16

   For mode-specific help: swift-cli <mode> --help

Every mode has its own help: ``swift-cli <mode> --help`` (and
``swift-cli analyse <analysis> --help``).

Modes
-----

========================  ====================================================
Mode                      What it does
========================  ====================================================
``init``                  Clone or use SWIFT and create the first profile
``profile``               Create, list, show, switch, edit, update, remove and
                          adopt profiles
``config``                Set the current profile's configure flags and
                          reconfigure
``make``                  Compile the current profile
``update``                Pull (rebase onto upstream) and rebuild the current
                          profile
``switch``                Make another profile the current one
``show``                  Show a profile and its parameter file's cosmology
``new``                   Create a run directory with a ``params.yml``
``output-times``          Generate an output-time list
``analyse``               Analyse SWIFT logs and diagnostics
``utils``                 Small helper calculations (``softenings``)
========================  ====================================================

Global options
--------------

Global options must come before the first mode:

+ ``--profile NAME``: run every mode that acts on one profile against
  profile ``NAME`` instead of the current one (e.g.
  ``swift-cli --profile zoom make -j 16``). The name is checked before any
  mode runs; ``profile --list``, ``--new`` and ``--switch`` ignore it.
+ ``--debug``: show full tracebacks instead of a one-line ``error: ...``
  (or set ``SWIFT_CLI_DEBUG=1``).
+ ``--version``, ``-h/--help``.

Chaining modes
--------------

.. code-block:: bash

   # Configure with the debug preset, then compile
   swift-cli config --debug-build make -j 16

   # Compile, then create a run directory
   swift-cli make -j 32 new -o test_run --ics initial_conditions.hdf5

   # Analyse a log, then show the profile
   swift-cli analyse log run/swift.log -o run show

Modes run in order and the chain stops at the first error. A word that
names a mode starts the next mode, unless the current mode expects it as a
value: ``new -o make`` creates a run directory called ``make``, and
``switch show`` switches to a profile called ``show``. ``--`` always ends
the current mode's arguments (``config --enable-debug -- make -j 8``).

Every mode parses its arguments before the first one runs, so a typo in the
last mode fails before anything is built. Input files and directories are
checked then too, except the parameter file (``-P/--params``), which is
checked when its mode runs because an earlier ``new`` may create it. Long
options must be spelled out in full (no abbreviations).

Common flags
------------

The same flag means the same thing in every mode:

+ ``-P/--params FILE``: a SWIFT parameter file.
+ ``--ics FILE``: initial conditions.
+ ``-o/--out DIR``: always an output *directory*.
+ ``-p/--prefix PREFIX``: prepended once to every output filename.
+ ``-j/--nr-threads N``: ``make`` threads.
+ ``-l/--labels LABEL ...``: one label per input (default: the file stems).
+ ``--steps START END``, ``--rank N|all``, ``--tasks NAME ...``.
+ ``--show``: also show plots interactively.

Errors and exit codes
---------------------

Errors are reported as a single ``error: ...`` line on stderr; use
``--debug`` or ``SWIFT_CLI_DEBUG=1`` for the traceback. Warnings (for
example one of several inputs that could not be used) are ``warning: ...``
lines and do not stop the chain.

+ ``0``: success (also for ``--help`` and ``--version``).
+ ``1``: an error while running a mode (e.g. an unreadable input or an
  analysis that found no usable data), or no mode given (the help is
  printed).
+ ``2``: a usage error (unknown option, invalid value, missing input file).
+ A failed external command (``configure``, ``make``, ``git``) exits with
  its own code; an interrupt exits with 130.
