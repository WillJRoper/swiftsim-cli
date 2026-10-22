Setting up runs
===============

Run directories (``new``)
-------------------------

.. code-block:: bash

   swift-cli new -o runs/L100 --ics ics/L100.hdf5 --set Snapshots:basename=L100
   swift-cli new -o runs/L100 --ics ics/L100.hdf5 --template cosmo-hydro --check
   swift-cli new -o runs/L100 --ics ics/L100.hdf5 -P my_params.yml --force

``new`` writes ``runs/L100/params.yml``, starting from one of the bundled,
runnable parameter templates (derived from SWIFT's own examples):

====================  =============================================  ======================================
Template              For                                            SWIFT flags
====================  =============================================  ======================================
``cosmo-dmo``         cosmological, dark matter only                 ``--cosmology --self-gravity``
``cosmo-hydro``       cosmological, gravity + hydro                  ``--cosmology --self-gravity --hydro``
``isolated-gravity``  non-cosmological, non-periodic, gravity        ``--self-gravity``
``isolated-hydro``    non-cosmological, non-periodic, gravity+hydro  ``--self-gravity --hydro``
====================  =============================================  ======================================

The template is chosen from the initial conditions: cosmological if the
header has a positive ``Redshift`` or 0 < ``Time`` < 1, hydro if there is
gas. ``--template NAME`` picks one explicitly, and ``-P/--params FILE`` (or
the profile's ``parameter_file``, unless it is SWIFT's
``parameter_example.yml``) starts from your own file instead. ``new`` then
fills in:

+ absolute paths to the ICs and to ``snapshots/`` and ``restart/`` in the
  run directory;
+ the unit system from the ICs' ``Units`` group. Cosmological ICs without
  one are taken to be Gadget-style (h\ :sup:`-1` units, sqrt(a)
  velocities), so ``cleanup_h_factors`` and ``cleanup_velocity_factors``
  are set to 1;
+ ``a_begin`` (cosmological) or ``time_begin``, with the first snapshot
  and statistics times to match; non-cosmological runs get no
  ``Cosmology`` section;
+ softenings: the profile's ``softening_coeff`` times the mean
  inter-particle separation (comoving, with the maximal physical softening
  at the pivot redshift, for cosmological runs; maximal physical only, with
  a warning, otherwise). Zoom ICs (with a ``PartType2`` background) use the
  high-resolution particle mass and set
  ``Gravity:softening_ratio_background``, plus a ``ZoomRegion`` section and
  the ``--zoom`` flag when the profile's SWIFT has zoom support
  (``src/zoom_region``);
+ ``generate_gas_in_ics`` when a hydro template is used with
  dark-matter-only ICs;
+ your ``--set SECTION:KEY=VALUE`` overrides (repeatable). Integers and
  floats are written as numbers, ``[a, b]`` as a list and anything else
  verbatim; new keys or sections are added with a warning (in case of a
  typo), and empty values are rejected.

Before anything is written the result is checked against SWIFT's parameter
parser: no duplicate keys or sections, lines of at most 255 characters, at
most 700 parameters and 64 sections, and no placeholder values such as
``FLT_MAX``. The run directory is created only once it passes. ``new``
prints the softenings it wrote and the ``swift ...`` command to run, adding
``--stars`` when the ICs contain stars (SWIFT drops them otherwise). An
existing ``params.yml`` is only replaced with ``--force``.

``--check`` runs the profile's ``swift`` binary with ``--dry-run`` and the
right physics flags on the new file, from inside the run directory, and
reports SWIFT's verdict.

Output times (``output-times``)
-------------------------------

.. code-block:: bash

   # Snapshots every 0.5 in redshift from z = 10 to a_end in the params file
   swift-cli output-times -P params.yml --first-snap-z 10 --delta-z 0.5

   # Snapshots every 0.05 in scale factor, plus snipshots every 0.01
   swift-cli output-times -P params.yml --first-snap-scale-factor 0.1 \
       --delta-scale-factor 0.05 --snipshot-delta-scale-factor 0.01 \
       --output-list output_list.txt --force

   # Non-cosmological run: times in internal units
   swift-cli output-times --first-snap-time 0 --delta-time 0.1 --final-snap-time 2

The first output, the final output, the snapshot interval and the snipshot
interval are each given in one unit (``--first-snap-{z,time,scale-factor}``,
``--final-snap-...``, ``--delta-{z,time,scale-factor,log-scale-factor}``,
``--snipshot-delta-...``); contradictory, zero or negative values are usage
errors. ``--delta-log-scale-factor`` is the interval in log\ :sub:`10`\ (a).
The final output defaults to ``Cosmology:a_end`` from the parameter file.

The list is written to ``--output-list FILE`` (default: ``output_list.txt``
in the current directory); an existing list is only replaced with
``--force``. The interval chooses the units of the list, and
``--list-units`` overrides them. The list stops at the final output,
snapshot and snipshot entries are never duplicated (snapshots win), and it
must hold at least two outputs. With snipshots the list gets a
``Select Output`` column, and SWIFT also needs
``Snapshots:select_output_on: 1`` and a select-output file (a note is
printed).

With a cosmology, ``--*-time`` values are ages of the Universe in Gyr, and
the list is written as ages in SWIFT's *internal* time units
(``UnitLength_in_cgs / UnitVelocity_in_cgs`` from the parameter file's
``InternalUnitSystem``), which is what SWIFT expects for a ``# Time`` list
in a cosmological run. ``--no-cosmology`` ignores the parameter file's
``Cosmology`` section. Redshifts and scale factors need no cosmology; only
times do. Conversions follow SWIFT's own cosmology (radiation, neutrinos,
w\ :sub:`0`/w\ :sub:`a`).

Softenings (``utils softenings``)
---------------------------------

.. code-block:: bash

   swift-cli utils softenings --ics ics.hdf5
   swift-cli utils softenings --ics ics.hdf5 -P params.yml --softening-coeff 0.04

Prints the mean inter-particle separation and the comoving and maximal
physical softening of each particle species (dark matter, baryons,
neutrinos and the zoom background), with exactly the calculation ``new``
uses. ``-P`` is optional (the default is the template ``new`` would pick);
the coefficient and pivot redshift default to the profile's values.
