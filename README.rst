swiftsim-cli
============

|Python version| |PyPI version| |Build status| |License| |Ruff|

.. |Python version| image:: https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FWillJRoper%2Fswiftsim-cli%2Fmain%2Fpyproject.toml
   :alt: Supported python versions
.. |PyPI version| image:: https://img.shields.io/pypi/v/swiftsim-cli
   :target: https://pypi.org/project/swiftsim-cli
   :alt: Version released on PyPI
.. |Build status| image:: https://github.com/WillJRoper/swiftsim-cli/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/WillJRoper/swiftsim-cli/actions/workflows/ci.yml
   :alt: Build status
.. |License| image:: https://img.shields.io/badge/license-GPL--3.0--or--later-blue
   :target: https://github.com/WillJRoper/swiftsim-cli/blob/main/LICENSE
   :alt: License: GPL-3.0-or-later
.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff

.. INTRO_START_LABEL

``swift-cli`` is a command-line tool for building, running and analysing
simulations with the SWIFT_ astrophysics code.

It manages any number of independent SWIFT builds ("profiles"), each with
its own git worktree, branch, configure flags and build products, so
switching between them never needs a rebuild. It sets up run directories
with a parameter file checked against SWIFT's own parser, generates
output-time lists, and analyses what SWIFT writes: where the time goes in
every step, timestep evolution, gravity accuracy, task timings, MPI
communication and scaling.

Modes can be chained, so a whole development loop is one command:

.. code-block:: bash

   swift-cli update config --debug-build make -j 32

.. _SWIFT: https://swift.strw.leidenuniv.nl/

.. INTRO_END_LABEL

Full documentation is in the docs_ directory.

.. _docs: https://github.com/WillJRoper/swiftsim-cli/tree/main/docs/source

Requirements
------------

.. REQS_START_LABEL

Python 3.10 or later and ``git``. Building and running SWIFT itself needs
SWIFT's own dependencies (see the `SWIFT documentation`_).

The Python dependencies are installed with ``swift-cli``:

+ ``numpy``, ``astropy`` and ``scipy``, for cosmology and unit conversions.
+ ``h5py`` and ``swiftsimio``, to read initial conditions and snapshots.
+ ``ruamel.yaml``, to read and write parameter and profile files.
+ ``matplotlib``, ``pandas`` and ``tqdm``, for the analyses.
+ ``tree-sitter`` and ``tree-sitter-c``, to read timer and task names from
  the SWIFT source.

.. _SWIFT documentation: https://swift.strw.leidenuniv.nl/docs/index.html

.. REQS_END_LABEL

Installing
----------

.. INSTALL_START_LABEL

``swift-cli`` can be installed with ``pip``:

.. code-block:: bash

   pip install swiftsim-cli

or from source:

.. code-block:: bash

   git clone https://github.com/WillJRoper/swiftsim-cli.git
   cd swiftsim-cli
   pip install .

Check the installation with ``swift-cli --version``.

.. INSTALL_END_LABEL

Usage example
-------------

.. USAGE_START_LABEL

.. code-block:: bash

   # Clone SWIFT into ./swiftsim and create the "default" profile
   # (or use an existing clone: swift-cli init --swift-path ~/swiftsim)
   swift-cli init

   # Configure and build it
   swift-cli config --debug-build make -j 16

   # Add a second profile with its own branch, flags and build
   swift-cli profile --new zoom --upstream origin/zoom \
       --configure-flags="--with-hydro=sphenix" --build -j 16

   # Set up a run directory for some initial conditions
   swift-cli new -o runs/test --ics ics.hdf5

   # Output times every 0.5 in redshift from z = 10
   swift-cli output-times -P runs/test/params.yml --first-snap-z 10 \
       --delta-z 0.5 --output-list runs/test/output_list.txt

   # After running SWIFT with -v 1, see where the time went
   swift-cli analyse log runs/test/swift.log -o runs/test

In the above:

+ ``init`` runs once. Every later profile is another worktree of the same
  SWIFT clone, under ``~/.swiftsim-cli/worktrees`` by default.
+ ``config``, ``make``, ``new`` and ``analyse`` act on the current profile;
  ``swift-cli --profile zoom make -j 16`` targets another one without
  switching.
+ ``new`` picks a parameter template from the initial conditions, fills in
  paths, units, start time and softenings, and checks the result against
  SWIFT's parser before anything is written.
+ Every mode has its own help: ``swift-cli <mode> --help``.

.. USAGE_END_LABEL

Citing
------

.. CITING_START_LABEL

If you use ``swift-cli`` in your research, please cite:

.. code-block:: bibtex

   @software{swiftsim_cli,
     author = {Roper, W. J.},
     title = {swiftsim-cli: A command-line toolkit for SWIFT simulations},
     url = {https://github.com/WillJRoper/swiftsim-cli},
     version = {1.0.0},
     year = {2026}
   }

.. CITING_END_LABEL

Community
---------

.. COMMUNITY_START_LABEL

Contributions are welcome! Start with the `contributing guide`_; bugs and
feature requests go to the `issue tracker`_.

``swift-cli`` is licensed under `GPL-3.0-or-later`_.

.. _contributing guide: https://github.com/WillJRoper/swiftsim-cli/blob/main/CONTRIBUTING.md
.. _issue tracker: https://github.com/WillJRoper/swiftsim-cli/issues
.. _GPL-3.0-or-later: https://github.com/WillJRoper/swiftsim-cli/blob/main/LICENSE

.. COMMUNITY_END_LABEL
