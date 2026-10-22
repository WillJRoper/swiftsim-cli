"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("."))
import swiftsim_cli


# -- Project information -----------------------------------------------------

project = "swiftsim-cli"
copyright = "2026, W. J. Roper"
author = "W. J. Roper"

# The full version, including alpha/beta/rc tags
release = swiftsim_cli.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []

source_suffix = [".rst"]
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path: list[str] = []


# -- Options for automatic API doc

# The package uses Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render "Attributes:" sections as fields, so dataclass fields are not
# documented twice
napoleon_use_ivar = True

autodoc_member_order = "bysource"
autodoc_default_flags = ["members"]
autosummary_generate = True

# must be outside run_apidoc definition to be set successfully:
os.environ["SPHINX_APIDOC_OPTIONS"] = "members,undoc-members,show-inheritance"


def run_apidoc(_):
    from sphinx.ext.apidoc import main

    cur_dir = os.path.abspath(os.path.dirname(__file__))

    api_doc_dir = os.path.join(cur_dir, "modules")
    module = os.path.join(cur_dir, "../..", "src", "swiftsim_cli")
    ignore = [
        os.path.join(module, "_version.py"),
        os.path.join(module, "templates"),
    ]

    main(["-M", "-f", "-e", "-T", "-d 0", "-o", api_doc_dir, module, *ignore])


def setup(app):
    app.connect("builder-inited", run_apidoc)


intersphinx_mapping = dict(
    python=("https://docs.python.org/3/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    astropy=("https://docs.astropy.org/en/stable/", None),
    h5py=("https://docs.h5py.org/en/latest/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    swiftsimio=("https://swiftsimio.readthedocs.io/en/latest/", None),
)
