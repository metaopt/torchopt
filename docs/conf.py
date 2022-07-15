# Copyright 2022 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Configuration file for the Sphinx documentation builder."""
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys

import inspect
import os
import pathlib
import sys

import sphinx_rtd_theme
import sphinxcontrib.katex as katex
import torchopt



HERE = pathlib.Path(__file__).absolute().parent
PROJECT_ROOT = HERE.parent



def get_version() -> str:
    sys.path.insert(0, str(PROJECT_ROOT / 'torchopt'))
    import version  # noqa
    return version.__version__


# -- Project information -----------------------------------------------------

project = "TorchOpt"
copyright = "2022 MetaOPT Team"
author = "TorchOpt Contributors"

# The full version, including alpha/beta/rc tags
release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.spelling',
    'sphinxcontrib.katex',
    'sphinx_autodoc_typehints',
    'sphinx_rtd_theme',
    'myst_nb',  # This is used for the .ipynb notebooks
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = [".rst"]

# The root document.
root_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
spelling_exclude_patterns = [""]
spelling_word_list_filename='spelling_wordlist.txt'


# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': True,
    'exclude-members': '__repr__, __str__, __weakref__',
}

# -- Options for bibtex -----------------------------------------------------

bibtex_bibfiles = ['refs.bib']

# -- Options for myst -------------------------------------------------------

nb_execution_mode = 'force'
nb_execution_allow_errors = False

# -- Options for katex ------------------------------------------------------

# See: https://sphinxcontrib-katex.readthedocs.io/en/0.4.1/macros.html
latex_macros = r"""
    \def \d              #1{\operatorname{#1}}
"""

# Translate LaTeX macros to KaTeX and add to options for HTML builder
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = 'macros: {' + katex_macros + '}'

# Add LaTeX macros for LATEX builder
latex_elements = {'preamble': latex_macros}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/images/logod-05.png"


def setup(app):
    app.add_js_file("js/copybutton.js")
    app.add_css_file("css/style.css")


# -- Source code links -------------------------------------------------------


def linkcode_resolve(domain, info):
    """Resolve a GitHub URL corresponding to Python object."""
    if domain != 'py':
        return None

    try:
        mod = sys.modules[info['module']]
    except ImportError:
        return None

    obj = mod
    try:
        for attr in info['fullname'].split('.'):
            obj = getattr(obj, attr)
    except AttributeError:
        return None
    else:
        obj = inspect.unwrap(obj)

    try:
        filename = inspect.getsourcefile(obj)
    except TypeError:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    
    # TODO(slebedev): support tags after we release an initial version.
    return 'https://github.com/metaopt/TorchOpt/tree/main/TorchOpt/%s#L%d#L%d' % (
        os.path.relpath(filename, start=os.path.dirname(torchopt.__file__)), 
        lineno, 
        lineno + len(source) - 1
    )


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'https://docs.python.org/3/': None}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
# todo_include_todos = False
