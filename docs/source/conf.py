# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Quantax"
copyright = "2024, Ao Chen"
author = "Ao Chen, Christopher Roth"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary"]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_member_order = "bysource"
autosummary_generate = True
autosummary_generate_overwrite = False
default_role = "py:obj"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
#html_static_path = ["_static"]
html_use_relative_urls = True
html_baseurl = 'https://chenao-phys.github.io/quantax/'
