# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'minimel'
copyright = '2022, Benno Kruit'
author = 'Benno Kruit'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_autodoc_typehints']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']



apidoc_module_dir = '../minimel'
apidoc_toc_file = False
apidoc_module_first = True
apidoc_separate_modules = True
apidoc_extra_args = ['-F']
autodoc_default_flags = ['members', 'titlesonly', 'show-inheritance']

# autosummary_generate = True

napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True