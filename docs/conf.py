# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import pathlib
import shutil
import subprocess, re

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../minimel"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "minimel"
copyright = "2023, Benno Kruit"
author = "Benno Kruit"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxcontrib.apidoc",
    "sphinxcontrib.ansi",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx_mdinclude",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

# Apidoc and autodoc config
apidoc_module_dir = "../minimel"
apidoc_toc_file = False
apidoc_module_first = True
apidoc_separate_modules = True
apidoc_extra_args = ["-F"]
autodoc_default_flags = ["members"]
autodoc_member_order = "bysource"
autosummary_generate = True

# Napoleon config
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True

# Intersphinx config
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
# intersphinx_disabled_reftypes = ["*"]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

### Copy notebooks
nbdir = pathlib.Path('notebooks')
nbdir.mkdir(exist_ok=True)
for nb in pathlib.Path('..').glob('*.ipynb'):
    shutil.copy(nb, nbdir)

### CLI docs
print("Creating CLI docs...")

def run(cmd):
    out = subprocess.run(cmd.split(), capture_output=True).stdout.decode()
    return "".join("\n\t" + l for l in out.splitlines())


with open("cli.rst", "w") as fw:
    cli_doc = run("minimel -h")
    print(
        f"""
Command Line Interface
======================

.. ansi-block::

    {cli_doc}

""",
        file=fw,
    )
    for subcmd in next(re.finditer("\{[^}]+\}", cli_doc)).group()[1:-1].split(","):
        print(subcmd)
        sub_doc = run(f"minimel {subcmd} -h")
        print(
            f"""
{subcmd}
{"^"*len(subcmd)}

.. ansi-block::

    {sub_doc}

""",
            file=fw,
        )
