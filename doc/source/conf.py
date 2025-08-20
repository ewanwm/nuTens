# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nuTens'
copyright = '2025, Ewan Miller'
author = 'Ewan Miller'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 'sphinx.ext.autodoc', 'breathe', "sphinx.ext.graphviz", "sphinx_tabs.tabs" ]

breathe_projects = {"nuTens": "doxygen/xml"}
breathe_default_project = "nuTens"
breathe_default_members = ('members', 'undoc-members')

templates_path = ['_templates']
exclude_patterns = []

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

html_logo = "nuTens-logo-small.png"

html_theme_options = {
    'logo_only': True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
