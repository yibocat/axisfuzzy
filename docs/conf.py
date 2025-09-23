# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib

# -- Path setup --------------------------------------------------------------
# Add the project root to the Python path so Sphinx can find the axisfuzzy package
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'AxisFuzzy'
copyright = '2025, yibocat'
author = 'yibocat'

# -- Version information -----------------------------------------------------
# Read version from pyproject.toml
pyproject_path = os.path.join(os.path.abspath('..'), 'pyproject.toml')
try:
    with open(pyproject_path, 'rb') as f:
        data = tomllib.load(f)
        release = data['project']['version']
except Exception:
    release = '0.0.0'

version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',                  # Markdown support
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',          # Google/NumPy style docstrings
    'sphinx.ext.viewcode',          # Add source code links
    'sphinx_copybutton',            # Copy button for code blocks
    'sphinx_design',                # Provide design components (cards, grids, etc.)
    'sphinx_autodoc_typehints',     # Pretty render Python 3 type hints in autodoc output
    'sphinx.ext.mathjax',           # Math support
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx_tabs.tabs',             # Add tab functionality
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Support both .rst and .md files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'    # The name of the root document (for example, 'index.rst' or 'index.md').

# -- Extension configuration -------------------------------------------------
# Detailed settings for each Sphinx extension.

# MyST Parser configuration:
# Enable specific MyST extensions (for example, for colon-fenced code blocks, allowing nested directives in Markdown).
# If you need to use ::: in Markdown to create admonitions or other block-level directives, uncomment it.

# myst_enable_extensions = ["colon_fence"]


# Autodoc settings
# Default options for the 'autodoc' extension.
# These options can be overridden by individual directives (e.g., using :members:).
autodoc_default_options = {
    'members': False,               # By default, all members are not automatically displayed
    'undoc-members': False,         # Do not display members without docstrings
    'inherited-members': True,      # Document members inherited from the base class
    'show-inheritance': True,       # Display base class list
    'member-order': 'alphabetical', # Member sorting method: Alphabetical order. 
                                    # The available options are 'bysource' (by source code order), 'alphabetical' (by alphabetical order), and 'groupwise' (by type grouping).
}

autodoc_typehints = "signature"             # Put type hints in the signature, not in the description
autodoc_typehints_format = "short"          # Simplify type hints, avoid full paths
python_use_unqualified_type_names = True    # Resolve duplicate warnings caused by .pyi and dynamic injection

# Autosummary settings
autosummary_generate = True                 # Automatically generate stub pages for the autosummary directive during the build process.

# Code highlighting
pygments_style = "sphinx"                   # The style name used for Pygments syntax highlighting in source code.
# pygments_dark_style = "monokai"           # Styles in dark mode

# Copy button configuration
# A regular expression used to remove text from prompts when copying from code blocks.
# This prevents copying prompt symbols such as '>>>' or '$' along with the code.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- HTML output options -----------------------------------------------------
# Other theme options (e.g., "furo", "pydata_sphinx_theme", "shibuya") can be tried once dependencies stabilize
html_theme = "sphinx_rtd_theme"
# Paths containing custom static files (such as CSS, JavaScript).
html_static_path = ['_static']

# Basic theme options for sphinx_rtd_theme
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# Do not display the "View Source" link at the bottom of the page.
# This is usually more suitable for clean public documents, avoiding users seeing the original rst files.
# html_show_sourcelink = False

# Custom page title
html_title = f"{project} Documentation"

# Add custom CSS

# html_css_files = [
#     'custom.css',
# ]
