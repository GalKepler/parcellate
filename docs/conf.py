from __future__ import annotations

import importlib.metadata
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "parcellate"
author = "parcellate contributors"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

try:
    release = importlib.metadata.version("parcellate")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
autodoc_typehints = "description"
autodoc_preserve_defaults = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
}

todo_include_todos = True

html_theme = "furo"
html_title = "parcellate documentation"
html_static_path = ["_static"]
html_css_files: list[str] = []

python_display_short_literal_types = True
