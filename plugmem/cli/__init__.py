"""Plugmem command-line interface.

Re-exports the Typer `app` so `plugmem.cli:app` is the canonical entry
point referenced from pyproject.toml's [project.scripts].
"""
from plugmem.cli.main import app

__all__ = ["app"]
