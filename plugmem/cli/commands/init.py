"""`plugmem init` — interactive setup wizard.

Delegates to `plugmem.cli.wizard.run_wizard` which walks LLM,
embedding, and service sections, runs a final live `/health` probe,
and writes the TOML config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from plugmem.cli.wizard import run_wizard


def init_cmd(
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing config without prompting."
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file (default: $XDG_CONFIG_HOME/plugmem/config.toml).",
    ),
) -> None:
    code = run_wizard(config_path, force=force)
    raise typer.Exit(code=code)
