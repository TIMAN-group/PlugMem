"""`plugmem start` — launch the PlugMem service."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from plugmem.cli.config import apply_env_overrides, default_config_path, load_config
from plugmem.cli.daemon import DaemonError, start_daemon
from plugmem.cli.wizard.ui import error, info, success


def start_cmd(
    foreground: bool = typer.Option(
        False, "--foreground", "-f", help="Run in foreground (skip daemonization)."
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file."
    ),
) -> None:
    path = config_path or default_config_path()
    if not path.exists():
        error(
            f"No config at {path}. Run `plugmem init` first."
        )
        raise typer.Exit(code=1)

    cfg = apply_env_overrides(load_config(path))

    if foreground:
        info(f"Starting in foreground on http://{cfg.service.host}:{cfg.service.port}")
        info("Press Ctrl-C to stop.")
        start_daemon(cfg, foreground=True)
        return  # not reached

    try:
        pid = start_daemon(cfg)
    except DaemonError as e:
        error(str(e))
        raise typer.Exit(code=1)

    success(
        f"Daemon started (PID {pid}) on http://{cfg.service.host}:{cfg.service.port}"
    )
