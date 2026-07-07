"""`plugmem restart` — stop + start."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from plugmem.cli.config import apply_env_overrides, default_config_path, load_config
from plugmem.cli.daemon import DaemonError, start_daemon, stop_daemon
from plugmem.cli.wizard.ui import error, success


def restart_cmd(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file."
    ),
) -> None:
    path = config_path or default_config_path()
    if not path.exists():
        error(f"No config at {path}. Run `plugmem init` first.")
        raise typer.Exit(code=1)

    stopped = stop_daemon()
    cfg = apply_env_overrides(load_config(path))
    try:
        pid = start_daemon(cfg)
    except DaemonError as e:
        error(str(e))
        raise typer.Exit(code=1)

    verb = "Restarted" if stopped else "Started"
    success(
        f"{verb} daemon (PID {pid}) on http://{cfg.service.host}:{cfg.service.port}"
    )
