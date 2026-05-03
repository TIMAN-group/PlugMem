"""`plugmem status` — daemon status report."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from plugmem.cli.config import default_config_path, load_config
from plugmem.cli.daemon import daemon_status
from plugmem.cli.wizard.ui import console, error


def status_cmd(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file."
    ),
) -> None:
    path = config_path or default_config_path()
    if not path.exists():
        error(f"No config at {path}. Run `plugmem init` first.")
        raise typer.Exit(code=1)

    cfg = load_config(path)
    state = daemon_status(cfg)

    if state["running"]:
        console.print(f"[green]●[/green] running (PID {state['pid']})")
    else:
        console.print("[red]○[/red] stopped")
    console.print(f"  url:  http://{state['host']}:{state['port']}")

    health = state.get("health")
    if state["running"] and health:
        flags = ["llm_available", "embedding_available", "chroma_available"]
        for f in flags:
            ok = health.get(f, False)
            mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
            console.print(f"  {mark} {f}")
    elif state["running"]:
        console.print("  [yellow]![/yellow] /health did not respond")
