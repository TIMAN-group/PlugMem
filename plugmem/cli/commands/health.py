"""`plugmem health` — one-shot health check.

Exits non-zero if the service is unreachable or any `*_available` flag
is false. Useful in scripts / monitoring (`plugmem health || alert ...`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests
import typer

from plugmem.cli.config import default_config_path, load_config
from plugmem.cli.wizard.ui import console, error


HEALTH_FLAGS = ("llm_available", "embedding_available", "chroma_available")


def health_cmd(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file."
    ),
    timeout: float = typer.Option(
        5.0, "--timeout", help="Seconds to wait for the /health response."
    ),
) -> None:
    path = config_path or default_config_path()
    if not path.exists():
        error(f"No config at {path}. Run `plugmem init` first.")
        raise typer.Exit(code=1)

    cfg = load_config(path)
    url = f"http://{cfg.service.host}:{cfg.service.port}/health"

    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        error(f"Could not reach {url}: {e}")
        raise typer.Exit(code=2)

    if resp.status_code != 200:
        error(f"{url} returned HTTP {resp.status_code}")
        raise typer.Exit(code=2)

    try:
        data = resp.json()
    except ValueError:
        error(f"{url} returned non-JSON")
        raise typer.Exit(code=2)

    overall_ok = True
    for flag in HEALTH_FLAGS:
        ok = data.get(flag, False)
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {mark} {flag}")
        if not ok:
            overall_ok = False

    version = data.get("version", "?")
    status = data.get("status", "?")
    console.print(f"\nstatus: {status}, version: {version}")

    if not overall_ok:
        raise typer.Exit(code=1)
