"""`plugmem stop` — terminate the daemon by PID."""
from __future__ import annotations

import typer

from plugmem.cli.daemon import stop_daemon
from plugmem.cli.wizard.ui import info, success


def stop_cmd() -> None:
    stopped = stop_daemon()
    if stopped:
        success("Daemon stopped.")
    else:
        info("No daemon was running.")
