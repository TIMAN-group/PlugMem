"""Top-level Typer app and subcommand registration.

Each subcommand has its own module under `plugmem.cli.commands`. This
file just wires them together and provides the global help text. Any
shared CLI infrastructure (rich console, exit codes, error handling)
lives in `plugmem.cli.console`.
"""
from __future__ import annotations

import typer

from plugmem.cli.commands import (
    health,
    init,
    logs,
    restart,
    start,
    status,
    stop,
)

app = typer.Typer(
    name="plugmem",
    help=(
        "PlugMem — pluggable long-term memory for LLM agents.\n\n"
        "Run `plugmem init` to set up your local instance, then "
        "`plugmem start` to launch the service."
    ),
    no_args_is_help=True,
    add_completion=False,
)

# Register each subcommand. The order here determines display order in `--help`.
app.command("init", help="Interactive setup wizard for LLM, embedding, and service settings.")(init.init_cmd)
app.command("start", help="Start the PlugMem service (daemonized by default).")(start.start_cmd)
app.command("stop", help="Stop the running PlugMem daemon.")(stop.stop_cmd)
app.command("restart", help="Restart the PlugMem daemon.")(restart.restart_cmd)
app.command("status", help="Show daemon status, PID, port, and last health probe.")(status.status_cmd)
app.command("logs", help="Print or tail the daemon log.")(logs.logs_cmd)
app.command("health", help="One-shot health check against the running service.")(health.health_cmd)
