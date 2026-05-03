"""Daemon process management.

`start_daemon` spawns uvicorn as a session-detached subprocess (the
"easy daemon" approach: setsid + DEVNULL stdin + log-file stdout/stderr,
then the parent exits, leaving the child orphaned to init). This is
simpler than python-daemon's double-fork dance and produces the same
observable behavior for our use case (background service, PID file, log
file, survives the CLI exiting).

Foreground mode bypasses the fork entirely via `os.execvpe`, replacing
the CLI process with uvicorn directly so Ctrl-C produces a clean exit.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import requests

from plugmem.cli.config import (
    PlugmemConfig,
    config_to_env,
    default_log_file,
    default_pid_file,
)


class DaemonError(Exception):
    """Raised by daemon ops on user-facing errors (already-running, etc.)."""


# ── PID management ──────────────────────────────────────────────────


def _read_pid() -> Optional[int]:
    pid_file = default_pid_file()
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_running(pid: int) -> bool:
    """True if a process with `pid` is alive (no permission required —
    SIGNAL 0 just probes existence)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still "running".
        return True
    except OSError:
        return False


def _clear_pid_file() -> None:
    try:
        default_pid_file().unlink(missing_ok=True)
    except OSError:
        pass


# ── Status ──────────────────────────────────────────────────────────


def daemon_status(cfg: PlugmemConfig) -> Dict[str, Any]:
    """Return a dict describing the daemon's current state."""
    pid = _read_pid()
    if pid is None or not _is_running(pid):
        return {
            "running": False,
            "pid": None,
            "host": cfg.service.host,
            "port": cfg.service.port,
            "health": None,
        }
    return {
        "running": True,
        "pid": pid,
        "host": cfg.service.host,
        "port": cfg.service.port,
        "health": _quick_health(cfg),
    }


def _quick_health(cfg: PlugmemConfig) -> Optional[Dict[str, Any]]:
    url = f"http://{cfg.service.host}:{cfg.service.port}/health"
    try:
        resp = requests.get(url, timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


# ── Start ───────────────────────────────────────────────────────────


def start_daemon(
    cfg: PlugmemConfig,
    *,
    foreground: bool = False,
    wait_for_health: bool = True,
    health_timeout: float = 15.0,
) -> int:
    """Launch the service.

    Returns the PID of the spawned uvicorn process (or 0 in foreground
    mode, since execvpe doesn't return). Raises DaemonError if a daemon
    is already running and we're in daemon mode.
    """
    existing = _read_pid()
    if existing and _is_running(existing) and not foreground:
        raise DaemonError(f"Daemon already running with PID {existing}")
    if existing and not _is_running(existing):
        # Stale PID file from a crashed run.
        _clear_pid_file()

    log_file = default_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file = default_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(config_to_env(cfg))

    cmd = _build_uvicorn_cmd(cfg)

    if foreground:
        os.execvpe(cmd[0], cmd, env)  # noqa: S606 — intentional process replacement
        return 0  # unreachable

    log_fh = open(log_file, "ab")
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from CLI's controlling terminal
        )
    finally:
        # Close the parent's copy of the fd; the child has its own dup'd fd.
        log_fh.close()

    pid_file.write_text(str(proc.pid))

    if wait_for_health:
        if not _wait_for_health(cfg, timeout=health_timeout, expect_pid=proc.pid):
            raise DaemonError(
                f"Daemon spawned (PID {proc.pid}) but /health did not respond "
                f"within {health_timeout:.0f}s. Check {log_file}."
            )

    return proc.pid


def _build_uvicorn_cmd(cfg: PlugmemConfig) -> list[str]:
    """Extracted for testability — tests patch this to inject a sleep."""
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "plugmem.api.app:app",
        "--host",
        cfg.service.host,
        "--port",
        str(cfg.service.port),
        "--log-level",
        cfg.service.log_level.lower(),
    ]


def _wait_for_health(cfg: PlugmemConfig, *, timeout: float, expect_pid: int) -> bool:
    url = f"http://{cfg.service.host}:{cfg.service.port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_running(expect_pid):
            return False
        try:
            resp = requests.get(url, timeout=1.5)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.3)
    return False


# ── Stop ────────────────────────────────────────────────────────────


def stop_daemon(*, timeout: float = 10.0) -> bool:
    """Stop the running daemon.

    Returns True if a daemon was running and we asked it to stop;
    False if no daemon was running. Always cleans up the PID file.
    """
    pid = _read_pid()
    if pid is None or not _is_running(pid):
        _clear_pid_file()
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_pid_file()
        return True

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_running(pid):
            _clear_pid_file()
            return True
        time.sleep(0.2)

    # Process didn't honor SIGTERM in time — escalate.
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # Wait a moment for the kernel to reap.
    time.sleep(0.5)
    _clear_pid_file()
    return True
