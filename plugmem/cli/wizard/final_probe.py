"""Final probe: launch uvicorn briefly, hit /health, confirm everything wires up.

Used at the end of `plugmem init` before writing the config — gives the
user a real green light instead of "we wrote some TOML, hope it works."
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Tuple

import requests

from plugmem.cli.config import PlugmemConfig, config_to_env


def run_final_probe(cfg: PlugmemConfig, *, timeout: float = 30.0) -> Tuple[bool, str]:
    """Spawn uvicorn with cfg's env, poll /health, return (ok, message).

    Always cleans up the subprocess. Never raises — every failure mode
    becomes a returned message.
    """
    env = os.environ.copy()
    env.update(config_to_env(cfg))

    cmd = [
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

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        ok, msg = _poll_health(cfg, timeout=timeout, proc=proc)
    finally:
        _terminate(proc)

    return ok, msg


def _poll_health(
    cfg: PlugmemConfig, *, timeout: float, proc: subprocess.Popen
) -> Tuple[bool, str]:
    url = f"http://{cfg.service.host}:{cfg.service.port}/health"
    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        # Bail early if the subprocess died.
        if proc.poll() is not None:
            stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace")
            tail = stderr.splitlines()[-5:] if stderr else []
            return False, (
                "Service exited before responding. "
                f"Last log lines:\n  " + "\n  ".join(tail)
            )
        try:
            resp = requests.get(url, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                missing = [
                    k for k in ("llm_available", "embedding_available", "chroma_available")
                    if not data.get(k, False)
                ]
                if missing:
                    return False, (
                        "Service responded but the following are not available: "
                        + ", ".join(missing)
                    )
                return True, "All checks passed."
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(0.5)
    return False, f"Timed out after {timeout:.0f}s. Last error: {last_err}"


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        pass
