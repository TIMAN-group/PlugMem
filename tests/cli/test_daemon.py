"""Tests for plugmem.cli.daemon — PID file lifecycle + start/stop state machine.

Uses a sleep subprocess as a stand-in for uvicorn so the tests don't need
the real PlugMem service. Tests `_build_uvicorn_cmd` separately (pure
function, no IO).

Note on zombies: in production the CLI process exits after Popen, so the
spawned daemon is reparented to init which reaps it on death. In tests
the test process stays alive AND is the spawner, so zombies linger until
we explicitly reap them. `_reap` below does that.
"""
from __future__ import annotations

import errno
import os
import sys
import time
from pathlib import Path

import pytest

from plugmem.cli.config import PlugmemConfig
from plugmem.cli.daemon import (
    DaemonError,
    _build_uvicorn_cmd,
    _is_running,
    _read_pid,
    daemon_status,
    start_daemon,
    stop_daemon,
)


@pytest.fixture(autouse=True)
def isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Force XDG_STATE_HOME into a temp dir so tests don't touch the
    user's real ~/.local/state/plugmem."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def _sleep_cmd_factory(seconds: int = 30):
    """Return a callable that mimics _build_uvicorn_cmd but spawns a long-sleep."""
    def _build(_cfg):
        return [sys.executable, "-c", f"import time; time.sleep({seconds})"]
    return _build


def _reap(pid: int) -> None:
    """Reap a zombie child non-blocking; ignore if already gone or not ours."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except OSError as e:
        if e.errno not in (errno.ECHILD, errno.ESRCH):
            raise


def test_build_cmd_uses_python_module() -> None:
    cmd = _build_uvicorn_cmd(PlugmemConfig())
    assert cmd[0] == sys.executable
    assert "uvicorn" in cmd
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--port" in cmd and "8080" in cmd


def test_status_when_no_pid_file() -> None:
    state = daemon_status(PlugmemConfig())
    assert state["running"] is False
    assert state["pid"] is None
    assert state["health"] is None


def test_stop_when_not_running_returns_false() -> None:
    assert stop_daemon() is False


def test_start_writes_pid_then_stop_clears_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "plugmem.cli.daemon._build_uvicorn_cmd",
        _sleep_cmd_factory(seconds=30),
    )

    cfg = PlugmemConfig()
    pid = start_daemon(cfg, wait_for_health=False)
    try:
        assert pid > 0
        assert _read_pid() == pid
        assert _is_running(pid)
        # Status should reflect running.
        state = daemon_status(cfg)
        assert state["running"] is True
        assert state["pid"] == pid
    finally:
        stopped = stop_daemon(timeout=5.0)
        assert stopped is True
        # PID file cleared.
        assert _read_pid() is None
        # Reap the zombie so /proc no longer reports the PID; then assert.
        _reap(pid)
        time.sleep(0.05)
        assert not _is_running(pid)


def test_start_refuses_when_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "plugmem.cli.daemon._build_uvicorn_cmd",
        _sleep_cmd_factory(seconds=30),
    )

    cfg = PlugmemConfig()
    pid = start_daemon(cfg, wait_for_health=False)
    try:
        with pytest.raises(DaemonError, match="already running"):
            start_daemon(cfg, wait_for_health=False)
    finally:
        stop_daemon(timeout=5.0)
        _reap(pid)


def test_stale_pid_file_is_cleared_before_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "plugmem.cli.daemon._build_uvicorn_cmd",
        _sleep_cmd_factory(seconds=10),
    )
    # Plant a stale PID (a process number that doesn't exist).
    from plugmem.cli.config import default_pid_file
    pid_file = default_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("999999")  # almost certainly not a real PID

    cfg = PlugmemConfig()
    pid = start_daemon(cfg, wait_for_health=False)
    try:
        # The new PID should differ from the stale one.
        assert pid != 999999
        assert _read_pid() == pid
    finally:
        stop_daemon(timeout=5.0)
        _reap(pid)


def test_status_health_field_none_when_service_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "plugmem.cli.daemon._build_uvicorn_cmd",
        _sleep_cmd_factory(seconds=10),
    )

    cfg = PlugmemConfig()
    cfg.service.port = 1  # nothing listens on port 1
    pid = start_daemon(cfg, wait_for_health=False)
    try:
        # Give the subprocess a beat to actually exist.
        time.sleep(0.1)
        state = daemon_status(cfg)
        assert state["running"] is True
        assert state["health"] is None
    finally:
        stop_daemon(timeout=5.0)
        _reap(pid)
