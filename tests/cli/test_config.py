"""Tests for plugmem.cli.config — TOML round-trip + env-var translation."""
from __future__ import annotations

from pathlib import Path

import pytest

from plugmem.cli.config import (
    PlugmemConfig,
    apply_env_overrides,
    config_to_env,
    default_config_path,
    default_data_dir,
    load_config,
    save_config,
)


def test_load_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.service.port == 8080
    assert cfg.llm.base_url == ""
    assert cfg.embedding.model == "nomic-embed-text"


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    cfg = PlugmemConfig()
    cfg.service.api_key = "sk-test"
    cfg.service.port = 9090
    cfg.llm.base_url = "http://ollama.local/v1"
    cfg.llm.model = "qwen2.5:14b"
    cfg.embedding.model = "nomic-embed-text"

    p = tmp_path / "config.toml"
    written = save_config(cfg, p)
    assert written == p

    loaded = load_config(p)
    assert loaded.service.api_key == "sk-test"
    assert loaded.service.port == 9090
    assert loaded.llm.base_url == "http://ollama.local/v1"
    assert loaded.llm.model == "qwen2.5:14b"


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    p = tmp_path / "deep" / "nested" / "config.toml"
    save_config(PlugmemConfig(), p)
    assert p.exists()


def test_save_atomic_via_tmp_rename(tmp_path: Path) -> None:
    """The .tmp file should be cleaned up after the rename."""
    p = tmp_path / "config.toml"
    save_config(PlugmemConfig(), p)
    assert p.exists()
    assert not (tmp_path / "config.toml.tmp").exists()


def test_invalid_port_rejected() -> None:
    with pytest.raises(Exception):  # pydantic ValidationError
        PlugmemConfig(service={"port": 0})  # type: ignore[arg-type]


def test_config_to_env_translation() -> None:
    cfg = PlugmemConfig()
    cfg.service.api_key = "key"
    cfg.llm.base_url = "http://l"
    cfg.llm.model = "m"
    cfg.embedding.base_url = "http://e"
    cfg.embedding.model = "em"

    env = config_to_env(cfg)
    assert env["PLUGMEM_API_KEY"] == "key"
    assert env["LLM_BASE_URL"] == "http://l"
    assert env["LLM_MODEL"] == "m"
    assert env["EMBEDDING_BASE_URL"] == "http://e"
    assert env["EMBEDDING_MODEL"] == "em"
    # Default-applied: data_dir gets filled even if user didn't set it.
    assert "CHROMA_PATH" in env


def test_config_to_env_skips_empty_strings() -> None:
    cfg = PlugmemConfig()  # everything default-empty/zero
    env = config_to_env(cfg)
    # base_url, api_key are blank → not emitted
    assert "LLM_BASE_URL" not in env
    assert "LLM_API_KEY" not in env
    # Numeric defaults that aren't 0 are emitted
    assert env["LLM_MAX_RETRIES"] == "5"
    assert env["LLM_MAX_TOKENS"] == "4096"


def test_apply_env_overrides_takes_precedence_over_config() -> None:
    cfg = PlugmemConfig()
    cfg.llm.api_key = "from-config"
    cfg.llm.max_tokens = 4096

    overridden = apply_env_overrides(cfg, {
        "LLM_API_KEY": "from-env",
        "LLM_MAX_TOKENS": "2048",
    })
    assert overridden.llm.api_key == "from-env"
    assert overridden.llm.max_tokens == 2048

    # Original cfg untouched.
    assert cfg.llm.api_key == "from-config"


def test_apply_env_overrides_ignores_blank_env() -> None:
    cfg = PlugmemConfig()
    cfg.llm.api_key = "keep"
    overridden = apply_env_overrides(cfg, {"LLM_API_KEY": ""})
    assert overridden.llm.api_key == "keep"


def test_default_paths_use_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert default_config_path() == tmp_path / "cfg" / "plugmem" / "config.toml"
    assert default_data_dir() == tmp_path / "data" / "plugmem" / "chroma"
