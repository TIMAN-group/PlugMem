"""TOML config schema + IO for the plugmem CLI.

Config lives at `$XDG_CONFIG_HOME/plugmem/config.toml` (or
`~/.config/plugmem/config.toml`). Sections map 1:1 to the env vars the
existing FastAPI app reads via `plugmem/api/dependencies.py` — the CLI
just translates config → env-var dict before exec'ing uvicorn.

Env vars override config at runtime (12-factor); useful for one-off
overrides without editing the file (`LLM_API_KEY=... plugmem start`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

# tomli ships with stdlib as `tomllib` on 3.11+; fall back for 3.10.
if sys.version_info >= (3, 11):
    import tomllib as toml_reader  # type: ignore[import-not-found]
else:
    import tomli as toml_reader  # type: ignore[import-not-found]

import tomli_w


# ── Default paths (XDG) ──────────────────────────────────────────────


def _xdg(env_var: str, fallback: str) -> Path:
    raw = os.environ.get(env_var)
    base = Path(raw) if raw else Path.home() / fallback
    return base / "plugmem"


def default_config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config")


def default_config_path() -> Path:
    return default_config_dir() / "config.toml"


def default_data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / "chroma"


def default_state_dir() -> Path:
    return _xdg("XDG_STATE_HOME", ".local/state")


def default_pid_file() -> Path:
    return default_state_dir() / "plugmem.pid"


def default_log_file() -> Path:
    return default_state_dir() / "plugmem.log"


# ── Schema ──────────────────────────────────────────────────────────


class ServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""
    data_dir: str = ""  # filled in by `with_defaults_applied` if blank
    log_level: str = "INFO"
    token_usage_file: str = ""

    @field_validator("port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"port must be 1-65535, got {v}")
        return v


class LLMConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_retries: int = 5
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096


class EmbeddingConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = "nomic-embed-text"
    max_text_len: int = 8192


class PlugmemConfig(BaseModel):
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    def with_defaults_applied(self) -> "PlugmemConfig":
        """Fill in computed defaults (paths) without mutating self."""
        c = self.model_copy(deep=True)
        if not c.service.data_dir:
            c.service.data_dir = str(default_data_dir())
        return c


# ── IO ──────────────────────────────────────────────────────────────


def load_config(path: Optional[Path] = None) -> PlugmemConfig:
    """Load TOML config from `path` (defaults to the XDG location).

    Returns a `PlugmemConfig` populated with defaults if the file
    doesn't exist — letting callers always work against a config object,
    not Optional[Config].
    """
    p = path or default_config_path()
    if not p.exists():
        return PlugmemConfig()
    with open(p, "rb") as f:
        raw = toml_reader.load(f)
    return PlugmemConfig(**raw)


def save_config(cfg: PlugmemConfig, path: Optional[Path] = None) -> Path:
    """Write `cfg` to `path` (defaults to the XDG location). Returns the
    path written. Creates parent dirs as needed. Atomic via tmp + rename."""
    p = path or default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump()
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        tomli_w.dump(payload, f)
    tmp.replace(p)
    return p


# ── Env-var translation ─────────────────────────────────────────────

# Maps config dotted-path → env var name. Keys missing from the
# resolved config produce no env entry (unset; the FastAPI app uses
# its own default). Order is irrelevant — we build a flat dict.
_CONFIG_TO_ENV: Dict[str, str] = {
    "service.api_key": "PLUGMEM_API_KEY",
    "service.data_dir": "CHROMA_PATH",
    "service.token_usage_file": "TOKEN_USAGE_FILE",
    "llm.base_url": "LLM_BASE_URL",
    "llm.api_key": "LLM_API_KEY",
    "llm.model": "LLM_MODEL",
    "llm.max_retries": "LLM_MAX_RETRIES",
    "llm.temperature": "LLM_TEMPERATURE",
    "llm.top_p": "LLM_TOP_P",
    "llm.max_tokens": "LLM_MAX_TOKENS",
    "embedding.base_url": "EMBEDDING_BASE_URL",
    "embedding.api_key": "EMBEDDING_API_KEY",
    "embedding.model": "EMBEDDING_MODEL",
    "embedding.max_text_len": "EMBEDDING_MAX_TEXT_LEN",
}


def config_to_env(cfg: PlugmemConfig) -> Dict[str, str]:
    """Translate config → env-var dict for handing to subprocess /
    daemon launches. Empty / zero values are skipped so we don't
    overwrite the FastAPI app's internal defaults with a blank string."""
    cfg = cfg.with_defaults_applied()
    out: Dict[str, str] = {}
    payload = cfg.model_dump()
    for dotted, env_name in _CONFIG_TO_ENV.items():
        section, field = dotted.split(".")
        v = payload[section][field]
        if v in (None, "", 0, 0.0):
            # 0 is a meaningful value for some numeric fields (temperature),
            # but we treat it as "use default" since the wizard never
            # produces 0 for those — defaults are the same.
            continue
        out[env_name] = str(v)
    return out


def apply_env_overrides(cfg: PlugmemConfig, env: Optional[Dict[str, str]] = None) -> PlugmemConfig:
    """Return a copy of `cfg` with values overridden by env vars.

    Used at start time so `LLM_API_KEY=... plugmem start` works without
    editing the config file.
    """
    src = env if env is not None else os.environ
    payload = cfg.model_dump()
    # Reverse the map: env var → (section, field).
    for dotted, env_name in _CONFIG_TO_ENV.items():
        if env_name in src and src[env_name] != "":
            section, field = dotted.split(".")
            # Cast to the existing field's type. Pydantic re-validates on
            # construction below.
            current = payload[section][field]
            raw = src[env_name]
            try:
                if isinstance(current, bool):
                    cast: Any = raw.lower() in ("1", "true", "yes")
                elif isinstance(current, int):
                    cast = int(raw)
                elif isinstance(current, float):
                    cast = float(raw)
                else:
                    cast = raw
            except ValueError:
                cast = raw
            payload[section][field] = cast
    return PlugmemConfig(**payload)
