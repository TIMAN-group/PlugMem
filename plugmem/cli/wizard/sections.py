"""Wizard sections: LLM, embedding, service.

Each section mutates the relevant slice of `PlugmemConfig` in place and
returns True on success / False on user-skipped. Probes are validated
inline; on failure the user picks retry / edit / skip.
"""
from __future__ import annotations

import secrets
import socket
from pathlib import Path
from typing import List, Optional

from plugmem.cli.config import PlugmemConfig, default_data_dir
from plugmem.cli.wizard.probes import (
    OllamaInfo,
    detect_ollama,
    probe_embedding,
    probe_llm,
)
from plugmem.cli.wizard.ui import (
    error,
    header,
    info,
    prompt_action,
    prompt_choice,
    prompt_text,
    success,
    warn,
)

# ── Provider presets ────────────────────────────────────────────────

PROVIDER_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "needs_key": True,
        "default_llm_model": "gpt-4o-mini",
        "default_embed_model": "text-embedding-3-small",
    },
    "azure": {
        "base_url": "",  # user supplies
        "needs_key": True,
        "default_llm_model": "",
        "default_embed_model": "",
    },
    "vllm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "needs_key": False,
        "default_llm_model": "",
        "default_embed_model": "",
    },
    "other": {
        "base_url": "",
        "needs_key": False,
        "default_llm_model": "",
        "default_embed_model": "",
    },
}

PROVIDER_CHOICES = ["ollama", "openai", "azure", "vllm", "other"]


# ── LLM section ─────────────────────────────────────────────────────


def run_llm_section(cfg: PlugmemConfig, *, ollama: Optional[OllamaInfo] = None) -> bool:
    """Configure cfg.llm interactively. Returns True on success, False
    if the user skipped after repeated failures."""
    header("LLM endpoint")
    detected = ollama if ollama is not None else detect_ollama()

    while True:
        provider = _ask_provider(detected, kind="LLM")
        base_url, api_key, model = _ask_llm_fields(provider, detected, cfg.llm)

        info("Validating LLM endpoint…")
        ok, msg = probe_llm(base_url, api_key, model)
        if ok:
            success(f"LLM probe succeeded ({model} @ {base_url}).")
            cfg.llm.base_url = base_url
            cfg.llm.api_key = api_key
            cfg.llm.model = model
            return True

        error(f"Validation failed: {msg}")
        action = prompt_action()
        if action == "skip":
            warn("Skipping LLM configuration. You can re-run `plugmem init` later.")
            return False
        if action == "edit":
            # Loop back, re-using current values as defaults via cfg.llm
            cfg.llm.base_url = base_url
            cfg.llm.api_key = api_key
            cfg.llm.model = model
            continue
        # retry — same values, re-probe immediately.
        info("Retrying with the same values…")
        # tight loop — try the probe once more without prompting again.
        ok, msg = probe_llm(base_url, api_key, model)
        if ok:
            success(f"LLM probe succeeded on retry.")
            cfg.llm.base_url = base_url
            cfg.llm.api_key = api_key
            cfg.llm.model = model
            return True
        error(f"Still failing: {msg}")
        # Fall through to next loop iteration so user can edit/skip.


# ── Embedding section ───────────────────────────────────────────────


def run_embedding_section(
    cfg: PlugmemConfig, *, ollama: Optional[OllamaInfo] = None
) -> bool:
    header("Embedding endpoint")
    detected = ollama if ollama is not None else detect_ollama()

    while True:
        provider = _ask_provider(detected, kind="embedding")
        base_url, api_key, model = _ask_embedding_fields(provider, detected, cfg.embedding)

        info("Validating embedding endpoint…")
        ok, msg = probe_embedding(base_url, api_key, model)
        if ok:
            success(f"Embedding probe succeeded ({model} @ {base_url}).")
            cfg.embedding.base_url = base_url
            cfg.embedding.api_key = api_key
            cfg.embedding.model = model
            return True

        error(f"Validation failed: {msg}")
        action = prompt_action()
        if action == "skip":
            warn("Skipping embedding configuration.")
            return False
        if action == "edit":
            cfg.embedding.base_url = base_url
            cfg.embedding.api_key = api_key
            cfg.embedding.model = model
            continue
        info("Retrying with the same values…")
        ok, msg = probe_embedding(base_url, api_key, model)
        if ok:
            success("Embedding probe succeeded on retry.")
            cfg.embedding.base_url = base_url
            cfg.embedding.api_key = api_key
            cfg.embedding.model = model
            return True
        error(f"Still failing: {msg}")


# ── Shared prompt helpers (LLM + embedding both call) ───────────────


def _ask_provider(
    detected: Optional[OllamaInfo], *, kind: str
) -> str:
    if detected and detected.models:
        info(
            f"Detected Ollama at {detected.base_url} with "
            f"{len(detected.models)} model(s)."
        )
    elif detected:
        info(f"Detected Ollama at {detected.base_url} but no models pulled.")
    else:
        info("No Ollama detected on 127.0.0.1:11434.")

    default = "ollama" if detected and detected.models else "openai"
    return prompt_choice(
        f"{kind.title()} provider", PROVIDER_CHOICES, default=default
    )


def _ask_llm_fields(
    provider: str, detected: Optional[OllamaInfo], current
) -> tuple[str, str, str]:
    return _ask_endpoint_fields(provider, detected, current, kind="llm")


def _ask_embedding_fields(
    provider: str, detected: Optional[OllamaInfo], current
) -> tuple[str, str, str]:
    return _ask_endpoint_fields(provider, detected, current, kind="embedding")


def _ask_endpoint_fields(
    provider: str, detected: Optional[OllamaInfo], current, *, kind: str
) -> tuple[str, str, str]:
    if provider == "ollama":
        if not detected:
            warn("Ollama wasn't detected; falling back to manual entry.")
            return _manual_endpoint_fields(provider, current, kind=kind)
        base_url = detected.base_url
        # Ollama doesn't require an API key.
        api_key = ""
        model = _ask_model_from_list(detected.models, current_model=getattr(current, "model", ""))
        return base_url, api_key, model

    return _manual_endpoint_fields(provider, current, kind=kind)


def _manual_endpoint_fields(provider: str, current, *, kind: str) -> tuple[str, str, str]:
    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["other"])
    default_url = current.base_url or preset["base_url"] or None
    base_url = prompt_text("base_url", default=default_url)

    api_key_default = current.api_key or None
    if preset["needs_key"]:
        api_key = prompt_text("api_key", default=api_key_default, password=True)
    else:
        api_key = prompt_text(
            "api_key (leave blank if not required)",
            default=api_key_default,
            password=True,
            allow_empty=True,
        )

    model_default = current.model or (
        preset["default_llm_model"] if kind == "llm" else preset["default_embed_model"]
    ) or None
    model = prompt_text("model", default=model_default)
    return base_url, api_key, model


def _ask_model_from_list(models: List[str], *, current_model: str) -> str:
    if not models:
        return prompt_text("model")
    info("Pulled Ollama models:")
    for i, name in enumerate(models, start=1):
        info(f"  {i}. {name}")
    default = current_model if current_model in models else models[0]
    return prompt_choice("Model", choices=models, default=default)


# ── Service section ─────────────────────────────────────────────────


def run_service_section(cfg: PlugmemConfig) -> bool:
    """Configure host / port / data_dir / api_key / log_level.

    Doesn't require a probe (the final-probe step covers end-to-end
    verification). Returns True always — there's no skip path because
    the service section produces values with sensible defaults.
    """
    header("Service settings")

    cfg.service.host = prompt_text(
        "host", default=cfg.service.host or "127.0.0.1"
    )

    suggested_port = _next_free_port(cfg.service.port or 8080, host=cfg.service.host)
    if suggested_port != (cfg.service.port or 8080):
        warn(f"Port {cfg.service.port or 8080} is in use; suggesting {suggested_port}.")
    port_str = prompt_text("port", default=str(suggested_port))
    cfg.service.port = int(port_str)

    data_dir_default = cfg.service.data_dir or str(default_data_dir())
    data_dir = prompt_text("data_dir (chroma persistence)", default=data_dir_default)
    if not _ensure_writable_dir(Path(data_dir).expanduser()):
        warn(f"Could not create or write to {data_dir}; using anyway, may fail at start.")
    cfg.service.data_dir = data_dir

    if not cfg.service.api_key:
        cfg.service.api_key = secrets.token_hex(16)
        success(f"Generated service api_key: {cfg.service.api_key}")
        info("(Store this somewhere safe — clients use it as `X-API-Key`.)")
    else:
        info(f"Reusing existing service api_key: {cfg.service.api_key[:8]}…")

    cfg.service.log_level = prompt_choice(
        "log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=cfg.service.log_level or "INFO",
    )
    return True


def _next_free_port(start: int, *, host: str = "127.0.0.1", attempts: int = 20) -> int:
    """Return `start` if available; otherwise scan upward up to `attempts` times."""
    for offset in range(attempts):
        port = start + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return port
        except OSError:
            continue
    # All taken — return the original; the user can override.
    return start


def _ensure_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        # Trivial writeability test.
        probe = p / ".plugmem_probe"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False
