"""Multi-LLM router — assigns different LLM clients to different operations.

Reads a YAML configuration that maps operation roles (structuring, retrieval,
reasoning, consolidation) to separate LLM endpoints/models.  Any role not
explicitly configured inherits from ``default``.

Example YAML (``llm_config.yaml``)::

    default:
      base_url: "http://localhost:8000/v1"
      api_key: "tok-xxx"
      model: "qwen-2.5-32b-instruct"

    structuring:
      model: "qwen-2.5-32b-instruct"

    retrieval:
      model: "gpt-4o-mini"

    reasoning:
      base_url: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4o"

    consolidation:
      # omitted → uses default

Environment variable references like ``${VAR}`` are expanded at load time.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from plugmem.clients.llm import LLMClient, OpenAICompatibleLLMClient

logger = logging.getLogger(__name__)

ROLES = ("default", "structuring", "retrieval", "reasoning", "consolidation")

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)}")


def _expand_env(value: str) -> str:
    """Replace ``${VAR}`` references with environment variable values."""
    if not isinstance(value, str):
        return value
    return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


def _expand_dict(d: dict) -> dict:
    return {k: _expand_env(v) if isinstance(v, str) else v for k, v in d.items()}


class LLMRouter:
    """Holds per-role LLMClient instances.

    Use ``for_role(role)`` to get the client for a specific operation.
    Implements ``LLMClient`` itself so it can be used as a drop-in replacement
    (calls go to the ``default`` role).
    """

    def __init__(self, clients: Dict[str, LLMClient]):
        if "default" not in clients:
            raise ValueError("LLMRouter requires a 'default' client")
        self._clients = clients

    # -- LLMClient protocol (delegates to default) -----------------------

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        return self._clients["default"].complete(
            messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        )

    # -- Role-based access -----------------------------------------------

    def for_role(self, role: str) -> LLMClient:
        """Return the client for *role*, falling back to ``default``."""
        return self._clients.get(role, self._clients["default"])

    @property
    def structuring(self) -> LLMClient:
        return self.for_role("structuring")

    @property
    def retrieval(self) -> LLMClient:
        return self.for_role("retrieval")

    @property
    def reasoning(self) -> LLMClient:
        return self.for_role("reasoning")

    @property
    def consolidation(self) -> LLMClient:
        return self.for_role("consolidation")

    # -- Factory ---------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LLMRouter":
        """Load router from a YAML config file."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"Empty config file: {path}")

        default_cfg = _expand_dict(raw.get("default", {}))
        if not default_cfg:
            raise ValueError("Config must contain a 'default' section")

        clients: Dict[str, LLMClient] = {}
        for role in ROLES:
            if role == "default":
                clients[role] = _build_client(default_cfg)
            elif role in raw:
                # Merge: role-specific values override default
                merged = {**default_cfg, **_expand_dict(raw[role])}
                clients[role] = _build_client(merged)
            # else: not configured → for_role() falls back to default

        configured = ["default"] + [r for r in ROLES[1:] if r in clients]
        logger.info("LLMRouter loaded from %s — roles: %s", path, configured)
        return cls(clients)

    @classmethod
    def from_single_client(cls, client: LLMClient) -> "LLMRouter":
        """Wrap a single client as a router (all roles use the same client)."""
        return cls({"default": client})


def _build_client(cfg: dict) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        base_url=cfg.get("base_url", ""),
        api_key=cfg.get("api_key", ""),
        model=cfg.get("model", ""),
        max_retries=int(cfg.get("max_retries", 5)),
        retry_delay=float(cfg.get("retry_delay", 5.0)),
        is_azure=bool(cfg.get("is_azure", False)),
        azure_api_version=cfg.get("azure_api_version", "2024-05-01-preview"),
        token_usage_file=cfg.get("token_usage_file"),
    )
