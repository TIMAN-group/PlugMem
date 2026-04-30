"""FastAPI dependency injection — wires up config, clients, and GraphManager."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import chromadb

from plugmem.clients.embedding import EmbeddingClient, HTTPEmbeddingClient, PlugMemEmbeddingFunction
from plugmem.clients.llm import LLMClient, OpenAICompatibleLLMClient
from plugmem.clients.llm_router import LLMRouter
from plugmem.config import PlugMemConfig
from plugmem.graph_manager import GraphManager
from plugmem.storage.chroma import ChromaStorage

logger = logging.getLogger(__name__)


@lru_cache
def get_config() -> PlugMemConfig:
    """Build config from environment variables."""
    return PlugMemConfig(
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nvidia/NV-Embed-v2"),
        chroma_mode=os.getenv("CHROMA_MODE", "persistent"),
        chroma_path=os.getenv("CHROMA_PATH", "./data/chroma"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        api_key=os.getenv("PLUGMEM_API_KEY"),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "5")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        llm_top_p=float(os.getenv("LLM_TOP_P", "1.0")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        embedding_max_text_len=int(os.getenv("EMBEDDING_MAX_TEXT_LEN", "8192")),
    )


_llm_client: LLMClient | LLMRouter | None = None
_embedding_client: EmbeddingClient | None = None
_graph_manager: GraphManager | None = None


def get_llm(config: PlugMemConfig | None = None) -> LLMClient | LLMRouter:
    """Return the LLM client(s).

    If ``LLM_CONFIG_PATH`` points to a YAML file, an :class:`LLMRouter` with
    per-role clients is returned.  Otherwise a single
    :class:`OpenAICompatibleLLMClient` built from env vars is returned.
    """
    global _llm_client
    if _llm_client is None:
        cfg = config or get_config()

        llm_config_path = os.getenv("LLM_CONFIG_PATH", "")
        if llm_config_path and Path(llm_config_path).is_file():
            _llm_client = LLMRouter.from_yaml(llm_config_path)
            logger.info("LLM routing loaded from %s", llm_config_path)
        else:
            _llm_client = OpenAICompatibleLLMClient(
                base_url=cfg.llm_base_url,
                api_key=cfg.llm_api_key,
                model=cfg.llm_model,
                max_retries=cfg.max_retries,
            )
    return _llm_client


def get_embedder(config: PlugMemConfig | None = None) -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        cfg = config or get_config()
        _embedding_client = HTTPEmbeddingClient(
            base_url=cfg.embedding_base_url,
            model=cfg.embedding_model,
            max_text_len=cfg.embedding_max_text_len,
        )
    return _embedding_client


def get_graph_manager(config: PlugMemConfig | None = None) -> GraphManager:
    global _graph_manager
    if _graph_manager is None:
        cfg = config or get_config()

        # ChromaDB client
        if cfg.chroma_mode == "http":
            chroma_client = chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
        elif cfg.chroma_mode == "ephemeral":
            chroma_client = chromadb.EphemeralClient()
        else:
            chroma_client = chromadb.PersistentClient(path=cfg.chroma_path)

        embedder = get_embedder(cfg)
        embed_fn = PlugMemEmbeddingFunction(embedder)

        storage = ChromaStorage(
            client=chroma_client,
            embedding_function=embed_fn,
            embedding_client=embedder,
        )
        llm = get_llm(cfg)

        _graph_manager = GraphManager(
            storage=storage,
            llm=llm,
            embedder=embedder,
        )
    return _graph_manager


def reset_singletons() -> None:
    """Reset cached singletons — useful for testing."""
    global _llm_client, _embedding_client, _graph_manager
    _llm_client = None
    _embedding_client = None
    _graph_manager = None
    get_config.cache_clear()
