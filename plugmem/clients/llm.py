"""LLM client abstractions.

Consolidates call_gpt, call_qwen, call_dpsk, call_llm_openrouter_api into a
single OpenAI-compatible client that accepts base_url + api_key + model.
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract interface for LLM completion."""

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        ...


class OpenAICompatibleLLMClient(LLMClient):
    """Unified LLM client for any OpenAI-compatible API.

    Covers vLLM (Qwen), OpenAI, Azure OpenAI, OpenRouter, DeepSeek, etc.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        is_azure: bool = False,
        azure_api_version: str = "2024-05-01-preview",
        token_usage_file: Optional[str] = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.token_usage_file = token_usage_file

        if is_azure:
            self._client = AzureOpenAI(
                azure_endpoint=base_url,
                api_key=api_key,
                api_version=azure_api_version,
            )
        else:
            self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                self._log_usage(response, messages)
                content = response.choices[0].message.content
                return content.strip() if content else ""

            except Exception as e:
                logger.warning("[Attempt %d/%d] LLM error: %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        return ""

    def _log_usage(self, response: Any, messages: List[Dict[str, str]]) -> None:
        if not self.token_usage_file:
            return
        try:
            usage = response.usage
            if usage is None:
                return
            prompt_preview = messages[1]["content"][:100] if len(messages) > 1 else ""
            entry = {"model": self.model, "prompt_first100": prompt_preview}
            entry.update(usage.model_dump())
            with open(self.token_usage_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


def create_llm_client_from_env() -> LLMClient:
    """Create an LLM client from environment variables (backward compat)."""
    model_name = os.environ.get("LLM_NAME", "")
    qwen_aliases = {
        "qwen-2.5-32b-instruct",
        "qwen2.5-32b-instruct",
        "qwen2.5-7b-instruct",
        "qwen/qwen2.5-7b-instruct",
        "calamitousfelicitousness/qwen2.5-32b-instruct-fp8-dynamic",
    }

    if model_name.lower() in qwen_aliases:
        return OpenAICompatibleLLMClient(
            base_url=os.environ["QWEN_BASE_URL"],
            api_key=os.environ.get("VLLM_QWEN_API_KEY", ""),
            model="CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic",
            token_usage_file=os.environ.get("TOKEN_USAGE_FILE"),
        )
    elif model_name.lower() == "deepseek-v3-0324":
        return OpenAICompatibleLLMClient(
            base_url=os.environ.get("AZURE_DPSK_ENDPOINT", ""),
            api_key=os.environ.get("AZURE_DPSK_API_KEY", ""),
            model="DeepSeek-V3-0324",
            is_azure=True,
            token_usage_file=os.environ.get("TOKEN_USAGE_FILE"),
        )
    elif model_name in {"4o", "4o-mini", "4.1", "o1", "5.1", "5.2"}:
        azure_endpoint = os.environ.get("AZURE_ENDPOINT", "")
        if azure_endpoint:
            return OpenAICompatibleLLMClient(
                base_url=azure_endpoint,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=model_name,
                is_azure=True,
                azure_api_version="2024-12-01-preview",
                token_usage_file=os.environ.get("TOKEN_USAGE_FILE"),
            )
        return OpenAICompatibleLLMClient(
            base_url="https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=model_name,
            token_usage_file=os.environ.get("TOKEN_USAGE_FILE"),
        )
    else:
        # OpenRouter fallback
        return OpenAICompatibleLLMClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            model=model_name,
            token_usage_file=os.environ.get("TOKEN_USAGE_FILE"),
        )
