"""Prompt base class — copied from src/prompt_base.py with fixed imports."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class PromptBase(ABC):
    def __init__(self, default_variables: Optional[Mapping[str, Any]] = None) -> None:
        self._default_variables: Dict[str, Any] = dict(default_variables or {})

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_default_variables(self) -> Dict[str, Any]:
        return dict(self._default_variables)

    def with_variables(self, overrides: Optional[Mapping[str, Any]] = None) -> PromptBase:
        merged: Dict[str, Any] = self.get_default_variables()
        if overrides:
            merged.update(overrides)
        return self.__class__(default_variables=merged)  # type: ignore[misc]

    def render(self, variables: Optional[Mapping[str, Any]] = None) -> List[ChatMessage]:
        merged_variables = self._merge_variables(variables)
        return self.build_messages(merged_variables)

    def format_text(self, template: str, variables: Mapping[str, Any]) -> str:
        try:
            return template.format(**variables)
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            raise ValueError(f"Missing variable '{missing_key}' for prompt '{self.name}'.") from exc

    @abstractmethod
    def build_messages(self, variables: Mapping[str, Any]) -> List[ChatMessage]:
        raise NotImplementedError

    def _merge_variables(self, variables: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = self.get_default_variables()
        if variables:
            merged.update(variables)
        return merged
