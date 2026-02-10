from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class ChatMessage:
    """
    Simple chat message representation compatible with most chat model APIs.
    The 'role' is typically one of: 'system', 'user', 'assistant', 'tool'.
    """
    role: str
    content: str


class PromptBase(ABC):
    """
    Base class for building prompts.
    Subclass this and implement 'build_messages' to produce a list of ChatMessage.
    """

    def __init__(self, default_variables: Optional[Mapping[str, Any]] = None) -> None:
        self._default_variables: Dict[str, Any] = dict(default_variables or {})

    @property
    def name(self) -> str:
        """
        A human-readable name for the prompt. Subclasses can override.
        Defaults to the class name.
        """
        return self.__class__.__name__

    def get_default_variables(self) -> Dict[str, Any]:
        """
        Returns a shallow copy of the default variables to avoid accidental mutation.
        """
        return dict(self._default_variables)

    def with_variables(self, overrides: Optional[Mapping[str, Any]] = None) -> "PromptBase":
        """
        Returns a new prompt instance that has default variables merged with 'overrides'.
        Subclasses inherit the same type.
        """
        merged: Dict[str, Any] = self.get_default_variables()
        if overrides:
            merged.update(overrides)
        # Recreate instance with merged defaults. Supports subclasses with the same __init__ signature.
        return self.__class__(default_variables=merged)  # type: ignore[misc]

    def render(self, variables: Optional[Mapping[str, Any]] = None) -> List[ChatMessage]:
        """
        Public entry point to produce the final list of messages.
        Merges default_variables with 'variables' (latter wins), and delegates to build_messages.
        """
        merged_variables = self._merge_variables(variables)
        return self.build_messages(merged_variables)

    def format_text(self, template: str, variables: Mapping[str, Any]) -> str:
        """
        Helper for simple string templating using str.format.
        Subclasses can override to use a different templating engine if needed.
        """
        try:
            return template.format(**variables)
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            raise ValueError(f"Missing variable '{missing_key}' for prompt '{self.name}'.") from exc

    @abstractmethod
    def build_messages(self, variables: Mapping[str, Any]) -> List[ChatMessage]:
        """
        Subclasses must implement this to return the final message list given variables.
        Typical implementation:
            return [
                ChatMessage(role="system", content=self.format_text(system_tmpl, variables)),
                ChatMessage(role="user", content=self.format_text(user_tmpl, variables)),
            ]
        """
        raise NotImplementedError

    # ----- Internal utilities -----

    def _merge_variables(self, variables: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = self.get_default_variables()
        if variables:
            merged.update(variables)
        return merged


