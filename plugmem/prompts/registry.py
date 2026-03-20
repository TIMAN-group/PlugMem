"""Prompt registry with YAML-based per-graph customization.

Resolution order: request override → per-graph YAML → service defaults YAML → built-in code defaults.

YAML format (each prompt key maps to a list of messages):

    get_subgoal:
      system: "You are analyzing agent behavior..."
      user: "Overall Goal: {goal}\\nObservation: {observation}\\n..."

    get_plan:
      system: "You are an assistant..."
      user: "Goal: {goal}\\n..."

Any prompt not found in YAML falls back to the built-in PromptBase subclass.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from plugmem.prompts.base import ChatMessage, PromptBase
from plugmem.prompts.reasoning import (
    DefaultEpisodicPrompt,
    DefaultProceduralPrompt,
    DefaultSemanticPrompt,
)
from plugmem.prompts.retrieving import (
    GetModePrompt,
    GetNewSemanticPrompt,
    GetNewSubgoalPrompt,
    GetPlanPrompt,
)
from plugmem.prompts.structuring import (
    GetProceduralPrompt,
    GetReturnPrompt,
    GetRewardPrompt,
    GetSemanticPrompt,
    GetStatePrompt,
    GetSubgoalPrompt,
)

logger = logging.getLogger(__name__)

# Canonical prompt names → built-in default classes
_BUILTIN_DEFAULTS: Dict[str, type] = {
    # structuring
    "get_subgoal": GetSubgoalPrompt,
    "get_reward": GetRewardPrompt,
    "get_state": GetStatePrompt,
    "get_semantic": GetSemanticPrompt,
    "get_return": GetReturnPrompt,
    "get_procedural": GetProceduralPrompt,
    # retrieving
    "get_plan": GetPlanPrompt,
    "get_new_semantic": GetNewSemanticPrompt,
    "get_new_subgoal": GetNewSubgoalPrompt,
    "get_mode": GetModePrompt,
    # reasoning
    "reasoning_episodic": DefaultEpisodicPrompt,
    "reasoning_semantic": DefaultSemanticPrompt,
    "reasoning_procedural": DefaultProceduralPrompt,
}


class TemplatePrompt(PromptBase):
    """A prompt built from YAML-defined system/user templates."""

    def __init__(
        self,
        name: str,
        system_template: str,
        user_template: str,
        default_variables: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(default_variables)
        self._name = name
        self._system_template = system_template
        self._user_template = user_template

    @property
    def name(self) -> str:
        return self._name

    def build_messages(self, variables: Mapping[str, Any]) -> List[ChatMessage]:
        return [
            ChatMessage("system", self.format_text(self._system_template, variables)),
            ChatMessage("user", self.format_text(self._user_template, variables)),
        ]


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file, returning {} on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("Failed to load prompt YAML %s: %s", path, e)
        return {}


def _yaml_entry_to_prompt(name: str, entry: Dict[str, str]) -> Optional[TemplatePrompt]:
    """Convert a YAML entry to a TemplatePrompt, or None if invalid."""
    system = entry.get("system")
    user = entry.get("user")
    if system is None or user is None:
        logger.warning("Prompt '%s' in YAML missing 'system' or 'user' key, skipping", name)
        return None
    return TemplatePrompt(name=name, system_template=system, user_template=user)


class PromptRegistry:
    """Layered prompt registry.

    Resolution: request override → per-graph YAML → service defaults YAML → built-in code.

    Usage::

        registry = PromptRegistry(prompts_dir="data/prompts")
        registry.load_graph("my_graph")

        # Get a prompt (resolves through layers)
        prompt = registry.get("get_subgoal", graph_id="my_graph")
        messages = prompt.render({"goal": "...", "observation": "..."})
    """

    def __init__(self, prompts_dir: Optional[str] = None):
        self._prompts_dir = Path(prompts_dir) if prompts_dir else None

        # Layer 1: built-in defaults (always present)
        self._builtins: Dict[str, PromptBase] = {
            name: cls() for name, cls in _BUILTIN_DEFAULTS.items()
        }

        # Layer 2: service-wide YAML overrides (_defaults.yaml)
        self._service_overrides: Dict[str, PromptBase] = {}

        # Layer 3: per-graph YAML overrides ({graph_id}.yaml)
        self._graph_overrides: Dict[str, Dict[str, PromptBase]] = {}

        # Load service defaults if directory exists
        if self._prompts_dir and self._prompts_dir.exists():
            self._load_service_defaults()

    def _load_service_defaults(self) -> None:
        path = self._prompts_dir / "_defaults.yaml"
        data = _load_yaml(str(path))
        for name, entry in data.items():
            if isinstance(entry, dict):
                prompt = _yaml_entry_to_prompt(name, entry)
                if prompt:
                    self._service_overrides[name] = prompt
        if self._service_overrides:
            logger.info("Loaded %d service-wide prompt overrides from %s", len(self._service_overrides), path)

    def load_graph(self, graph_id: str) -> None:
        """Load per-graph prompt overrides from {graph_id}.yaml."""
        if not self._prompts_dir:
            return
        path = self._prompts_dir / f"{graph_id}.yaml"
        data = _load_yaml(str(path))
        if not data:
            return
        overrides: Dict[str, PromptBase] = {}
        for name, entry in data.items():
            if isinstance(entry, dict):
                prompt = _yaml_entry_to_prompt(name, entry)
                if prompt:
                    overrides[name] = prompt
        self._graph_overrides[graph_id] = overrides
        if overrides:
            logger.info("Loaded %d prompt overrides for graph %s", len(overrides), graph_id)

    def get(
        self,
        name: str,
        graph_id: Optional[str] = None,
        override: Optional[PromptBase] = None,
    ) -> PromptBase:
        """Resolve a prompt by name through the layer stack.

        Resolution order:
            1. Explicit override (passed at call site)
            2. Per-graph YAML override
            3. Service-wide YAML override
            4. Built-in code default
        """
        # Layer 0: explicit override
        if override is not None:
            return override

        # Layer 1: per-graph YAML
        if graph_id and graph_id in self._graph_overrides:
            prompt = self._graph_overrides[graph_id].get(name)
            if prompt is not None:
                return prompt

        # Layer 2: service-wide YAML
        prompt = self._service_overrides.get(name)
        if prompt is not None:
            return prompt

        # Layer 3: built-in
        prompt = self._builtins.get(name)
        if prompt is not None:
            return prompt

        raise KeyError(f"Unknown prompt: '{name}'")

    def set(self, name: str, prompt: PromptBase, graph_id: Optional[str] = None) -> None:
        """Programmatically override a prompt at runtime."""
        if graph_id:
            self._graph_overrides.setdefault(graph_id, {})[name] = prompt
        else:
            self._service_overrides[name] = prompt

    def list_prompts(self) -> List[str]:
        """List all known prompt names."""
        return sorted(_BUILTIN_DEFAULTS.keys())

    def save_defaults_yaml(self, path: Optional[str] = None) -> str:
        """Export built-in defaults as a YAML reference file.

        Useful for creating a starting point for customization.
        """
        if path is None:
            if self._prompts_dir:
                self._prompts_dir.mkdir(parents=True, exist_ok=True)
                path = str(self._prompts_dir / "_defaults.yaml")
            else:
                path = "_defaults.yaml"

        output: Dict[str, Dict[str, str]] = {}
        for name, cls in _BUILTIN_DEFAULTS.items():
            instance = cls()
            # Render with placeholder variables to extract the templates
            # We use the build_messages with dummy variables — but that would fail
            # on format. Instead, introspect the class.
            output[name] = _extract_templates(instance)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)
        logger.info("Saved default prompts reference to %s", path)
        return path


# All known variable names used across prompts
_ALL_VARIABLE_NAMES = [
    "goal", "state", "observation", "action", "subgoal",
    "task_type", "trajectory", "procedural_memory",
    "memory_earlier", "memory_later", "goal_1", "goal_2",
    "semantic_memory", "episodic_memory", "time",
    "information", "question",
]


def _extract_templates(prompt: PromptBase) -> Dict[str, str]:
    """Best-effort extraction of system/user templates from a PromptBase instance.

    Provides placeholder values like '{goal}' for all known variable names,
    so format() substitutions resolve to the placeholder strings.
    """
    # Build a dict where each key maps to its own placeholder string
    passthrough = {name: "{" + name + "}" for name in _ALL_VARIABLE_NAMES}

    try:
        messages = prompt.build_messages(passthrough)
        result = {}
        for msg in messages:
            result[msg.role] = msg.content
        return result
    except Exception:
        return {
            "system": "# Could not auto-extract — customize manually",
            "user": "# Could not auto-extract — customize manually",
        }
