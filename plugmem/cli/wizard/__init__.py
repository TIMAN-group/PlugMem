"""Interactive setup wizard for `plugmem init`.

Composed of three sections (LLM, embedding, service) plus a final
probe, all wired together by `wizard.main.run_wizard`. Each section is
independently callable and returns success/failure so the orchestrator
can re-run individual sections on failure without restarting the whole
flow.
"""
from plugmem.cli.wizard.main import run_wizard
from plugmem.cli.wizard.sections import (
    run_embedding_section,
    run_llm_section,
    run_service_section,
)

__all__ = [
    "run_wizard",
    "run_llm_section",
    "run_embedding_section",
    "run_service_section",
]
