"""Shared Rich console + prompt helpers for the wizard.

A single `console` instance is exported so all sections share the same
output target — important for testing (we redirect stdout) and for
keeping rendering consistent.
"""
from __future__ import annotations

from typing import List, Optional

from rich.console import Console
from rich.prompt import Prompt

console = Console()


def header(text: str) -> None:
    """Section header — bolded with a leading rule."""
    console.rule(f"[bold]{text}[/bold]")


def info(text: str) -> None:
    console.print(text)


def success(text: str) -> None:
    console.print(f"[green]✓[/green] {text}")


def warn(text: str) -> None:
    console.print(f"[yellow]![/yellow] {text}")


def error(text: str) -> None:
    console.print(f"[red]✗[/red] {text}")


def prompt_text(
    label: str,
    *,
    default: Optional[str] = None,
    password: bool = False,
    allow_empty: bool = False,
) -> str:
    """Prompt for a string. Loops if empty input is rejected."""
    while True:
        value = Prompt.ask(label, default=default, password=password, console=console)
        # Rich returns None when default is None and user hits enter.
        value = (value or "").strip()
        if value or allow_empty:
            return value
        warn("Value cannot be empty.")


def prompt_choice(
    label: str,
    choices: List[str],
    *,
    default: Optional[str] = None,
) -> str:
    """Prompt for one of a fixed set of choices."""
    return Prompt.ask(
        label,
        choices=choices,
        default=default if default in choices else choices[0],
        console=console,
    )


def prompt_action(label: str = "What now?") -> str:
    """Standard retry/edit/skip prompt used after validation failures."""
    return prompt_choice(label, ["retry", "edit", "skip"], default="retry")
