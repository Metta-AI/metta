"""Integration helpers that bridge Tribal Village into CoGames."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - optional Typer dependency
    from typer import Typer


def register_policies() -> None:
    # Importing policy registers short names via metaclass side effects
    from . import policy  # noqa: F401


def register_cli(app: "Typer") -> None:
    register_policies()
    from .cli import attach_train_command

    attach_train_command(app)


# Ensure policies register when the package is imported in environments that expect it.
register_policies()

__all__ = ["register_cli", "register_policies"]
