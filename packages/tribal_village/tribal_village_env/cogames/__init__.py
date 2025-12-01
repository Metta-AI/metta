"""Integration helpers that bridge Tribal Village into CoGames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - optional Typer dependency
    from typer import Typer


@dataclass
class TribalVillagePlugin:
    """Small helper that wires Tribal Village features into host CLIs."""

    def register_policies(self) -> None:
        # Importing policy registers short names via metaclass side effects
        from . import policy  # noqa: F401

    def register_cli(self, app: "Typer", *, require_cogames: bool = True) -> None:
        from .cli import attach_train_command

        attach_train_command(app, require_cogames=require_cogames)


plugin = TribalVillagePlugin()
plugin.register_policies()


def register_cli(app: "Typer") -> None:
    plugin.register_policies()
    plugin.register_cli(app)


def register_policies() -> None:
    plugin.register_policies()


__all__ = ["plugin", "register_cli", "register_policies"]
