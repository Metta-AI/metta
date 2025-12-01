"""CoGames integration helpers for Tribal Village."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import side effect only
    from typer import Typer

from .cli import register_cli  # noqa: F401

__all__ = ["register_cli"]
