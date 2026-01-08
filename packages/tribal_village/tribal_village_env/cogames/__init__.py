"""Integration hook for CoGames."""

from .cli import register_cli
from .policy import TribalVillagePufferPolicy  # noqa: F401 - triggers policy registration side effects

__all__ = ["register_cli", "TribalVillagePufferPolicy"]
