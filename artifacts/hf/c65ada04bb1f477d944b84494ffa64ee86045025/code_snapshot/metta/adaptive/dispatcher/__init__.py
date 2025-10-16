"""Dispatcher implementations for job execution (adaptive namespace)."""

from .local import LocalDispatcher
from .skypilot import SkypilotDispatcher

__all__ = ["LocalDispatcher", "SkypilotDispatcher"]
