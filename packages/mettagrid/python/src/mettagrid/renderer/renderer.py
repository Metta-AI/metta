"""Base renderer classes for game rendering."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    from mettagrid import MettaGridEnv


class Renderer(ABC):
    """Abstract base class for game renderers."""

    @abstractmethod
    def on_episode_start(self, env: "MettaGridEnv") -> None:
        """Initialize the renderer for a new episode."""
        pass

    @abstractmethod
    def render(self) -> None:
        """Render the current state. Override this for interactive renderers that need to handle input."""
        pass

    @abstractmethod
    def on_step(
        self,
        current_step: int,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:
        """Render a single step and optionally get user input."""
        pass

    @abstractmethod
    def should_continue(self) -> bool:
        """Check if rendering should continue.

        Returns:
            True if should continue, False to exit
        """
        pass

    def get_user_actions(self) -> Dict[int, tuple[int, int]]:
        """Get the current user actions for all agents.

        Returns:
            Dictionary mapping agent_id to (action_id, action_param)
        """
        return {}

    @abstractmethod
    def on_episode_end(self, infos: Dict[str, Any]) -> None:
        """Clean up renderer resources."""
        pass


class NoRenderer(Renderer):
    """Renderer for headless mode (no rendering)."""

    def on_episode_start(self, env: "MettaGridEnv") -> None:
        pass

    def render(self) -> None:
        pass

    def on_step(
        self,
        current_step: int,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:
        return None

    def should_continue(self) -> bool:
        return True

    def on_episode_end(self, infos: Dict[str, Any]) -> None:
        return None
