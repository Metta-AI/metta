"""Base renderer classes for game rendering."""

from abc import abstractmethod
from typing import Any, Dict, Literal

import numpy as np
from typing_extensions import override

from mettagrid.simulator import Simulator, SimulatorEventHandler

RenderMode = Literal["gui", "unicode", "none"]


class Renderer(SimulatorEventHandler):
    """Abstract base class for game renderers."""

    def __init__(self, simulator: Simulator):
        super().__init__(simulator)

    @override
    def on_episode_start(self) -> None:
        """Initialize the renderer for a new episode."""
        pass

    @override
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

    def on_episode_start(self) -> None:
        pass

    @override
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


def create_renderer(simulator: Simulator, render_mode: RenderMode) -> Renderer:
    """Create the appropriate renderer based on render_mode."""
    if render_mode in ("unicode"):
        # Text-based interactive rendering
        from mettagrid.renderer.miniscope import MiniscopeRenderer

        return MiniscopeRenderer(simulator)
    elif render_mode in ("gui"):
        # GUI-based interactive rendering
        from mettagrid.renderer.mettascope import MettascopeRenderer

        return MettascopeRenderer(simulator)
    elif render_mode in ("none"):
        # No rendering
        return NoRenderer(simulator)
    raise ValueError(f"Invalid render_mode: {render_mode}")
