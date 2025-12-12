"""Base renderer classes for game rendering."""

from abc import abstractmethod
from typing import Literal

from typing_extensions import override

from mettagrid.simulator.interface import SimulatorEventHandler

RenderMode = Literal["gui", "unicode", "log", "none"]


class Renderer(SimulatorEventHandler):
    """Abstract base class for game renderers."""

    def __init__(self):
        super().__init__()

    @override
    def on_episode_start(self) -> None:
        """Initialize the renderer for a new episode."""
        pass

    def render(self) -> None:
        """Render the current state. Override this for interactive renderers that need to handle input."""
        pass

    @override
    def on_step(self) -> None:
        """Called after each simulator step. Subclasses can access simulator state."""
        pass

    @abstractmethod
    def on_episode_end(self) -> None:
        """Clean up renderer resources."""
        pass


class NoRenderer(Renderer):
    """Renderer for headless mode (no rendering)."""

    def on_episode_start(self) -> None:
        pass

    @override
    def render(self) -> None:
        pass

    def on_step(self) -> None:
        pass

    def on_episode_end(self) -> None:
        pass


def create_renderer(render_mode: RenderMode) -> Renderer:
    """Create the appropriate renderer based on render_mode."""
    if render_mode == "unicode":
        # Text-based interactive rendering
        from mettagrid.renderer.miniscope.miniscope import MiniscopeRenderer

        return MiniscopeRenderer()
    elif render_mode == "gui":
        # GUI-based interactive rendering
        from mettagrid.renderer.mettascope import MettascopeRenderer

        return MettascopeRenderer()
    elif render_mode == "log":
        # Logger-based rendering for debugging
        from mettagrid.renderer.log_renderer import LogRenderer

        return LogRenderer()
    elif render_mode == "none":
        # No rendering
        return NoRenderer()
    raise ValueError(f"Invalid render_mode: {render_mode}")
