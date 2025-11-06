"""Base renderer classes for game rendering."""

import abc
import typing

import typing_extensions

import mettagrid.simulator

RenderMode = typing.Literal["gui", "unicode", "log", "none"]


class Renderer(mettagrid.simulator.SimulatorEventHandler):
    """Abstract base class for game renderers."""

    def __init__(self):
        super().__init__()

    @typing_extensions.override
    def on_episode_start(self) -> None:
        """Initialize the renderer for a new episode."""
        pass

    def render(self) -> None:
        """Render the current state. Override this for interactive renderers that need to handle input."""
        pass

    @typing_extensions.override
    def on_step(self) -> None:
        """Called after each simulator step. Subclasses can access simulator state."""
        pass

    @abc.abstractmethod
    def on_episode_end(self) -> None:
        """Clean up renderer resources."""
        pass


class NoRenderer(Renderer):
    """Renderer for headless mode (no rendering)."""

    def on_episode_start(self) -> None:
        pass

    @typing_extensions.override
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
        import mettagrid.renderer.miniscope

        return mettagrid.renderer.miniscope.MiniscopeRenderer()
    elif render_mode == "gui":
        # GUI-based interactive rendering
        import mettagrid.renderer.mettascope

        return mettagrid.renderer.mettascope.MettascopeRenderer()
    elif render_mode == "log":
        # Logger-based rendering for debugging
        import mettagrid.renderer.log_renderer

        return mettagrid.renderer.log_renderer.LogRenderer()
    elif render_mode == "none":
        # No rendering
        return NoRenderer()
    raise ValueError(f"Invalid render_mode: {render_mode}")
