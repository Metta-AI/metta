"""Base component class for miniscope renderer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from rich.console import Console, RenderableType

from mettagrid.renderer.miniscope.miniscope_panel import MiniscopePanel, PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState
from mettagrid.simulator.simulator import Simulation


class MiniscopeComponent(ABC):
    """Base class for miniscope renderer components."""

    def __init__(
        self,
        sim: Simulation,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the component.

        Args:
            env: MettaGrid environment reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        self._sim = sim
        self._state = state
        self._panels = panels
        self._panel: Optional[MiniscopePanel] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._console = Console()

    @property
    def env(self) -> "Simulation":
        """Get the environment."""
        return self._sim

    @property
    def state(self) -> MiniscopeState:
        """Return the shared renderer state."""
        return self._state

    @property
    def panels(self) -> PanelLayout:
        """Return the panel layout registry."""
        return self._panels

    def _set_panel(self, panel: MiniscopePanel) -> None:
        """Set the panel for this component and update dimensions.

        Args:
            panel: The panel to use for this component
        """
        self._panel = panel
        self._width = panel.width
        self._height = panel.height

    def _pad_lines(self, lines: List[str], width: int) -> List[str]:
        """Pad lines to a specific width.

        Args:
            lines: Lines to pad
            width: Target width

        Returns:
            Padded lines
        """
        return [line[:width].ljust(width) for line in lines]

    def _table_to_lines(self, renderable: RenderableType) -> List[str]:
        """Render a Rich table or other renderable to plain text lines."""
        with self._console.capture() as capture:
            self._console.print(renderable)
        return capture.get().split("\n")

    def handle_input(self, ch: str) -> bool:
        """Handle user input for this component."""
        return False

    @abstractmethod
    def update(self) -> None:
        """Update the component and set its panel content.

        Components should call self._panel.set_content() to update their display.
        """
        pass
