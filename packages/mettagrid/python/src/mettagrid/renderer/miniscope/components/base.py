"""Base component class for miniscope renderer."""

from abc import ABC, abstractmethod
from typing import List, Optional

from rich.console import Console

from mettagrid import MettaGridEnv
from mettagrid.renderer.miniscope.miniscope_panel import MiniscopePanel, PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState


class MiniscopeComponent(ABC):
    """Base class for miniscope renderer components."""

    def __init__(
        self,
        env: MettaGridEnv,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the component.

        Args:
            env: MettaGrid environment reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        self._env = env
        self._state = state
        self._panels = panels
        self._panel: Optional[MiniscopePanel] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._console = Console()

    @property
    def env(self) -> MettaGridEnv:
        """Return the current environment reference."""
        return self._env

    @property
    def state(self) -> MiniscopeState:
        """Return the shared renderer state."""
        return self._state

    @property
    def panels(self) -> PanelLayout:
        """Return the panel layout registry."""
        return self._panels

    def _set_panel(self, panel: Optional[MiniscopePanel]) -> None:
        """Set the panel for this component and update dimensions.

        Args:
            panel: The panel to use for this component
        """
        if panel is None:
            raise ValueError(f"{self.__class__.__name__} requires a configured panel")

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
        if width is None:
            return lines
        return [line[:width].ljust(width) for line in lines]

    def handle_input(self, ch: str) -> bool:
        """Handle user input for this component."""
        return False

    @abstractmethod
    def update(self) -> None:
        """Update the component and set its panel content.

        Components should call self._panel.set_content() to update their display.
        """
        pass
