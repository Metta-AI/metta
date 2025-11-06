"""Base component class for miniscope renderer."""

import abc
import typing

import rich.console

import mettagrid.renderer.miniscope.miniscope_panel
import mettagrid.renderer.miniscope.miniscope_state
import mettagrid.simulator


class MiniscopeComponent(abc.ABC):
    """Base class for miniscope renderer components."""

    def __init__(
        self,
        sim: mettagrid.simulator.Simulation,
        state: mettagrid.renderer.miniscope.miniscope_state.MiniscopeState,
        panels: mettagrid.renderer.miniscope.miniscope_panel.PanelLayout,
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
        self._panel: typing.Optional[mettagrid.renderer.miniscope.miniscope_panel.MiniscopePanel] = None
        self._width: typing.Optional[int] = None
        self._height: typing.Optional[int] = None
        self._console = rich.console.Console()

    @property
    def env(self) -> mettagrid.simulator.Simulation:
        """Get the environment."""
        return self._sim

    @property
    def state(self) -> mettagrid.renderer.miniscope.miniscope_state.MiniscopeState:
        """Return the shared renderer state."""
        return self._state

    @property
    def panels(self) -> mettagrid.renderer.miniscope.miniscope_panel.PanelLayout:
        """Return the panel layout registry."""
        return self._panels

    def _set_panel(self, panel: typing.Optional[mettagrid.renderer.miniscope.miniscope_panel.MiniscopePanel]) -> None:
        """Set the panel for this component and update dimensions.

        Args:
            panel: The panel to use for this component
        """
        if panel is None:
            raise ValueError(f"{self.__class__.__name__} requires a configured panel")

        self._panel = panel
        self._width = panel.width
        self._height = panel.height

    def _pad_lines(self, lines: typing.List[str], width: int) -> typing.List[str]:
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

    @abc.abstractmethod
    def update(self) -> None:
        """Update the component and set its panel content.

        Components should call self._panel.set_content() to update their display.
        """
        pass
