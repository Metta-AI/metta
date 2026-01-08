"""Map component for miniscope renderer."""

from mettagrid.renderer.miniscope.buffer import MapBuffer
from mettagrid.renderer.miniscope.components.base import MiniscopeComponent
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState, RenderMode
from mettagrid.simulator.simulator import Simulation


class MapComponent(MiniscopeComponent):
    """Component for rendering the game map."""

    def __init__(
        self,
        sim: Simulation,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the map component.

        Args:
            sim: MettaGrid simulator reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(sim=sim, state=state, panels=panels)
        self._set_panel(panels.map_view)

        # Create map buffer - will be initialized with data from state
        self._map_buffer = MapBuffer(
            symbol_map=state.symbol_map or {},
            initial_height=sim.map_height,
            initial_width=sim.map_width,
        )

    def _update_buffer_config(self) -> None:
        """Update buffer configuration from state."""
        self._map_buffer._symbol_map = self.state.symbol_map or {}

    def handle_input(self, ch: str) -> bool:
        """Handle map-specific inputs (cursor movement in SELECT mode).

        Args:
            ch: The character input from the user

        Returns:
            True if the input was handled
        """
        # Only handle cursor movement when in SELECT mode
        # Camera panning is now handled by SimControlComponent
        if self._state.mode != RenderMode.SELECT:
            return False

        # Handle cursor movement with shift-key acceleration
        if ch == "i":
            self._state.move_cursor(-1, 0)
            return True
        elif ch == "I":
            self._state.move_cursor(-10, 0)
            return True
        elif ch == "k":
            self._state.move_cursor(1, 0)
            return True
        elif ch == "K":
            self._state.move_cursor(10, 0)
            return True
        elif ch == "j":
            self._state.move_cursor(0, -1)
            return True
        elif ch == "J":
            self._state.move_cursor(0, -10)
            return True
        elif ch == "l":
            self._state.move_cursor(0, 1)
            return True
        elif ch == "L":
            self._state.move_cursor(0, 10)
            return True

        return False

    def update(self) -> None:
        """Update the map display."""
        panel = self._panel
        assert panel is not None
        # Update buffer configuration from state
        self._update_buffer_config()

        # Get grid objects from environment
        grid_objects = self._sim.grid_objects()

        # Get viewport size from panel
        panel_width, panel_height = panel.size()
        # Each map cell takes 2 chars in width
        viewport_width = panel_width // 2 if panel_width else self.state.viewport_width
        viewport_height = panel_height if panel_height else self.state.viewport_height

        # Update viewport with computed size
        self._map_buffer.set_viewport(
            self.state.camera_row,
            self.state.camera_col,
            viewport_height,
            viewport_width,
        )

        # Set cursor if in select mode
        if self.state.mode == RenderMode.SELECT:
            self._map_buffer.set_cursor(self.state.cursor_row, self.state.cursor_col)
        else:
            self._map_buffer.set_cursor(None, None)

        # Highlight selected agent if in vibe picker mode
        if self.state.mode == RenderMode.VIBE_PICKER:
            self._map_buffer.set_highlighted_agent(self.state.selected_agent)
        else:
            self._map_buffer.set_highlighted_agent(None)

        # Render with viewport and set panel content
        buffer = self._map_buffer.render(grid_objects, use_viewport=True)
        panel.set_content(buffer.split("\n"))
