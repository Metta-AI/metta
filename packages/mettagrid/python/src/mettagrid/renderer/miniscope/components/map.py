"""Map component for miniscope renderer."""

from mettagrid import MettaGridEnv
from mettagrid.core import BoundingBox
from mettagrid.renderer.miniscope.buffer import MapBuffer
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState, RenderMode

from .base import MiniscopeComponent


class MapComponent(MiniscopeComponent):
    """Component for rendering the game map."""

    def __init__(
        self,
        env: MettaGridEnv,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the map component.

        Args:
            env: MettaGrid environment reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(env=env, state=state, panels=panels)
        self._set_panel(panels.map_view)

        # Create map buffer - will be initialized with data from state
        self._map_buffer = MapBuffer(
            object_type_names=state.object_type_names or [],
            symbol_map=state.symbol_map or {},
            initial_height=env.map_height,
            initial_width=env.map_width,
        )

    def _update_buffer_config(self) -> None:
        """Update buffer configuration from state."""
        if self.state:
            self._map_buffer._object_type_names = self.state.object_type_names or []
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
        # Update buffer configuration from state
        self._update_buffer_config()

        # Get grid objects from environment
        bbox = BoundingBox(
            min_row=0,
            max_row=self.env.map_height,
            min_col=0,
            max_col=self.env.map_width,
        )
        grid_objects = self.env.grid_objects(bbox)

        # Get viewport size from panel
        panel_width, panel_height = self._panel.size()
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

        # Render with viewport and set panel content
        buffer = self._map_buffer.render(grid_objects, use_viewport=True)
        self._panel.set_content(buffer.split("\n"))
