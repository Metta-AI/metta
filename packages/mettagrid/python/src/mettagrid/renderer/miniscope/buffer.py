"""Buffer building utilities for miniscope rendering."""

from typing import Dict, Optional

from .symbol import get_symbol_for_object


class MapBuffer:
    """Encapsulates map buffer handling logic for miniscope rendering."""

    def __init__(
        self,
        object_type_names: list[str],
        symbol_map: dict[str, str],
        initial_height: int = 0,
        initial_width: int = 0,
    ):
        """Initialize the MapBuffer.

        Args:
            object_type_names: List mapping type IDs to names
            symbol_map: Map from object type names to render symbols
            initial_height: Initial map height (0 to auto-compute)
            initial_width: Initial map width (0 to auto-compute)
        """
        self._object_type_names = object_type_names
        self._symbol_map = symbol_map

        # Bounds
        self._min_row = 0
        self._min_col = 0
        self._height = initial_height
        self._width = initial_width
        self._bounds_set = initial_height > 0 and initial_width > 0

        # Viewport state
        self._viewport_center_row: Optional[int] = None
        self._viewport_center_col: Optional[int] = None
        self._viewport_height: Optional[int] = None
        self._viewport_width: Optional[int] = None

        # Cursor state
        self._cursor_row: Optional[int] = None
        self._cursor_col: Optional[int] = None

        # Highlighted agent state (for glyph picker)
        self._highlighted_agent_id: Optional[int] = None

        # Cached grid objects
        self._last_grid_objects: Optional[Dict[int, dict]] = None

    def set_viewport(
        self,
        center_row: Optional[int] = None,
        center_col: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> None:
        """Set viewport parameters for map rendering."""
        self._viewport_center_row = center_row
        self._viewport_center_col = center_col
        self._viewport_height = height
        self._viewport_width = width

    def set_cursor(self, row: Optional[int], col: Optional[int]) -> None:
        """Set the cursor position used in select mode."""
        self._cursor_row = row
        self._cursor_col = col

    def set_highlighted_agent(self, agent_id: Optional[int]) -> None:
        """Track which agent should be highlighted in the viewport."""
        self._highlighted_agent_id = agent_id

    def move_viewport(self, delta_row: int = 0, delta_col: int = 0) -> None:
        """Move the viewport by the given deltas."""
        if self._viewport_center_row is not None:
            self._viewport_center_row += delta_row
        if self._viewport_center_col is not None:
            self._viewport_center_col += delta_col

    def center_viewport_on(self, row: int, col: int) -> None:
        """Center the viewport on the given coordinates."""
        self._viewport_center_row = row
        self._viewport_center_col = col

    def get_bounds(self) -> tuple[int, int, int, int]:
        """Return cached full-map bounds as (min_row, min_col, height, width)."""
        return (self._min_row, self._min_col, self._height, self._width)

    def get_viewport_bounds(self) -> tuple[int, int, int, int]:
        """Return the active viewport bounds as (row, col, height, width)."""
        if self._viewport_center_row is not None and self._viewport_height is not None:
            view_min_row = max(self._min_row, self._viewport_center_row - self._viewport_height // 2)
            view_max_row = min(self._min_row + self._height, view_min_row + self._viewport_height)
            view_height = view_max_row - view_min_row
        else:
            view_min_row = self._min_row
            view_height = self._height

        if self._viewport_center_col is not None and self._viewport_width is not None:
            view_min_col = max(self._min_col, self._viewport_center_col - self._viewport_width // 2)
            view_max_col = min(self._min_col + self._width, view_min_col + self._viewport_width)
            view_width = view_max_col - view_min_col
        else:
            view_min_col = self._min_col
            view_width = self._width

        return (view_min_row, view_min_col, view_height, view_width)

    def _ensure_bounds(self, grid_objects: Dict[int, dict]) -> None:
        """Compute bounds if needed and cache the grid snapshot."""
        if not self._bounds_set or grid_objects != self._last_grid_objects:
            self._min_row, self._min_col, self._height, self._width = self._compute_bounds(grid_objects)
            self._bounds_set = True
            self._last_grid_objects = grid_objects

    def _compute_bounds(self, grid_objects: Dict[int, dict]) -> tuple[int, int, int, int]:
        """Compute a bounding box for the provided grid objects."""
        rows = []
        cols = []
        for obj in grid_objects.values():
            type_name = obj["type_name"]
            if type_name == "wall":
                rows.append(obj["r"])
                cols.append(obj["c"])
        if not rows or not cols:
            for obj in grid_objects.values():
                rows.append(obj["r"])
                cols.append(obj["c"])

        # Handle empty grid case
        if not rows or not cols:
            return (0, 0, 1, 1)

        min_row = min(rows)
        min_col = min(cols)
        height = max(rows) - min_row + 1
        width = max(cols) - min_col + 1
        return (min_row, min_col, height, width)

    def _build_grid_buffer(self, grid_objects: Dict[int, dict], use_viewport: bool = True) -> str:
        """Build the emoji grid buffer using instance attributes.

        Args:
            grid_objects: Dictionary of grid objects to render
            use_viewport: Whether to use viewport settings (False for full map)

        Returns:
            Buffer string with newlines
        """
        # Use instance attributes for viewport settings
        viewport_center_row = self._viewport_center_row if use_viewport else None
        viewport_center_col = self._viewport_center_col if use_viewport else None
        viewport_height = self._viewport_height if use_viewport else None
        viewport_width = self._viewport_width if use_viewport else None

        # Determine viewport bounds
        if viewport_center_row is not None and viewport_height is not None:
            clamped_height = max(1, min(self._height, viewport_height))
            view_min_row = max(self._min_row, viewport_center_row - clamped_height // 2)
            view_max_row = view_min_row + clamped_height
            if view_max_row > self._min_row + self._height:
                view_max_row = self._min_row + self._height
                view_min_row = max(self._min_row, view_max_row - clamped_height)
            view_height = view_max_row - view_min_row
        else:
            view_min_row = self._min_row
            view_height = self._height
            view_max_row = self._min_row + self._height

        if viewport_center_col is not None and viewport_width is not None:
            clamped_width = max(1, min(self._width, viewport_width))
            view_min_col = max(self._min_col, viewport_center_col - clamped_width // 2)
            view_max_col = view_min_col + clamped_width
            if view_max_col > self._min_col + self._width:
                view_max_col = self._min_col + self._width
                view_min_col = max(self._min_col, view_max_col - clamped_width)
            view_width = view_max_col - view_min_col
        else:
            view_min_col = self._min_col
            view_width = self._width
            view_max_col = self._min_col + self._width

        # Initialize grid with empty spaces
        empty_symbol = self._symbol_map.get("empty", "⬜")
        grid = [[empty_symbol for _ in range(view_width)] for _ in range(view_height)]

        # Place objects in viewport
        for obj in grid_objects.values():
            obj_r = obj["r"]
            obj_c = obj["c"]
            r = obj_r - view_min_row
            c = obj_c - view_min_col
            # Skip objects outside viewport bounds
            if 0 <= r < view_height and 0 <= c < view_width:
                # Check if this is the highlighted agent
                if self._highlighted_agent_id is not None and obj.get("agent_id") == self._highlighted_agent_id:
                    # Use a distinctive symbol for highlighted agent
                    grid[r][c] = "⭐"
                else:
                    grid[r][c] = get_symbol_for_object(obj, self._object_type_names, self._symbol_map)

        # Add selection cursor if in select mode
        if self._cursor_row is not None and self._cursor_col is not None:
            cursor_r = self._cursor_row - view_min_row
            cursor_c = self._cursor_col - view_min_col
            if 0 <= cursor_r < view_height and 0 <= cursor_c < view_width:
                cursor_symbol = self._symbol_map.get("cursor", "🎯")
                grid[cursor_r][cursor_c] = cursor_symbol

        # Add directional arrows at edges if there's more content beyond viewport
        has_more_top = view_min_row > self._min_row
        has_more_bottom = view_max_row < self._min_row + self._height
        has_more_left = view_min_col > self._min_col
        has_more_right = view_max_col < self._min_col + self._width

        # Replace edges with full-width arrows
        if has_more_top:
            for c in range(view_width):
                grid[0][c] = "▲ " if c < view_width - 1 else "▲"
        if has_more_bottom:
            for c in range(view_width):
                grid[view_height - 1][c] = "▼ " if c < view_width - 1 else "▼"
        if has_more_left:
            for r in range(view_height):
                grid[r][0] = "◀ " if view_width > 1 else "◀"
        if has_more_right:
            for r in range(view_height):
                grid[r][view_width - 1] = "▶" if view_width > 1 else "▶"

        # Handle corners - show diagonal arrows when both edges have more content
        if has_more_top and has_more_left:
            grid[0][0] = "◤ " if view_width > 1 else "◤"
        if has_more_top and has_more_right:
            grid[0][view_width - 1] = "◥" if view_width > 1 else "◥"
        if has_more_bottom and has_more_left:
            grid[view_height - 1][0] = "◣ " if view_width > 1 else "◣"
        if has_more_bottom and has_more_right:
            grid[view_height - 1][view_width - 1] = "◢" if view_width > 1 else "◢"

        lines = ["".join(row) for row in grid]
        return "\n".join(lines)

    def render(
        self,
        grid_objects: Dict[int, dict],
        use_viewport: bool = True,
    ) -> str:
        """Render the grid buffer as a newline-delimited string."""
        self._ensure_bounds(grid_objects)
        return self._build_grid_buffer(grid_objects, use_viewport)

    def render_full_map(self, grid_objects: Dict[int, dict]) -> str:
        """Render the full map without viewport restrictions.

        Args:
            grid_objects: Dictionary of grid objects to render

        Returns:
            Buffer string with newlines
        """
        return self.render(grid_objects, use_viewport=False)
