"""Buffer building utilities for miniscope rendering."""

from typing import Dict

# Agent-specific colored squares for agent IDs 0-9 (consistent width)
AGENT_SQUARES = ["🟦", "🟧", "🟩", "🟨", "🟪", "🟥", "🟫", "⬛", "🟦", "🟧"]


def _cells_for_object(obj: dict) -> list[tuple[int, int, int]]:
    """Return normalized cell coordinates for rendering.

    Prefer the explicit 'cells' array when present. Fall back to location fields for
    legacy data that predates the cells contract so miniscope stays compatible with
    old replays and tooling output.
    """
    cells = obj.get("cells")
    if isinstance(cells, list) and cells:
        normalized: list[tuple[int, int, int]] = []
        for entry in cells:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                continue
            c, r, layer = entry
            normalized.append((int(c), int(r), int(layer)))
        if normalized:
            return normalized
    location = obj.get("location")
    if isinstance(location, (list, tuple)) and len(location) == 3:
        c, r, layer = location
        return [(int(c), int(r), int(layer))]
    col = obj.get("c")
    row = obj.get("r")
    if col is not None and row is not None:
        layer = obj.get("layer", obj.get("z", 0))
        return [(int(col), int(row), int(layer))]
    return []


def get_symbol_for_object(obj: dict, object_type_names: list[str], symbol_map: dict[str, str]) -> str:
    """Get the emoji symbol for an object.

    Args:
        obj: Object dict with 'type' and optional 'agent_id'
        object_type_names: List mapping type IDs to names
        symbol_map: Map from object type names to render symbols

    Returns:
        Emoji symbol string
    """
    type_name = object_type_names[obj["type"]]

    # Handle numbered agents specially
    if type_name.startswith("agent"):
        agent_id = obj.get("agent_id")
        if agent_id is not None and 0 <= agent_id < 10:
            return AGENT_SQUARES[agent_id]

    # Try full type name first, then base type
    if type_name in symbol_map:
        return symbol_map[type_name]

    base = type_name.split(".")[0]
    return symbol_map.get(base, symbol_map.get("?", "❓"))


def compute_bounds(grid_objects: Dict[int, dict], object_type_names: list[str]) -> tuple[int, int, int, int]:
    """Compute bounding box for grid objects.

    Args:
        grid_objects: Dictionary of grid objects
        object_type_names: List mapping type IDs to names

    Returns:
        Tuple of (min_row, min_col, height, width)
    """
    rows = []
    cols = []
    for obj in grid_objects.values():
        type_name = object_type_names[obj["type"]]
        if type_name == "wall":
            for c, r, _ in _cells_for_object(obj):
                rows.append(r)
                cols.append(c)
    if not rows or not cols:
        for obj in grid_objects.values():
            for c, r, _ in _cells_for_object(obj):
                rows.append(r)
                cols.append(c)

    # Handle empty grid case
    if not rows or not cols:
        return (0, 0, 1, 1)

    min_row = min(rows)
    min_col = min(cols)
    height = max(rows) - min_row + 1
    width = max(cols) - min_col + 1
    return (min_row, min_col, height, width)


def build_grid_buffer(
    grid_objects: Dict[int, dict],
    object_type_names: list[str],
    symbol_map: dict[str, str],
    min_row: int,
    min_col: int,
    height: int,
    width: int,
    viewport_center_row: int | None = None,
    viewport_center_col: int | None = None,
    viewport_height: int | None = None,
    viewport_width: int | None = None,
    cursor_row: int | None = None,
    cursor_col: int | None = None,
) -> str:
    """Build the emoji grid buffer.

    Args:
        grid_objects: Dictionary of grid objects to render
        object_type_names: List mapping type IDs to names
        symbol_map: Map from object type names to render symbols
        min_row: Minimum row in full grid
        min_col: Minimum column in full grid
        height: Full grid height
        width: Full grid width
        viewport_center_row: Center row for viewport (None for full map)
        viewport_center_col: Center column for viewport (None for full map)
        viewport_height: Height of viewport (None for full map)
        viewport_width: Width of viewport (None for full map)
        cursor_row: Row position of selection cursor (None if not in select mode)
        cursor_col: Column position of selection cursor (None if not in select mode)

    Returns:
        Buffer string with newlines
    """
    # Determine viewport bounds
    if viewport_center_row is not None and viewport_height is not None:
        view_min_row = max(min_row, viewport_center_row - viewport_height // 2)
        view_max_row = min(min_row + height, view_min_row + viewport_height)
        view_height = view_max_row - view_min_row
    else:
        view_min_row = min_row
        view_height = height
        view_max_row = min_row + height

    if viewport_center_col is not None and viewport_width is not None:
        view_min_col = max(min_col, viewport_center_col - viewport_width // 2)
        view_max_col = min(min_col + width, view_min_col + viewport_width)
        view_width = view_max_col - view_min_col
    else:
        view_min_col = min_col
        view_width = width
        view_max_col = min_col + width

    # Initialize grid with empty spaces
    empty_symbol = symbol_map.get("empty", "⬜")
    grid = [[empty_symbol for _ in range(view_width)] for _ in range(view_height)]

    # Place objects in viewport using all occupied cells (anchor + extras)
    for obj in grid_objects.values():
        for obj_c, obj_r, _ in _cells_for_object(obj):
            r = obj_r - view_min_row
            c = obj_c - view_min_col
            if 0 <= r < view_height and 0 <= c < view_width:
                grid[r][c] = get_symbol_for_object(obj, object_type_names, symbol_map)

    # Add selection cursor if in select mode
    if cursor_row is not None and cursor_col is not None:
        cursor_r = cursor_row - view_min_row
        cursor_c = cursor_col - view_min_col
        if 0 <= cursor_r < view_height and 0 <= cursor_c < view_width:
            cursor_symbol = symbol_map.get("cursor", "🎯")
            grid[cursor_r][cursor_c] = cursor_symbol

    # Add directional arrows at edges if there's more content beyond viewport
    has_more_top = view_min_row > min_row
    has_more_bottom = view_max_row < min_row + height
    has_more_left = view_min_col > min_col
    has_more_right = view_max_col < min_col + width

    # Replace edges with full-width arrows
    if has_more_top:
        for c in range(view_width):
            grid[0][c] = "🔼"
    if has_more_bottom:
        for c in range(view_width):
            grid[view_height - 1][c] = "🔽"
    if has_more_left:
        for r in range(view_height):
            grid[r][0] = "◀️"
    if has_more_right:
        for r in range(view_height):
            grid[r][view_width - 1] = "▶️"

    # Handle corners - show diagonal arrows when both edges have more content
    if has_more_top and has_more_left:
        grid[0][0] = "↖️"
    if has_more_top and has_more_right:
        grid[0][view_width - 1] = "↗️"
    if has_more_bottom and has_more_left:
        grid[view_height - 1][0] = "↙️"
    if has_more_bottom and has_more_right:
        grid[view_height - 1][view_width - 1] = "↘️"

    lines = ["".join(row) for row in grid]
    return "\n".join(lines)
