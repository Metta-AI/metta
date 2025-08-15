from metta.mettagrid.char_encoder import grid_object_to_char


class NethackRenderer:
    """Simple NetHack-style renderer for ``MettaGridEnv`` with perfect alignment."""

    def __init__(self, object_type_names: list[str]):
        self._object_type_names = object_type_names
        self._bounds_set = False
        self._min_row = 0
        self._min_col = 0
        self._height = 0
        self._width = 0
        self._last_buffer = None
        # Clear screen and hide cursor on init
        print("\033[?25l", end="")  # Hide cursor
        print("\033[2J", end="")  # Clear screen
        print("\033[H", end="")  # Move to home position

    def __del__(self):
        # Show cursor when renderer is destroyed
        print("\033[?25h", end="")

    def _symbol_for(self, obj: dict) -> str:
        """Get the symbol for an object, with NetHack-style conversion."""
        type_name = self._object_type_names[obj["type"]]
        base = type_name.split(".")[0]

        if base == "agent":
            agent_id = obj.get("agent_id")
            if agent_id is not None:
                if agent_id < 10:
                    return str(agent_id)
                idx = (agent_id - 10) % 26
                return chr(ord("a") + idx)

        return grid_object_to_char(base)

    def _compute_bounds(self, grid_objects: dict[int, dict]):
        rows = []
        cols = []
        for obj in grid_objects.values():
            type_name = self._object_type_names[obj["type"]]
            if type_name == "wall":
                rows.append(obj["r"])
                cols.append(obj["c"])
        if not rows or not cols:
            for obj in grid_objects.values():
                rows.append(obj["r"])
                cols.append(obj["c"])

        # Handle empty grid case
        if not rows or not cols:
            self._min_row = 0
            self._min_col = 0
            self._height = 1
            self._width = 1
        else:
            self._min_row = min(rows)
            self._min_col = min(cols)
            self._height = max(rows) - self._min_row + 1
            self._width = max(cols) - self._min_col + 1

        self._bounds_set = True

    def render(self, step: int, grid_objects: dict[int, dict]) -> str:
        if not self._bounds_set:
            self._compute_bounds(grid_objects)

        # Initialize grid with NetHack-style empty spaces (dots)
        grid = [["." for _ in range(self._width)] for _ in range(self._height)]

        for obj in grid_objects.values():
            r = obj["r"] - self._min_row
            c = obj["c"] - self._min_col
            if 0 <= r < self._height and 0 <= c < self._width:
                symbol = self._symbol_for(obj)
                grid[r][c] = symbol

        lines = ["".join(row) for row in grid]

        # Create current buffer
        current_buffer = "\n".join(lines)

        # Validate alignment (all lines should have same length)
        line_lengths = [len(line) for line in lines]
        if len(set(line_lengths)) > 1:
            # This should not happen if we're not using emojis
            print(f"Warning: Inconsistent line lengths detected: {line_lengths}", flush=True)

        # Build the complete frame in memory first to eliminate flashing
        if self._last_buffer is None or current_buffer != self._last_buffer:
            # Build entire frame as single string with clear screen for atomic update
            frame_buffer = "\033[2J\033[H" + current_buffer

            # Write entire frame at once - atomic screen update
            print(frame_buffer, end="", flush=True)

            # Update last buffer
            self._last_buffer = current_buffer

        return current_buffer
