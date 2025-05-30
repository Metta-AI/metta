from __future__ import annotations

from typing import Dict, List

from mettagrid.room.ascii import SYMBOLS as MAP_SYMBOLS


class AsciiRenderer:
    """Simple ASCII renderer for ``MettaGridEnv``."""

    SYMBOLS = {v: k for k, v in MAP_SYMBOLS.items()}
    SYMBOLS["wall"] = "█"

    def __init__(self, object_type_names: List[str]):
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

    def _compute_bounds(self, grid_objects: Dict[int, dict]):
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
        self._min_row = min(rows)
        self._min_col = min(cols)
        self._height = max(rows) - self._min_row + 1
        self._width = max(cols) - self._min_col + 1
        self._bounds_set = True

    def _symbol_for(self, obj: dict) -> str:
        type_name = self._object_type_names[obj["type"]]
        base = type_name.split(".")[0]
        if base == "agent":
            agent_id = obj.get("agent_id")
            if agent_id is not None:
                if agent_id < 10:
                    return str(agent_id)
                idx = (agent_id - 10) % 26
                return chr(ord("a") + idx)
        return self.SYMBOLS.get(base, "?")

    def render(self, step: int, grid_objects: Dict[int, dict]) -> str:
        if not self._bounds_set:
            self._compute_bounds(grid_objects)
        grid = [[" "] * self._width for _ in range(self._height)]
        for obj in grid_objects.values():
            r = obj["r"] - self._min_row
            c = obj["c"] - self._min_col
            if 0 <= r < self._height and 0 <= c < self._width:
                grid[r][c] = self._symbol_for(obj)
        lines = ["".join(row) for row in grid]

        # Create current buffer
        current_buffer = "\n".join(lines)

        # If this is the first render or the buffer has changed
        if self._last_buffer is None or current_buffer != self._last_buffer:
            # Move to home position
            print("\033[H", end="")
            # Print the new buffer
            print(current_buffer, end="", flush=True)
            # Update last buffer
            self._last_buffer = current_buffer

        return current_buffer
