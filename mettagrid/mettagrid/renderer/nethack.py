from __future__ import annotations

import unicodedata
from typing import Dict, List

from mettagrid.room.ascii import SYMBOLS as MAP_SYMBOLS


class NethackRenderer:
    """Simple NetHack-style renderer for ``MettaGridEnv`` with perfect alignment."""

    SYMBOLS = {v: k for k, v in MAP_SYMBOLS.items()}

    # NetHack-style conversion mapping for perfect alignment
    NETHACK_CONVERSION = {
        "🧱": "#",  # wall emoji → NetHack wall
        "⚙": "G",  # gear emoji → Generator
        "⛩": "_",  # torii emoji → Altar
        "🏭": "F",  # factory emoji → Factory
        "🔬": "L",  # microscope emoji → Lab
        "🏰": "T",  # castle emoji → Temple
        " ": ".",  # space → NetHack empty
    }

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

    def _is_double_width_char(self, char: str) -> bool:
        """Check if a character is double-width (like emojis)."""
        if not char:
            return False

        # Check Unicode category and width
        code_point = ord(char[0])

        # ASCII characters are always single-width
        if code_point <= 127:
            return False

        # Use unicodedata to check East Asian width
        east_asian_width = unicodedata.east_asian_width(char[0])
        return east_asian_width in ("F", "W")  # Fullwidth or Wide

    def _convert_to_nethack_style(self, char: str) -> str:
        """Convert double-width characters to NetHack-style single-width equivalents."""
        # Direct conversion mapping
        if char in self.NETHACK_CONVERSION:
            return self.NETHACK_CONVERSION[char]

        # If it's a double-width character without specific mapping, use a fallback
        if self._is_double_width_char(char):
            return "?"  # Fallback for unknown double-width chars

        return char

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

        # Get the original symbol
        original_symbol = self.SYMBOLS.get(base, "?")

        # Convert to NetHack-style for consistent alignment
        return self._convert_to_nethack_style(original_symbol)

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

    def render(self, step: int, grid_objects: Dict[int, dict]) -> str:
        if not self._bounds_set:
            self._compute_bounds(grid_objects)

        # Initialize grid with NetHack-style empty spaces (dots)
        grid = [["." for _ in range(self._width)] for _ in range(self._height)]

        for obj in grid_objects.values():
            r = obj["r"] - self._min_row
            c = obj["c"] - self._min_col
            if 0 <= r < self._height and 0 <= c < self._width:
                symbol = self._symbol_for(obj)
                # Ensure the symbol is single-width
                if self._is_double_width_char(symbol):
                    symbol = self._convert_to_nethack_style(symbol)
                grid[r][c] = symbol

        lines = ["".join(row) for row in grid]

        # Create current buffer
        current_buffer = "\n".join(lines)

        # Validate alignment (all lines should have same length)
        line_lengths = [len(line) for line in lines]
        if len(set(line_lengths)) > 1:
            # This should not happen with NetHack-style conversion, but log if it does
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
