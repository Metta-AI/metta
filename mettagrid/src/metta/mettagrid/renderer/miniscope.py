from __future__ import annotations

from typing import Dict, List


class MiniscopeRenderer:
    """Emoji-based renderer for MettaGridEnv with full-width emoji support."""

    # Using emoji that are consistently rendered as double-width characters
    # These are selected to ensure visual clarity and consistent alignment
    MINISCOPE_SYMBOLS = {
        # Basic terrain
        "wall": "🧱",  # Wall/barrier (#)
        "empty": "⬜",  # Empty space (white square for visibility)
        # Agents
        "agent": "🤖",  # Default agent
        "agent.agent": "🤖",  # Standard agent (A)
        "agent.team_1": "🔵",  # Team 1 (1)
        "agent.team_2": "🔴",  # Team 2 (2)
        "agent.team_3": "🟢",  # Team 3 (3)
        "agent.team_4": "🟡",  # Team 4 (4)
        "agent.prey": "🐰",  # Prey agent (Ap)
        "agent.predator": "🦁",  # Predator agent (AP)
        # Resources and Items
        "mine_red": "🔺",  # Red mine (r)
        "mine_blue": "🔷",  # Blue mine (b)
        "mine_green": "💚",  # Green mine (g)
        # Generators/Converters
        "generator": "⚡",  # Generic generator (n)
        "generator_red": "🔋",  # Red generator (R)
        "generator_blue": "🔌",  # Blue generator (B)
        "generator_green": "🌿",  # Green generator (G)
        "converter": "🔄",  # Converter (c)
        # Special Objects
        "altar": "🎯",  # Altar/shrine (a) - using target instead of torii
        "block": "📦",  # Movable block (s)
        "lasery": "🔫",  # Laser weapon ('L' in ASCII maps)
        "factory": "🏭",  # Factory
        "lab": "🔬",  # Laboratory
        "temple": "🏛️",  # Temple
        # Markers and indicators
        "marker": "📍",  # Location marker ('m' in ASCII maps)
        "shrine": "🎌",  # Shrine/checkpoint ('s' in ASCII maps)
        "launcher": "🚀",  # Launcher
        # Fallback
        "?": "❓",  # Unknown object
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

    def _symbol_for(self, obj: dict) -> str:
        """Get the emoji symbol for an object."""
        type_name = self._object_type_names[obj["type"]]

        # Handle numbered agents specially
        if type_name.startswith("agent"):
            agent_id = obj.get("agent_id")
            if agent_id is not None and agent_id < 10:
                # Use colored squares for agents 0-9 (consistent width)
                agent_squares = ["🟦", "🟧", "🟩", "🟨", "🟪", "🟥", "🟫", "⬛", "🟦", "🟧"]
                if 0 <= agent_id < 10:
                    return agent_squares[agent_id]

        # Try full type name first, then base type
        if type_name in self.MINISCOPE_SYMBOLS:
            return self.MINISCOPE_SYMBOLS[type_name]

        base = type_name.split(".")[0]
        return self.MINISCOPE_SYMBOLS.get(base, self.MINISCOPE_SYMBOLS["?"])

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

    def _build_buffer(self, grid_objects: Dict[int, dict]) -> str:
        """Construct emoji map buffer without printing."""
        if not self._bounds_set:
            self._compute_bounds(grid_objects)
        grid = [[self.MINISCOPE_SYMBOLS["empty"] for _ in range(self._width)] for _ in range(self._height)]
        for obj in grid_objects.values():
            r = obj["r"] - self._min_row
            c = obj["c"] - self._min_col
            if 0 <= r < self._height and 0 <= c < self._width:
                grid[r][c] = self._symbol_for(obj)
        lines = ["".join(row) for row in grid]
        return "\n".join(lines)

    def render(self, step: int, grid_objects: Dict[int, dict]) -> str:
        """Render the environment buffer and print to screen."""
        current_buffer = self._build_buffer(grid_objects)
        header = f"🎮 Metta AI Miniscope - Step: {step} 🎮"
        separator = "═" * (self._width * 2)
        frame_buffer = f"\033[2J\033[H{header}\n{separator}\n{current_buffer}\n{separator}"
        print(frame_buffer, end="", flush=True)
        self._last_buffer = current_buffer
        return current_buffer

    def get_buffer(self, grid_objects: Dict[int, dict]) -> str:
        """Return emoji map buffer without side effects."""
        return self._build_buffer(grid_objects)
