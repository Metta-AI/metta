from __future__ import annotations

import yaml

from mettagrid.mapgen.types import MapGrid

MAP_KEY = "map_data"
LEGEND_KEY = "char_to_name_map"


def parse_ascii_map(text: str) -> tuple[list[str], dict[str, str]]:
    """Parse map content from YAML."""

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError("Map file must be valid YAML") from exc

    if not isinstance(data, dict):
        raise ValueError("Map file must be a YAML mapping with 'map' and 'legend'")

    if MAP_KEY not in data or LEGEND_KEY not in data:
        raise ValueError("Map YAML must include both 'map' and 'legend'")

    from mettagrid.map_builder.ascii import AsciiMapBuilder

    map_rows = AsciiMapBuilder.Config._normalize_map_data(data[MAP_KEY])
    legend_map = AsciiMapBuilder.Config._normalize_char_map(data[LEGEND_KEY])

    map_lines = ["".join(row) for row in map_rows]
    return map_lines, legend_map


DEFAULT_CHAR_TO_NAME: dict[str, str] = {
    "#": "wall",
    ".": "empty",
    "@": "agent.agent",
    "p": "agent.prey",
    "P": "agent.predator",
    "_": "altar",
    "c": "converter",
    "C": "chest",
    "Z": "assembler",
    "1": "agent.team_1",
    "2": "agent.team_2",
    "3": "agent.team_3",
    "4": "agent.team_4",
}


def default_char_to_name() -> dict[str, str]:
    """Default character-to-name mapping for common test scenarios.

    Deprecated: Use DEFAULT_CHAR_TO_NAME constant directly instead.
    """
    return DEFAULT_CHAR_TO_NAME.copy()


def add_pretty_border(lines: list[str]) -> list[str]:
    width = len(lines[0])
    border_lines = ["┌" + "─" * width + "┐"]
    for row in lines:
        border_lines.append("│" + row + "│")
    border_lines.append("└" + "─" * width + "┘")
    lines = border_lines
    return lines


def grid_to_lines(grid: MapGrid, name_to_char: dict[str, str] | None = None, border: bool = False) -> list[str]:
    """Convert a grid to lines of text using the provided name-to-char mapping."""
    if name_to_char is None:
        # Reverse the default char_to_name mapping to get name_to_char
        name_to_char = {name: char for char, name in DEFAULT_CHAR_TO_NAME.items()}

    lines: list[str] = []
    for r in range(grid.shape[0]):
        row = []
        for c in range(grid.shape[1]):
            obj_name = grid[r, c]
            row.append(name_to_char.get(obj_name, obj_name[0] if obj_name else "?"))
        lines.append("".join(row))

    if border:
        lines = add_pretty_border(lines)

    return lines


def char_grid_to_lines(text: str) -> tuple[list[str], int, int]:
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        lines.append(line)

    height = len(lines)
    width = max(len(line) for line in lines)
    if not all(len(line) == width for line in lines):
        raise ValueError("All lines must be the same width")

    return (lines, width, height)
