from __future__ import annotations

import yaml

from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.types import MapGrid


MAP_KEY = "map"
LEGEND_KEY = "legend"


def _validate_token(token: str) -> str:
    token = token.strip().strip("'\"")
    if len(token) != 1 or any(ch.isspace() for ch in token):
        raise ValueError(f"Legend token must be a single non-whitespace character: {token!r}")
    return token


def _validate_value(value: str) -> str:
    value = value.strip()
    if not value or any(ch.isspace() for ch in value):
        raise ValueError(f"Legend values must be non-empty and contain no whitespace: {value!r}")
    return value


def _map_lines_from_yaml(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        lines = value.splitlines()
    elif isinstance(value, list):
        lines = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("All map entries must be strings")
            lines.append(item)
    else:
        raise ValueError("'map' must be a string or list of strings")

    if not lines:
        raise ValueError("Map must contain at least one line")

    width = len(lines[0])
    if any(len(line) != width for line in lines):
        raise ValueError("All lines in the map must have the same length")

    return lines


def _legend_from_yaml(value: dict[str, str]) -> dict[str, str]:
    legend: dict[str, str] = {}
    for token_raw, name_raw in value.items():
        if not isinstance(token_raw, str) or not isinstance(name_raw, str):
            raise ValueError("Legend keys and values must be strings")
        token = _validate_token(token_raw)
        legend[token] = _validate_value(name_raw)
    return legend


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

    map_lines = _map_lines_from_yaml(data[MAP_KEY])
    legend_value = data[LEGEND_KEY]
    if not isinstance(legend_value, dict):
        raise ValueError("'legend' must be a mapping")

    legend_map = _legend_from_yaml(legend_value)
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


def print_grid(grid: MapGrid, name_to_char: dict[str, str], border: bool = True):
    """Print a grid using the provided name-to-char mapping."""
    lines = grid_to_lines(grid, name_to_char, border=border)
    for line in lines:
        print(line)


def lines_to_grid(lines: list[str], char_to_name: dict[str, str]) -> MapGrid:
    """Convert lines of text to a grid using the provided char-to-name mapping."""

    grid = create_grid(len(lines), len(lines[0]))
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            grid[r, c] = char_to_name.get(char, char)
    return grid


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
