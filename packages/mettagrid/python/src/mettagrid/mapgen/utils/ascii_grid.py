from __future__ import annotations

from collections.abc import Mapping

from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.types import MapGrid

GLOBAL_DEFAULT_MAPPINGS: dict[str, str] = {
    "#": "wall",
    ".": "empty",
    "@": "agent.agent",
}


def merge_with_global_defaults(char_to_name: Mapping[str, str]) -> dict[str, str]:
    """Merge a legend with the immutable global defaults."""
    merged: dict[str, str] = {**GLOBAL_DEFAULT_MAPPINGS}
    for char, name in char_to_name.items():
        default_name = GLOBAL_DEFAULT_MAPPINGS.get(char)
        if default_name is not None and default_name != name:
            raise ValueError(
                f"Cannot override global default mapping for '{char}': expected '{default_name}', received '{name}'."
            )
        merged[char] = name
    return merged


DEFAULT_CHAR_TO_NAME: dict[str, str] = {
    **GLOBAL_DEFAULT_MAPPINGS,
    "p": "agent.prey",
    "P": "agent.predator",
    "_": "assembler",
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


def lines_to_grid(lines: list[str], char_to_name: dict[str, str]) -> MapGrid:
    """Convert lines of text to a grid using the provided char-to-name mapping."""

    grid = create_grid(len(lines), len(lines[0]))
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            grid[r, c] = char_to_name.get(char, char)
    return grid
