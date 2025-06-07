import numpy as np

from metta.map.types import MapGrid
from mettagrid.char_encoder import char_to_grid_object, grid_object_to_char


def add_pretty_border(lines: list[str]) -> list[str]:
    width = len(lines[0])
    border_lines = ["┌" + "─" * width + "┐"]
    for row in lines:
        border_lines.append("│" + row + "│")
    border_lines.append("└" + "─" * width + "┘")
    lines = border_lines
    return lines


def grid_to_ascii(grid: MapGrid, border: bool = False) -> list[str]:
    lines: list[str] = []
    for r in range(grid.shape[0]):
        row = []
        for c in range(grid.shape[1]):
            row.append(grid_object_to_char(grid[r, c]))
        lines.append("".join(row))

    if border:
        lines = add_pretty_border(lines)

    return lines


def ascii_to_grid(lines: list[str]) -> MapGrid:
    grid = np.full((len(lines), len(lines[0])), "empty", dtype="<U50")
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            grid[r, c] = char_to_grid_object(char)
    return grid


def bordered_text_to_lines(text: str) -> tuple[list[str], int, int]:
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line[0] != "|" or line[-1] != "|":
            raise ValueError("Text must be enclosed in | characters")
        line = line[1:-1]
        lines.append(line)

    height = len(lines)
    width = max(len(line) for line in lines)
    if not all(len(line) == width for line in lines):
        raise ValueError("All lines must be the same width")

    return (lines, width, height)
