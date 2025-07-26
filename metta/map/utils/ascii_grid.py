"""
Utilities for working with ASCII grid map files.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from metta.map.types import MapGrid
from metta.mettagrid.char_encoder import CHAR_TO_NAME, char_to_grid_object, grid_object_to_char


def add_pretty_border(lines: list[str]) -> list[str]:
    width = len(lines[0])
    border_lines = ["┌" + "─" * width + "┐"]
    for row in lines:
        border_lines.append("│" + row + "│")
    border_lines.append("└" + "─" * width + "┘")
    lines = border_lines
    return lines


def grid_to_lines(grid: MapGrid, border: bool = False) -> list[str]:
    lines: list[str] = []
    for r in range(grid.shape[0]):
        row = []
        for c in range(grid.shape[1]):
            row.append(grid_object_to_char(grid[r, c]))
        lines.append("".join(row))
    if border:
        lines = add_pretty_border(lines)
    return lines


def print_grid(grid: MapGrid, border=True):
    lines = grid_to_lines(grid, border=border)
    for line in lines:
        print(line)


def lines_to_grid(lines: list[str]) -> MapGrid:
    grid = np.full((len(lines), len(lines[0])), "empty", dtype="<U50")
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            grid[r, c] = char_to_grid_object(char)
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


def validate_map_file(file_path: str | Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that a map file contains only known characters and has consistent line lengths.

    Args:
        file_path: Path to the map file

    Returns:
        tuple: (is_valid, error_message)
            - is_valid: True if file is valid, False otherwise
            - error_message: Description of validation error, or None if valid
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return False, "File is empty"

        lines = content.splitlines()

        # Check for consistent line lengths
        line_lengths = [len(line) for line in lines]
        if len(set(line_lengths)) > 1:
            min_len, max_len = min(line_lengths), max(line_lengths)
            return False, f"Inconsistent line lengths: {min_len}-{max_len}"

        # Check for unknown characters using the CHAR_TO_NAME mapping
        all_chars = set(content)
        known_chars = set(CHAR_TO_NAME.keys())
        unknown_chars = all_chars - known_chars - {"\n", "\r", "\t"}
        if unknown_chars:
            return False, f"Unknown characters found: {sorted(unknown_chars)}"

        return True, None

    except FileNotFoundError:
        return False, f"File not found: {file_path}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def load_map_file(file_path: str | Path) -> list[str]:
    """
    Load a map file and return its lines.

    Args:
        file_path: Path to the map file

    Returns:
        List of lines from the map file

    Raises:
        ValueError: If the map file is invalid
    """
    is_valid, error_msg = validate_map_file(file_path)
    if not is_valid:
        raise ValueError(f"Invalid map file: {error_msg}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


# Transformation functions for ASCII grids


def rotate_lines_90(lines: list[str]) -> list[str]:
    """Rotate lines 90 degrees clockwise (transpose then reverse each row)."""
    if not lines:
        return []

    max_length = max(len(line) for line in lines)
    padded_lines = [line.ljust(max_length) for line in lines]

    # Transpose then reverse each row
    rotated = []
    for col in range(max_length):
        new_row = "".join(line[col] if col < len(line) else " " for line in reversed(padded_lines))
        rotated.append(new_row)

    return rotated


def rotate_lines_180(lines: list[str]) -> list[str]:
    """Rotate lines 180 degrees (reverse order of lines and each line)."""
    return [line[::-1] for line in reversed(lines)]


def rotate_lines_270(lines: list[str]) -> list[str]:
    """Rotate lines 270 degrees clockwise (reverse each row then transpose)."""
    if not lines:
        return []

    max_length = max(len(line) for line in lines)
    padded_lines = [line.ljust(max_length) for line in lines]

    # Transpose with reversed column order
    rotated = []
    for col in range(max_length - 1, -1, -1):
        new_row = "".join(line[col] if col < len(line) else " " for line in padded_lines)
        rotated.append(new_row)

    return rotated


def mirror_lines_horizontal(lines: list[str]) -> list[str]:
    """Mirror lines horizontally (reverse each line)."""
    return [line[::-1] for line in lines]


def mirror_lines_vertical(lines: list[str]) -> list[str]:
    """Mirror lines vertically (reverse order of lines)."""
    return list(reversed(lines))


def rotate_grid_90(grid: MapGrid) -> MapGrid:
    """Rotate a MapGrid 90 degrees clockwise."""
    lines = grid_to_lines(grid)
    rotated_lines = rotate_lines_90(lines)
    return lines_to_grid(rotated_lines)


def rotate_grid_180(grid: MapGrid) -> MapGrid:
    """Rotate a MapGrid 180 degrees."""
    lines = grid_to_lines(grid)
    rotated_lines = rotate_lines_180(lines)
    return lines_to_grid(rotated_lines)


def rotate_grid_270(grid: MapGrid) -> MapGrid:
    """Rotate a MapGrid 270 degrees clockwise."""
    lines = grid_to_lines(grid)
    rotated_lines = rotate_lines_270(lines)
    return lines_to_grid(rotated_lines)


def mirror_grid_horizontal(grid: MapGrid) -> MapGrid:
    """Mirror a MapGrid horizontally."""
    lines = grid_to_lines(grid)
    mirrored_lines = mirror_lines_horizontal(lines)
    return lines_to_grid(mirrored_lines)


def mirror_grid_vertical(grid: MapGrid) -> MapGrid:
    """Mirror a MapGrid vertically."""
    lines = grid_to_lines(grid)
    mirrored_lines = mirror_lines_vertical(lines)
    return lines_to_grid(mirrored_lines)
