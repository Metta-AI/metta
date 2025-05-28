import numpy as np

from metta.map.types import MapGrid

ascii_symbols = {
    "empty": " ",
    "wall": "#",
    "agent.agent": "A",
    "mine": "g",
    "generator": "c",
    "altar": "a",
    "armory": "r",
    "lasery": "l",
    "lab": "b",
    "factory": "f",
    "temple": "t",
}

reverse_ascii_symbols = {v: k for k, v in ascii_symbols.items()}
reverse_ascii_symbols.update(
    {
        "🧱": "wall",
        "⚙": "generator",
        "⛩": "altar",
        "🏭": "factory",
        "🔬": "lab",
        "🏰": "temple",
    }
)


def grid_object_to_ascii(name: str) -> str:
    if name in ascii_symbols:
        return ascii_symbols[name]

    if name == "block":
        # FIXME - store maps in a different format, or pick a different character
        raise ValueError("Block is not supported in ASCII mode")

    if name.startswith("mine."):
        raise ValueError("Colored mines are not supported in ASCII mode")

    if name.startswith("agent."):
        raise ValueError("Agent groups are not supported in ASCII mode")

    raise ValueError(f"Unknown object type: {name}")


def ascii_to_grid_object(ascii: str) -> str:
    if ascii in reverse_ascii_symbols:
        return reverse_ascii_symbols[ascii]

    raise ValueError(f"Unknown character: {ascii}")


def grid_to_ascii(grid: MapGrid, border: bool = False) -> list[str]:
    lines: list[str] = []
    for r in range(grid.shape[0]):
        row = []
        for c in range(grid.shape[1]):
            row.append(grid_object_to_ascii(grid[r, c]))
        lines.append("".join(row))

    if border:
        width = len(lines[0])
        border_lines = ["┌" + "─" * width + "┐"]
        for row in lines:
            border_lines.append("│" + row + "│")
        border_lines.append("└" + "─" * width + "┘")
        lines = border_lines

    return lines


def ascii_to_grid(lines: list[str]) -> MapGrid:
    grid = np.full((len(lines), len(lines[0])), "empty", dtype="<U50")
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            grid[r, c] = ascii_to_grid_object(char)
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
