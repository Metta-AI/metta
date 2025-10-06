#!/usr/bin/env -S uv run
import argparse
from pathlib import Path

from mettagrid.mapgen.utils.ascii_grid import (
    build_legend_lines,
    parse_legend_lines,
    split_ascii_map_sections,
)
from mettagrid.util.char_encoder import normalize_grid_char


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-place", action="store_true", help="Overwrite the map file with the normalized version")
    parser.add_argument("map_file", type=Path)
    args = parser.parse_args()

    with open(args.map_file, "r", encoding="utf-8") as f:
        content = f.read()

    legend_lines, body_lines = split_ascii_map_sections(content)
    legend_map = parse_legend_lines(legend_lines)

    normalized_map_lines = ["".join(normalize_grid_char(c) for c in line) for line in body_lines]
    normalized_lines: list[str] = normalized_map_lines

    if legend_map:
        normalized_lines.append("")
        normalized_lines.append("map legend:")
        normalized_lines.extend(build_legend_lines(legend_map))
    elif legend_lines:
        normalized_lines.append("")
        normalized_lines.extend(legend_lines)

    if args.in_place:
        with open(args.map_file, "w", encoding="utf-8") as f:
            f.write("\n".join(normalized_lines))
    else:
        print("\n".join(normalized_lines))


if __name__ == "__main__":
    main()
