#!/usr/bin/env -S uv run
import argparse
from pathlib import Path

from metta.mettagrid.char_encoder import normalize_grid_char


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-place", action="store_true", help="Overwrite the map file with the normalized version")
    parser.add_argument("map_file", type=Path)
    args = parser.parse_args()

    with open(args.map_file, "r", encoding="utf-8") as f:
        content = f.read()

    normalized_lines = []
    lines = content.splitlines()
    for line in lines:
        normalized_line = "".join(normalize_grid_char(c) for c in line)
        normalized_lines.append(normalized_line)

    if args.in_place:
        with open(args.map_file, "w", encoding="utf-8") as f:
            f.write("\n".join(normalized_lines))
    else:
        print("\n".join(normalized_lines))


if __name__ == "__main__":
    main()
