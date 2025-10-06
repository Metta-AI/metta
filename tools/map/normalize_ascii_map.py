#!/usr/bin/env -S uv run
import argparse
from collections import OrderedDict
from pathlib import Path

import yaml

from mettagrid.mapgen.utils.ascii_grid import LEGEND_KEY, MAP_KEY, parse_ascii_map
from mettagrid.util.char_encoder import normalize_grid_char


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-place", action="store_true", help="Overwrite the map file with the normalized version")
    parser.add_argument("map_file", type=Path)
    args = parser.parse_args()

    with open(args.map_file, "r", encoding="utf-8") as f:
        content = f.read()

    map_lines, legend_map = parse_ascii_map(content)

    normalized_lines = ["".join(normalize_grid_char(c) for c in line) for line in map_lines]

    data = OrderedDict()
    data[MAP_KEY] = "\n".join(normalized_lines)
    data[LEGEND_KEY] = OrderedDict(sorted(legend_map.items())) if legend_map else {}

    output = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    if args.in_place:
        with open(args.map_file, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
