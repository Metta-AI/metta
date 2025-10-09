#!/usr/bin/env python3
"""Convert legacy ASCII maps (plain text + legend footer) to YAML format."""

import argparse
from pathlib import Path

DEFAULT_CHAR_TO_NAME = {
    "#": "wall",  # Default barrier in legacy ASCII layouts (tests/mapgen/scenes/fixtures/test.map).
    ".": "empty",  # Open floor shared by navigation maps and encoding.json.
    "@": "agent.agent",  # Single-agent spawn kept for compatibility with early tests.
    "p": "agent.prey",  # Prey glyph mirrored in gridworks/src/lib/encoding.json.
    "P": "agent.predator",  # Predator partner for the 'p' tile.
    "_": "altar",  # Altars in training facility fixtures under packages/cogames/.
    "c": "converter",  # Converter stations defined in machina/training facility maps.
    "C": "chest",  # Storage chests from the same object-use map legends.
    "Z": "assembler",  # Legacy assembler symbol retained for older clients.
    "1": "agent.team_1",  # Team-coloured agent used by rendering/miniscope tooling.
    "2": "agent.team_2",  # Team-coloured agent used by rendering/miniscope tooling.
    "3": "agent.team_3",  # Team-coloured agent used by rendering/miniscope tooling.
    "4": "agent.team_4",  # Team-coloured agent used by rendering/miniscope tooling.
    "m": "mine_red",  # Mines referenced in tests/mapgen/utils/test_storable_map.py.
    "n": "generator_red",  # Generator paired with the mine in the same fixtures.
    "S": "special",  # Special markers appearing in canidate*_stations maps.
    "s": "swappable_wall",  # Swappable wall sections in cave_base_50 and machina maps.
    "&": "altar",  # Alternate altar rune from the canidate map set.
    "+": "wall",  # Extra wall glyph in the canidate station maps.
    "O": "altar",  # Capital O altar pads in the canidate station layouts.
    "B": "wall",  # Additional wall rune from training_facility_clipped.map.
    "o": "altar",  # Lowercase altar indicator from machina/cave_base maps.
    "G": "generator",  # Generator pads in the canidate station maps.
    "N": "generator",  # Generator rune in training_facility_clipped.map.
    "g": "generator",  # Generator variant seen in machina/cave_base maps.
    "=": "wall",  # Decorative wall strip in the canidate station layouts.
    "D": "door",  # Door tile used in cave_base_50 and training_facility_clipped maps.
    "H": "wall",  # Supplemental wall rune from training_facility_clipped map exports.
    "T": "wall",  # Supplemental wall rune from training_facility_clipped map exports.
    "F": "wall",  # Supplemental wall rune from training_facility_clipped map exports.
    "R": "wall",  # Supplemental wall rune from training_facility_clipped map exports.
}

LEGEND_PREFIX = "#:"
LEGEND_HEADER = "map legend:"
MAP_KEY = "map_data"
LEGEND_KEY = "char_to_name_map"
DEFAULT_DIRECTORIES = (
    Path("packages/mettagrid/configs/maps"),
    Path("packages/cogames/src/cogames/maps"),
)


def split_legacy_sections(text: str) -> tuple[list[str], list[str]]:
    lines = [line.rstrip("\r") for line in text.splitlines()]

    for idx, line in enumerate(lines):
        if line.strip().lower() == LEGEND_HEADER:
            return lines[idx + 1 :], lines[:idx]

    legend_lines: list[str] = []
    map_lines: list[str] = []
    in_legend = True
    for line in lines:
        if in_legend and line.startswith(LEGEND_PREFIX):
            legend_lines.append(line)
            continue
        in_legend = False
        map_lines.append(line)
    return legend_lines, map_lines


def parse_legend(legend_lines: list[str]) -> dict[str, str]:
    legend: dict[str, str] = {}
    for line in legend_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(LEGEND_PREFIX):
            payload = stripped[len(LEGEND_PREFIX) :].strip()
            if not payload:
                continue
            token, _, name = payload.partition(":")
            token = token.strip().strip("'\"")
            name = name.strip()
        else:
            token = stripped[0]
            remainder = stripped[1:].lstrip()
            delimiter = remainder[:1]
            if delimiter not in {"=", ":"}:
                raise ValueError(f"Invalid legend line: {line!r}")
            name = remainder[1:].strip()
        if len(token) != 1 or not token:
            raise ValueError(f"Legend token must be a single character: {line!r}")
        legend[token] = name
    return legend


def normalize_map_lines(map_lines: list[str]) -> list[str]:
    lines = [line for line in map_lines if line.strip()]
    if not lines:
        raise ValueError("Map is empty")
    width = len(lines[0])
    if any(len(line) != width for line in lines):
        raise ValueError("Map lines must have consistent width")
    return lines


def ordered_chars(map_lines: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for line in map_lines:
        for char in line:
            if char not in seen:
                seen.add(char)
                ordered.append(char)
    return ordered


def make_yaml(map_lines: list[str], legend: dict[str, str]) -> str:
    map_block = "\n".join(f"  {line}" for line in map_lines)
    legend_block = "\n".join(f'  "{token}": {name}' for token, name in legend.items())
    return f"type: mettagrid.map_builder.ascii.AsciiMapBuilder\n{MAP_KEY}: |-\n{map_block}\n{LEGEND_KEY}:\n{legend_block}\n"


def convert_map(path: Path, dry_run: bool) -> bool:
    raw_text = path.read_text(encoding="utf-8")
    legend_lines, map_section = split_legacy_sections(raw_text)
    map_lines = normalize_map_lines(map_section)
    legend_map = parse_legend(legend_lines)

    final_legend: dict[str, str] = {}
    for char in ordered_chars(map_lines):
        if char in legend_map:
            value = legend_map[char]
        elif char in DEFAULT_CHAR_TO_NAME:
            value = DEFAULT_CHAR_TO_NAME[char]
        else:
            raise ValueError(f"Missing legend entry for character {char!r} in {path}")
        if any(ch.isspace() for ch in value):
            raise ValueError(f"Legend value contains whitespace for {char!r} in {path}")
        final_legend[char] = value

    yaml_text = make_yaml(map_lines, final_legend)

    if dry_run:
        if raw_text != yaml_text:
            print(f"[DRY RUN] Would convert {path}")
            return True
        return False

    if raw_text != yaml_text:
        path.write_text(yaml_text, encoding="utf-8")
        print(f"Converted {path}")
        return True
    return False


def collect_targets(paths: list[Path]) -> list[Path]:
    targets: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".map":
            targets.append(path)
        elif path.is_dir():
            targets.extend(sorted(path.rglob("*.map")))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy ASCII maps to YAML format.")
    parser.add_argument("paths", nargs="*", type=Path, help="Files or directories to convert")
    parser.add_argument("--dry-run", action="store_true", help="Report files that would change")
    args = parser.parse_args()

    search_roots = args.paths or list(DEFAULT_DIRECTORIES)
    files = collect_targets(search_roots)

    updated = 0
    for file in files:
        try:
            if convert_map(file, args.dry_run):
                updated += 1
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to convert {file}: {exc}")

    if args.dry_run:
        print(f"Dry run complete. {updated} file(s) would be updated.")
    else:
        print(f"Conversion complete. {updated} file(s) updated.")


if __name__ == "__main__":
    main()
