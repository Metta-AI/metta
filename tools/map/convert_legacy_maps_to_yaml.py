#!/usr/bin/env python3
"""Convert legacy ASCII maps (plain text + legend footer) to YAML format."""

import argparse
from pathlib import Path
from types import ModuleType
from typing import Iterable

from cogames.cogs_vs_clips import missions, stations
from mettagrid.builder import building, empty_assemblers
from mettagrid.mapgen.utils.ascii_grid import (
    DEFAULT_CHAR_TO_NAME as ASCII_GRID_DEFAULT_CHAR_TO_NAME,
)

LEGEND_PREFIX = "#:"
LEGEND_HEADER = "map legend:"
MAP_KEY = "map_data"
LEGEND_KEY = "char_to_name_map"
DEFAULT_DIRECTORIES = (
    Path("packages/mettagrid/configs/maps"),
    Path("packages/cogames/src/cogames/maps"),
)
DEFAULT_TYPE_LINE = "type: mettagrid.map_builder.ascii.AsciiMapBuilder"
ASCII_GRID_LEGEND_KEY = "ascii_grid"
COGS_VS_CLIPS_LEGEND_KEY = "cogs_vs_clips"
COGS_VS_CLIPS_PATH_MARKER = Path("packages/cogames")


def _build_cogs_vs_clips_char_to_name() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for config in missions._get_default_map_objects().values():
        token = getattr(config, "map_char", None)
        name = getattr(config, "name", None)
        if not token or not name:
            continue
        if len(token) != 1:
            raise ValueError(f"Mission legend token must be single character, got {token!r}")
        mapping[token] = name

    # Add common characters that are used in cogames maps
    mapping["."] = "empty"
    mapping["#"] = "wall"
    mapping["@"] = "agent.agent"

    return mapping


def _collect_additional_char_mappings(modules: Iterable[ModuleType]) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    for module in modules:
        for attr_name in dir(module):
            attr = getattr(module, attr_name, None)
            token = getattr(attr, "map_char", None)
            name = getattr(attr, "name", None)
            if isinstance(token, str) and len(token) == 1 and isinstance(name, str):
                mapping.setdefault(token, set()).add(name)
    return {token: sorted(names) for token, names in mapping.items()}


# These mappings are compiled for future use when reconciling diverse legend sources.
# They are not consumed today so downstream behavior remains unchanged.
ADDITIONAL_CHAR_MAPPINGS = _collect_additional_char_mappings(modules=(building, empty_assemblers, stations, missions))


LEGEND_PRESETS: dict[str, dict[str, str]] = {
    ASCII_GRID_LEGEND_KEY: (
        dict(ASCII_GRID_DEFAULT_CHAR_TO_NAME) | {token: names[0] for token, names in ADDITIONAL_CHAR_MAPPINGS.items()}
    ),
    COGS_VS_CLIPS_LEGEND_KEY: _build_cogs_vs_clips_char_to_name(),
}


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


def parse_yaml_map(text: str) -> tuple[str, list[str], dict[str, str]]:
    type_line = DEFAULT_TYPE_LINE
    legend: dict[str, str] = {}
    map_lines: list[str] = []
    section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if not stripped and section != "map":
            continue

        if stripped.startswith("type:") and section is None:
            type_line = stripped
            continue

        if stripped.startswith(f"{MAP_KEY}:"):
            section = "map"
            continue

        if stripped.startswith(f"{LEGEND_KEY}:"):
            section = "legend"
            continue

        if section == "map":
            if line.startswith("  "):
                map_lines.append(line[2:])
                continue
            section = None

        if section == "legend":
            if not line.startswith("  "):
                section = None
                continue

            entry = line.strip()
            if not entry:
                continue

            token_part, _, value_part = entry.partition(":")
            token = token_part.strip().strip("'\"")
            value = value_part.strip()
            if len(token) != 1:
                raise ValueError(f"Legend token must be a single character: {line!r}")
            legend[token] = value

    if not map_lines:
        raise ValueError("YAML map missing map_data section")

    return type_line, map_lines, legend


def legend_from_key(key: str) -> dict[str, str]:
    try:
        base = LEGEND_PRESETS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown legend preset: {key}") from exc
    return dict(base)


def infer_default_legend_key(path: Path) -> str:
    try:
        relative = path.relative_to(Path.cwd())
    except ValueError:
        relative = path

    marker_posix = COGS_VS_CLIPS_PATH_MARKER.as_posix()

    def matches(candidate: Path) -> bool:
        candidate_posix = candidate.as_posix()
        if candidate_posix == marker_posix:
            return True
        if candidate_posix.startswith(f"{marker_posix}/"):
            return True
        return f"/{marker_posix}/" in candidate_posix

    if matches(relative) or matches(path):
        return COGS_VS_CLIPS_LEGEND_KEY

    return ASCII_GRID_LEGEND_KEY


def ordered_chars(map_lines: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for line in map_lines:
        for char in line:
            if char not in seen:
                seen.add(char)
                ordered.append(char)
    return ordered


def make_yaml(map_lines: list[str], legend: dict[str, str], type_line: str = DEFAULT_TYPE_LINE) -> str:
    map_block = "\n".join(f"  {line}" for line in map_lines)
    legend_block = "\n".join(f'  "{token}": {name}' for token, name in legend.items())
    return f"{type_line}\n{MAP_KEY}: |-\n{map_block}\n{LEGEND_KEY}:\n{legend_block}\n"


def load_map_components(raw_text: str) -> tuple[str, list[str], dict[str, str]]:
    if f"{MAP_KEY}:" in raw_text:
        return parse_yaml_map(raw_text)

    legend_lines, map_section = split_legacy_sections(raw_text)
    legend_map = parse_legend(legend_lines)
    return DEFAULT_TYPE_LINE, map_section, legend_map


def convert_map(path: Path, dry_run: bool) -> bool:
    raw_text = path.read_text(encoding="utf-8")
    type_line, map_section, _ = load_map_components(raw_text)
    map_lines = normalize_map_lines(map_section)
    default_legend_key = infer_default_legend_key(path)
    default_legend = legend_from_key(default_legend_key)

    final_legend: dict[str, str] = {}
    for char in ordered_chars(map_lines):
        default_value = default_legend.get(char)

        if default_value is not None:
            value = default_value
        else:
            raise ValueError(f"Missing legend entry for character {char!r} in {path}")
        if any(ch.isspace() for ch in value):
            raise ValueError(f"Legend value contains whitespace for {char!r} in {path}")
        final_legend[char] = value

    yaml_text = make_yaml(map_lines, final_legend, type_line)

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
    parser.add_argument("files", nargs="*", type=Path, help="Specific map files to convert")
    parser.add_argument("--all", action="store_true", help="Convert all map files in default directories")
    parser.add_argument("--dry-run", action="store_true", help="Report files that would change")
    args = parser.parse_args()

    if args.all:
        search_roots = list(DEFAULT_DIRECTORIES)
        files = collect_targets(search_roots)
    elif args.files:
        files = collect_targets(args.files)
    else:
        parser.error("Must specify either individual files or --all")

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
