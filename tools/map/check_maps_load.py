#!/usr/bin/env python3
"""Validate that all ASCII map files can be loaded with the current builder."""

from __future__ import annotations

import argparse
from pathlib import Path

from mettagrid.map_builder.ascii import AsciiMapBuilder

DEFAULT_DIRECTORIES = (
    Path("packages/mettagrid/configs/maps"),
    Path("packages/cogames/src/cogames/maps"),
)


def iter_map_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".map":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.map")))
    return files


def check_map(path: Path) -> tuple[bool, str | None]:
    try:
        config = AsciiMapBuilder.Config.from_uri(str(path))
        builder = config.create()
        builder.build()
        return True, None
    except Exception as exc:  # pylint: disable=broad-except
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure all map files load successfully.")
    parser.add_argument("paths", nargs="*", type=Path, help="Files or directories to check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print every file as it is checked")
    args = parser.parse_args()

    targets = args.paths or list(DEFAULT_DIRECTORIES)
    files = iter_map_files(targets)

    failures: list[tuple[Path, str]] = []
    for path in files:
        ok, error = check_map(path)
        if ok:
            if args.verbose:
                print(f"OK: {path}")
        else:
            print(f"FAIL: {path}\n  {error}")
            failures.append((path, error))

    print()
    print(f"Checked {len(files)} map file(s).")
    if failures:
        print(f"Encountered {len(failures)} failure(s).")
        raise SystemExit(1)
    print("All maps loaded successfully.")


if __name__ == "__main__":
    main()
