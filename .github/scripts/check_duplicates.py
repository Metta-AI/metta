#!/usr/bin/env python3

"""Utility to surface duplicate dependency declarations across the workspace."""

from __future__ import annotations

import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def extract_deps(toml_data: Mapping[str, Any]) -> dict[str, str]:
    """Return dependency specifiers indexed by normalized package name."""
    deps: dict[str, str] = {}
    project_section = toml_data.get("project")
    if isinstance(project_section, Mapping):
        dependencies = project_section.get("dependencies")
        if isinstance(dependencies, list):
            for dep in dependencies:
                if not isinstance(dep, str):
                    continue
                pkg_name = dep.split(">=")[0].split("==")[0].split("[")[0].strip()
                if pkg_name:
                    deps[pkg_name] = dep
    return deps


def main() -> int:
    pyproject_files = sorted(Path(".").rglob("*/pyproject.toml"))
    all_deps: defaultdict[str, dict[str, str]] = defaultdict(dict)

    for file_path in pyproject_files:
        try:
            with file_path.open("rb") as file_obj:
                data = tomllib.load(file_obj)
        except Exception as err:  # noqa: BLE001
            print(f"Warning: Could not parse {file_path}: {err}")
            continue

        deps = extract_deps(data)
        for pkg_name, dep_spec in deps.items():
            all_deps[pkg_name][str(file_path)] = dep_spec

    issues: list[str] = []
    for pkg_name, files_specs in all_deps.items():
        if len(files_specs) > 1:
            unique_specs = set(files_specs.values())
            if len(unique_specs) > 1:
                issues.append(f"❌ {pkg_name}: {dict(files_specs)}")
            else:
                print(
                    "✅ %s: consistent across %d files" % (pkg_name, len(files_specs)),
                    flush=True,
                )

    if issues:
        print("\n🚨 Dependency version inconsistencies found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nℹ️ Consider consolidating these dependencies to avoid conflicts")
        return 0  # Warning only, do not fail the CI

    print("✅ No dependency version inconsistencies found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
