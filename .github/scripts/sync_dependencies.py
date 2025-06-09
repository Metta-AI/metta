#!/usr/bin/env python3
"""
Version sync: Update mettagrid dependency versions to match metta versions
Uses a staged approach for better reliability and debugging
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Dependency:
    """Represents a parsed dependency with all its components."""

    name: str
    operator: str = ""  # ==, >=, <=, ~=, >, <
    version: str = ""
    extras: str = ""  # [standard], [dev], etc.

    def __str__(self) -> str:
        """Convert back to dependency string format."""
        extras_part = self.extras if self.extras else ""
        version_part = f"{self.operator}{self.version}" if self.operator and self.version else ""
        return f"{self.name}{extras_part}{version_part}"


def parse_dependency_line(line: str) -> Optional[Dependency]:
    """
    Parse a single dependency line into components.
    Examples:
      "boto3==1.38.32",
      "uvicorn[standard]==0.34.2",
      "pufferlib",
      "pytest>=8.3.3",
    """
    stripped = line.strip().rstrip(",").strip('"').strip("'")

    if not stripped or stripped.startswith("#"):
        return None

    # Handle extras like uvicorn[standard]
    extras = ""
    if "[" in stripped and "]" in stripped:
        bracket_start = stripped.find("[")
        bracket_end = stripped.find("]")
        extras = stripped[bracket_start : bracket_end + 1]
        base_dep = stripped[:bracket_start] + stripped[bracket_end + 1 :]
    else:
        base_dep = stripped

    # Parse version constraints
    operators = ["==", ">=", "<=", "~=", "!=", ">", "<"]

    for op in operators:
        if op in base_dep:
            name, version = base_dep.split(op, 1)
            return Dependency(name=name.strip(), operator=op, version=version.strip(), extras=extras)

    # No version constraint
    return Dependency(name=base_dep.strip(), extras=extras)


def parse_dependencies_from_content(content: str) -> List[Dependency]:
    """
    Stage 1: Parse all dependencies from a pyproject.toml content.
    Returns a list of Dependency objects.
    """
    lines = content.split("\n")
    dependencies = []
    in_deps_section = False

    for line in lines:
        stripped = line.strip()

        if stripped == "dependencies = [":
            in_deps_section = True
            continue
        elif in_deps_section and stripped == "]":
            break
        elif in_deps_section:
            dep = parse_dependency_line(line)
            if dep:
                dependencies.append(dep)

    return dependencies


def remove_dependencies_section(content: str) -> tuple[str, int, int]:
    """
    Stage 2: Remove the dependencies section from mettagrid pyproject.toml.
    Returns (content_without_deps, deps_start_line, deps_end_line).
    """
    lines = content.split("\n")

    # Find dependencies section boundaries
    deps_start = None
    deps_end = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "dependencies = [":
            deps_start = i
        elif deps_start is not None and stripped == "]":
            deps_end = i + 1  # Include the closing bracket
            break

    if deps_start is None or deps_end is None:
        raise ValueError("Could not find dependencies section in pyproject.toml")

    # Also remove any sync comments before dependencies
    comment_start = deps_start
    for i in range(deps_start - 1, -1, -1):
        line = lines[i].strip()
        if (
            line.startswith("# NOTE: Versions synced with metta")
            or line.startswith("# Dependencies list maintained manually")
            or line.startswith("# Run 'python scripts/sync_dependencies.py'")
        ):
            comment_start = i
        elif line and not line.startswith("#"):
            break
        elif not line:  # Empty line
            continue

    # Remove the section
    content_without_deps = "\n".join(lines[:comment_start] + lines[deps_end:])

    return content_without_deps, comment_start, deps_end


def sync_versions(mettagrid_deps: List[Dependency], metta_deps: List[Dependency]) -> tuple[List[Dependency], List[str]]:
    """
    Stage 3: Apply metta versions to mettagrid dependencies.
    Returns (updated_dependencies, list_of_changes).
    """
    # Create lookup map of metta dependencies
    metta_map = {dep.name: dep for dep in metta_deps}

    updated_deps = []
    changes = []

    for dep in mettagrid_deps:
        if dep.name in metta_map:
            metta_dep = metta_map[dep.name]

            # If metta has a version and it's different from mettagrid
            if metta_dep.version and dep.version != metta_dep.version:
                # Update to metta's version but keep mettagrid's extras
                updated_dep = Dependency(
                    name=dep.name,
                    operator=metta_dep.operator,
                    version=metta_dep.version,
                    extras=dep.extras,  # Preserve mettagrid's extras
                )
                updated_deps.append(updated_dep)
                changes.append(f"  {dep.name}: {dep.version or 'unpinned'} → {metta_dep.version}")
            else:
                # Keep as-is
                updated_deps.append(dep)
        else:
            updated_deps.append(dep)

    return updated_deps, changes


def recreate_dependencies_section(content_without_deps: str, dependencies: List[Dependency], insert_line: int) -> str:
    """
    Stage 4: Recreate the dependencies section in the pyproject.toml.
    """
    lines = content_without_deps.split("\n")

    comment_lines = [
        "# NOTE: Versions synced with metta pyproject.toml for compatibility.",
        "# Dependencies list maintained manually, versions updated automatically.",
        "# Run 'python scripts/sync_dependencies.py' to update versions.",
    ]

    deps_lines = ["dependencies = ["]

    for dep in dependencies:
        deps_lines.append(f'    "{dep}",')

    deps_lines.append("]")

    # Insert the new section
    new_content_lines = lines[:insert_line] + comment_lines + deps_lines + lines[insert_line:]

    return "\n".join(new_content_lines)


def main():
    """Main sync function using staged approach."""
    root_pyproject = Path("pyproject.toml")
    mettagrid_pyproject = Path("mettagrid/pyproject.toml")

    if not root_pyproject.exists():
        print("❌ Root pyproject.toml not found!")
        sys.exit(1)

    if not mettagrid_pyproject.exists():
        print("❌ mettagrid/pyproject.toml not found!")
        sys.exit(1)

    # Read both files
    root_content = root_pyproject.read_text()
    mettagrid_content = mettagrid_pyproject.read_text()

    print("🔍 Stage 1: Parsing dependencies...")

    # Stage 1: Parse dependencies from both files
    try:
        metta_deps = parse_dependencies_from_content(root_content)
        mettagrid_deps = parse_dependencies_from_content(mettagrid_content)
    except Exception as e:
        print(f"❌ Failed to parse dependencies: {e}")
        sys.exit(1)

    print(f"📦 Metta has {len(metta_deps)} dependencies")
    print(f"📦 Mettagrid has {len(mettagrid_deps)} dependencies")

    # Find shared dependencies
    metta_names = {dep.name for dep in metta_deps}
    mettagrid_names = {dep.name for dep in mettagrid_deps}
    shared_deps = mettagrid_names.intersection(metta_names)

    print(f"🔄 {len(shared_deps)} shared dependencies to sync:")
    for name in sorted(shared_deps):
        metta_dep = next(d for d in metta_deps if d.name == name)
        print(f"  • {name} (metta: {metta_dep.version or 'unpinned'})")

    print("\n🗑️  Stage 2: Removing existing dependencies section...")

    # Stage 2: Remove dependencies section
    try:
        content_without_deps, insert_line, _ = remove_dependencies_section(mettagrid_content)
    except Exception as e:
        print(f"❌ Failed to remove dependencies section: {e}")
        sys.exit(1)

    print("🔄 Stage 3: Syncing versions...")

    # Stage 3: Sync versions
    updated_deps, changes = sync_versions(mettagrid_deps, metta_deps)

    if not changes:
        print("✅ All shared dependency versions are already in sync!")
        return

    print(f"📝 Found {len(changes)} version updates:")
    for change in changes:
        print(change)

    print("\n🔧 Stage 4: Recreating dependencies section...")

    # Stage 4: Recreate dependencies section
    try:
        final_content = recreate_dependencies_section(content_without_deps, updated_deps, insert_line)
    except Exception as e:
        print(f"❌ Failed to recreate dependencies section: {e}")
        sys.exit(1)

    # Write updated file
    mettagrid_pyproject.write_text(final_content)
    print(f"✅ Updated {mettagrid_pyproject}")

    # Show summary
    print("\n📋 Summary:")
    print(f"  ✅ Synced {len(changes)} dependency versions with metta")
    print(f"  📦 Preserved {len(mettagrid_deps) - len(changes)} existing versions")

    print("\n🎉 Version sync completed successfully!")
    print("💡 Mettagrid dependencies are now compatible with metta versions")


if __name__ == "__main__":
    main()
