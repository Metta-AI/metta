#!/usr/bin/env -S uv run
"""
Publish a package to PyPI with automatic version bumping.

Usage:
    python devops/publish_to_pypi.py <package_name>
"""

import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path


def bump_last_version(version_str: str) -> str:
    """
    Bump the last version number (patch version).

    Examples:
        "0.1.0" -> "0.1.1"
        "1.2.3" -> "1.2.4"
        "0.2.0.2" -> "0.2.0.3"
        "0.2.1" -> "0.2.2"
    """
    parts = version_str.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid version format: {version_str}")

    # Increment the last part (patch or extra)
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


def update_pyproject_version(pyproject_path: Path) -> tuple[str, str]:
    """
    Update the version in pyproject.toml file.

    Returns:
        Tuple of (old_version, new_version)
    """
    content = pyproject_path.read_text()

    # Find current version using regex
    version_pattern = r'^version\s*=\s*["\']([\d.]+)["\']'
    match = re.search(version_pattern, content, re.MULTILINE)

    if not match:
        raise ValueError(f"Could not find version in {pyproject_path}")

    old_version = match.group(1)
    new_version = bump_last_version(old_version)

    # Replace version in content
    new_content = re.sub(version_pattern, f'version = "{new_version}"', content, count=1, flags=re.MULTILINE)

    # Write back to file
    pyproject_path.write_text(new_content)

    return old_version, new_version


def update_dependency_versions(
    project_root: Path,
    package_name: str,
    old_version: str,
    new_version: str,
    dry_run: bool = False,
    skip_path: Path = None,
) -> list[Path]:
    """
    Update all references to the package version in other pyproject.toml files.

    This will update any version specification for the package to use the new version,
    preserving the version operator (>=, ~=, ==, etc.).

    Returns:
        List of paths to updated files
    """
    updated_files = []

    # Find all pyproject.toml files in the repo
    pyproject_files = list(project_root.rglob("pyproject.toml"))

    for pyproject_file in pyproject_files:
        # Skip the package's own pyproject.toml
        if skip_path and pyproject_file == skip_path:
            continue

        content = pyproject_file.read_text()
        original_content = content

        # Patterns to match dependency specifications
        # These patterns will match ANY version number, not just the old_version
        patterns = [
            # In dependency list: "package>=version" or "package==version"
            (
                rf'(["\']){package_name}([~^>=<!=]+)([\d.]+)(["\'])',
                rf"\g<1>{package_name}\g<2>{new_version}\g<4>",
            ),
            # Simple version: package = ">=version" or package = "version"
            (
                rf'({package_name}\s*=\s*["\'])([~^>=<!=]*)([\d.]+)(["\'])',
                rf"\g<1>\g<2>{new_version}\g<4>",
            ),
            # Version in dict: package = {version = ">=version"}
            (
                rf'({package_name}\s*=\s*{{[^}}]*version\s*=\s*["\'])([~^>=<!=]*)([\d.]+)(["\'][^}}]*}})',
                rf"\g<1>\g<2>{new_version}\g<4>",
            ),
            # In extras/optional dependencies with ==
            (
                rf'(["\']){package_name}\[([^\]]*)\]==([\d.]+)(["\'])',
                rf"\g<1>{package_name}[\g<2>]=={new_version}\g<4>",
            ),
            # In extras/optional dependencies with other operators
            (
                rf'(["\']){package_name}\[([^\]]*)\]([~^>=<!=]+)([\d.]+)(["\'])',
                rf"\g<1>{package_name}[\g<2>]\g<3>{new_version}\g<5>",
            ),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)

        if content != original_content:
            if not dry_run:
                pyproject_file.write_text(content)
                updated_files.append(pyproject_file)
                print(f"  ✓ Updated {pyproject_file.relative_to(project_root)}")
            else:
                print(f"  (dry-run: would update {pyproject_file.relative_to(project_root)})")
                updated_files.append(pyproject_file)

    return updated_files


def run_command(cmd: list[str], cwd: Path = None) -> None:
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(f"Error: {result.stderr}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def main():
    parser = argparse.ArgumentParser(description="Publish a package to PyPI with automatic version bumping")
    parser.add_argument("package_name", help="Name of the package to publish (e.g., 'cogames')")
    parser.add_argument("--no-bump", action="store_true", help="Skip version bumping")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually publishing")

    args = parser.parse_args()

    # Find project root (where this script is in devops/)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    # Find package directory
    package_dir = project_root / "packages" / args.package_name
    if not package_dir.exists():
        print(f"Error: Package directory not found: {package_dir}", file=sys.stderr)
        sys.exit(1)

    pyproject_path = package_dir / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"Error: pyproject.toml not found: {pyproject_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Step 1: Bump version
        old_version = None
        if not args.no_bump:
            # Read current version
            content = pyproject_path.read_text()
            match = re.search(r'^version\s*=\s*["\']([\d.]+)["\']', content, re.MULTILINE)
            if not match:
                raise ValueError(f"Could not find version in {pyproject_path}")
            old_version = match.group(1)
            new_version = bump_last_version(old_version)

            if not args.dry_run:
                # Actually update the file
                update_pyproject_version(pyproject_path)
                print(f"✓ Bumped version from {old_version} to {new_version}")
            else:
                print(f"(dry-run: would bump version from {old_version} to {new_version})")
        else:
            # Read current version without bumping
            content = pyproject_path.read_text()
            match = re.search(r'^version\s*=\s*["\']([\d.]+)["\']', content, re.MULTILINE)
            old_version = new_version = match.group(1) if match else "unknown"
            print(f"Current version: {new_version}")

        # Step 2: Sync dependencies
        print("\n📦 Syncing dependencies...")
        if not args.dry_run:
            run_command(["uv", "sync"], cwd=project_root)
            print("✓ Dependencies synced")
        else:
            print("(dry-run: would run 'uv sync')")

        # Step 3: Build the package
        print(f"\n🔨 Building {args.package_name}...")
        if not args.dry_run:
            run_command(["uv", "build"], cwd=package_dir)
            print("✓ Package built")
        else:
            print(f"(dry-run: would run 'uv build' in {package_dir})")

        # Step 4: Publish to PyPI
        print(f"\n🚀 Publishing {args.package_name} v{new_version} to PyPI...")
        if not args.dry_run:
            # The dist files are created in the project root's dist/ directory
            dist_dir = project_root / "dist"
            dist_files = [
                dist_dir / f"{args.package_name}-{new_version}-py3-none-any.whl",
                dist_dir / f"{args.package_name}-{new_version}.tar.gz",
            ]

            # Check if the expected files exist
            existing_files = [f for f in dist_files if f.exists()]
            if not existing_files:
                # Try to find any matching files in case of different naming
                pattern = str(dist_dir / f"{args.package_name}-{new_version}*")
                existing_files = glob.glob(pattern)
                if not existing_files:
                    raise FileNotFoundError(
                        f"No distribution files found for {args.package_name} v{new_version} in {dist_dir}"
                    )

            # Convert paths to strings for the command
            file_args = [str(f) for f in existing_files]
            run_command(["uv", "publish"] + file_args, cwd=project_root)
            print(f"✓ Successfully published {args.package_name} v{new_version} to PyPI!")
        else:
            print(f"(dry-run: would run 'uv publish' in {package_dir})")

        # Step 5: Update dependency versions in other packages
        updated_files = []
        if old_version and old_version != new_version:
            print(f"\n🔄 Updating {args.package_name} version references from {old_version} to {new_version}...")
            updated_files = update_dependency_versions(
                project_root, args.package_name, old_version, new_version, args.dry_run, skip_path=pyproject_path
            )
            if updated_files:
                print(f"✓ Updated {len(updated_files)} file(s)")
            else:
                print("  No dependency references found to update")

        if args.dry_run:
            print("\n✓ Dry run completed. No changes were made.")
        else:
            print(f"\n✨ Done! {args.package_name} v{new_version} is now available on PyPI.")

            # Collect files to commit
            files_to_commit = [pyproject_path.relative_to(project_root)]
            if updated_files:
                files_to_commit.extend([f.relative_to(project_root) for f in updated_files if f.exists()])

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed: {' '.join(e.cmd)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
