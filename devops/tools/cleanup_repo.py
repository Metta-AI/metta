#!/usr/bin/env -S uv run
"""
Script to clean empty directories and directories containing only __pycache__
"""

import argparse
import os
import shutil
from pathlib import Path

from metta.common.util.colorama import bold, cyan, green, magenta, red, yellow


def is_dir_empty_or_pycache_only(dir_path):
    """
    Check if a directory is empty or contains only __pycache__
    """
    try:
        contents = list(os.listdir(dir_path))

        # Empty directory
        if not contents:
            return True

        # Only contains __pycache__
        if len(contents) == 1 and contents[0] == "__pycache__":
            return True

        return False
    except PermissionError:
        print(red(f"Permission denied: {dir_path}"))
        return False


def clean_directory(root_path, dry_run=True):
    """
    Recursively clean empty directories and those with only __pycache__
    """
    removed_dirs = []

    # Walk through directory tree bottom-up
    for dirpath, _, _ in os.walk(root_path, topdown=False):
        dir_path = Path(dirpath)

        # Skip .git directory and its subdirectories
        if ".git" in dir_path.parts:
            continue

        if is_dir_empty_or_pycache_only(dirpath):
            if dry_run:
                print(f"{yellow('Would remove:')} {cyan(dirpath)}")
            else:
                try:
                    shutil.rmtree(dirpath)
                    print(f"{green('Removed:')} {cyan(dirpath)}")
                    removed_dirs.append(dirpath)
                except Exception as e:
                    print(f"{red('Error removing')} {cyan(dirpath)}: {red(str(e))}")

    return removed_dirs


def main():
    parser = argparse.ArgumentParser(description="Clean empty directories and directories containing only __pycache__")
    parser.add_argument("path", nargs="?", default=".", help="Path to the repository (default: current directory)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually removing")

    args = parser.parse_args()

    root_path = os.path.abspath(args.path)

    if not os.path.exists(root_path):
        print(red(f"Error: Path '{root_path}' does not exist"))
        return 1

    if not os.path.isdir(root_path):
        print(red(f"Error: Path '{root_path}' is not a directory"))
        return 1

    print(bold(f"Scanning directory: {cyan(root_path)}"))
    print(magenta("=" * 50))

    # Default is to remove, use --dry-run to preview
    dry_run = args.dry_run

    if dry_run:
        print(yellow("DRY RUN MODE - No directories will be removed"))
        print(magenta("=" * 50))

    removed = clean_directory(root_path, dry_run=dry_run)

    print(magenta("=" * 50))
    if not dry_run:
        if removed:
            print(bold(green(f"Total directories removed: {len(removed)}")))
        else:
            print(yellow("No directories to remove"))
    else:
        if removed:
            print(yellow("Dry run complete. Run without --dry-run to remove these directories."))
        else:
            print(green("No directories to remove"))

    return 0


if __name__ == "__main__":
    exit(main())
