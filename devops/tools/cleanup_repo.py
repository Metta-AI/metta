#!/usr/bin/env -S uv run
"""
Script to clean empty directories and directories containing only __pycache__
"""

import argparse
import os
import pathlib
import shutil

import metta.common.util.text_styles


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
        print(metta.common.util.text_styles.red(f"Permission denied: {dir_path}"))
        return False


def clean_directory(root_path, dry_run=True):
    """
    Recursively clean empty directories and those with only __pycache__
    """
    removed_dirs = []

    # Walk through directory tree bottom-up
    for dirpath, _, _ in os.walk(root_path, topdown=False):
        dir_path = pathlib.Path(dirpath)

        # Skip .git directory and its subdirectories
        if ".git" in dir_path.parts:
            continue

        # Skip build cache directories
        if any(part in [".turbo", ".vite-temp", "node_modules"] for part in dir_path.parts):
            continue

        if is_dir_empty_or_pycache_only(dirpath):
            if dry_run:
                print(
                    f"{metta.common.util.text_styles.yellow('Would remove:')} {metta.common.util.text_styles.cyan(dirpath)}"
                )
            else:
                try:
                    shutil.rmtree(dirpath)
                    print(
                        f"{metta.common.util.text_styles.green('Removed:')} {metta.common.util.text_styles.cyan(dirpath)}"
                    )
                    removed_dirs.append(dirpath)
                except Exception as e:
                    print(
                        f"{metta.common.util.text_styles.red('Error removing')} {metta.common.util.text_styles.cyan(dirpath)}: {metta.common.util.text_styles.red(str(e))}"
                    )

    return removed_dirs


def main():
    parser = argparse.ArgumentParser(description="Clean empty directories and directories containing only __pycache__")
    parser.add_argument("path", nargs="?", default=".", help="Path to the repository (default: current directory)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually removing")
    parser.add_argument("--verbose", action="store_true", help="Show more information")

    args = parser.parse_args()

    root_path = os.path.abspath(args.path)

    if not os.path.exists(root_path):
        print(metta.common.util.text_styles.red(f"Error: Path '{root_path}' does not exist"))
        return 1

    if not os.path.isdir(root_path):
        print(metta.common.util.text_styles.red(f"Error: Path '{root_path}' is not a directory"))
        return 1

    if args.verbose:
        print(
            metta.common.util.text_styles.bold(f"Scanning directory: {metta.common.util.text_styles.cyan(root_path)}")
        )
        print(metta.common.util.text_styles.magenta("=" * 50))

    # Default is to remove, use --dry-run to preview
    dry_run = args.dry_run

    if dry_run:
        print(metta.common.util.text_styles.yellow("DRY RUN MODE - No directories will be removed"))
        print(metta.common.util.text_styles.magenta("=" * 50))

    removed = clean_directory(root_path, dry_run=dry_run)

    if args.verbose:
        print(metta.common.util.text_styles.magenta("=" * 50))
    if not dry_run:
        if removed:
            print(
                metta.common.util.text_styles.bold(
                    metta.common.util.text_styles.green(f"Total directories removed: {len(removed)}")
                )
            )
        elif args.verbose:
            print(metta.common.util.text_styles.yellow("No directories to remove"))
    else:
        if removed:
            print(
                metta.common.util.text_styles.yellow(
                    "Dry run complete. Run without --dry-run to remove these directories."
                )
            )
        else:
            # Dry run should print even if not verbose
            print(metta.common.util.text_styles.green("No directories to remove"))

    return 0


if __name__ == "__main__":
    exit(main())
