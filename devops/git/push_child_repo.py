#!/usr/bin/env python3
"""
Sync a child repository from monorepo with preserved git history.

Usage:
    uv run devops/git/push_child_repo.py <repo>
    uv run devops/git/push_child_repo.py <repo> --dry-run
    uv run devops/git/push_child_repo.py <repo> --yes  # Skip confirmations

Assumes any repo to publish is in packages/<repo>.
"""

import argparse
import sys
from pathlib import Path

import gitta as git
from metta.common.util.constants import METTA_GITHUB_ORGANIZATION


def get_remote_url(package_name: str) -> str:
    return f"https://github.com/{METTA_GITHUB_ORGANIZATION}/{package_name}.git"


def sync_repo(package_name: str, dry_run: bool = False, skip_confirmation: bool = False):
    """Filter and push repository subset to configured remote."""

    # Assume all packages are in packages/<repo_name>
    package_path = f"packages/{package_name}"
    paths = [package_path + "/"]

    remote_url = get_remote_url(package_name)

    print(f"Syncing: {package_name}")
    print(f"Paths: {', '.join(paths)}")
    print(f"Target: {remote_url}")

    # Step 1: Filter
    print("\nFiltering repository...")
    try:
        # Filter to package path and make it the repository root
        filtered_path = git.filter_repo(Path.cwd(), paths, make_root=package_path + "/")
    except Exception as e:
        print(f"Filter failed: {e}")
        sys.exit(1)

    # Step 2: Show what we got
    files = git.get_file_list(filtered_path)
    commits = git.get_commit_count(filtered_path)
    print(f"Result: {len(files)} files, {commits} commits")

    # Step 3: Safety checks before push
    try:
        current_origin = git.run_git("remote", "get-url", "origin").strip()
        if current_origin and remote_url.rstrip("/").rstrip(".git") == current_origin.rstrip("/").rstrip(".git"):
            print("\n*** SAFETY STOP ***")
            print("Target remote matches current origin!")
            print("This would destroy the main repository.")
            sys.exit(1)
    except Exception:
        pass  # No origin is fine

    # Step 4: Push (with confirmations)
    print(f"\n{'DRY RUN: ' if dry_run else ''}Push to {remote_url}")

    if not dry_run and not skip_confirmation:
        print("\nThis will FORCE PUSH and replace the target repository!")
        print("Type the target URL to confirm:")
        if input("> ").strip() != remote_url:
            print("Aborted - URLs don't match")
            sys.exit(1)

        if input("Final confirmation [y/N]: ").lower() != "y":
            print("Aborted")
            sys.exit(1)

    # Do the push
    try:
        git.add_remote("production", remote_url, filtered_path)

        push_cmd = ["push", "--force", "production", "HEAD:main"]
        if dry_run:
            push_cmd.insert(2, "--dry-run")

        output = git.run_git_in_dir(filtered_path, *push_cmd)
        if output:
            print(output)

        print(f"\n{'Dry run' if dry_run else 'Push'} completed!")
        print(f"Filtered repo at: {filtered_path}")

    except Exception as e:
        print(f"Push failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", help="Package name (will sync packages/<package>)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pushed")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    args = parser.parse_args()

    try:
        sync_repo(args.package, args.dry_run, args.yes)
    except KeyboardInterrupt:
        print("\nAborted")
        sys.exit(1)


if __name__ == "__main__":
    main()
