#!/usr/bin/env python3
"""
Sync child repositories from monorepo with preserved git history.

Usage:
    uv run devops/git/sync_package.py filter_repo_test
    uv run devops/git/sync_package.py filter_repo_test --dry-run
    uv run devops/git/sync_package.py filter_repo_test --yes  # Skip confirmations

Add new repos to CONFIG_REGISTRY below.
"""

import argparse
import sys
from pathlib import Path

from pydantic import field_validator

from metta.common.util.config import Config
from metta.common.util.git_filter import filter_repo
from metta.common.util.git import (
    run_git,
    run_git_in_dir,
    get_file_list,
    get_commit_count,
    add_remote
)


class FilterRepoConfig(Config):
    """Child repository configuration."""
    __init__ = Config.__init__

    name: str
    paths: list[str]
    remote: str

    @field_validator('paths')
    @classmethod
    def normalize_paths(cls, v: list[str]) -> list[str]:
        """Ensure paths end with /."""
        return [p.rstrip('/') + '/' for p in v]

    @field_validator('remote')
    @classmethod
    def validate_remote(cls, v: str) -> str:
        """Basic validation of git remote URL."""
        if not v.startswith(('git@', 'https://', 'http://')):
            raise ValueError("Remote must start with git@, https://, or http://")
        if 'Metta-AI/' not in v:
            raise ValueError("Remote must be from Metta-AI organization")
        return v


# Add new child repositories here
CONFIG_REGISTRY = {
    'filter_repo_test': FilterRepoConfig(
        name="filter_repo_test",
        paths=["mettagrid", "mettascope"],
        remote="git@github.com:Metta-AI/test_filter_repo.git"
    ),
}


def sync_repo(config_name: str, dry_run: bool = False, skip_confirmation: bool = False):
    """Filter and push repository subset to configured remote."""

    # Load config
    if config_name not in CONFIG_REGISTRY:
        available = ', '.join(CONFIG_REGISTRY.keys())
        print(f"Unknown config: {config_name}")
        print(f"Available: {available}")
        sys.exit(1)

    config = CONFIG_REGISTRY[config_name]

    print(f"Syncing: {config.name}")
    print(f"Paths: {', '.join(config.paths)}")
    print(f"Target: {config.remote}")

    # Step 1: Filter
    print("\nFiltering repository...")
    try:
        filtered_path = filter_repo(Path.cwd(), config.paths)
    except Exception as e:
        print(f"Filter failed: {e}")
        sys.exit(1)

    # Step 2: Show what we got
    files = get_file_list(filtered_path)
    commits = get_commit_count(filtered_path)
    print(f"Result: {len(files)} files, {commits} commits")

    # Step 3: Safety checks before push
    try:
        current_origin = run_git("remote", "get-url", "origin").strip()
        if current_origin and config.remote.rstrip('/').rstrip('.git') == current_origin.rstrip('/').rstrip('.git'):
            print("\n*** SAFETY STOP ***")
            print("Target remote matches current origin!")
            print("This would destroy the main repository.")
            sys.exit(1)
    except:
        pass  # No origin is fine

    # Step 4: Push (with confirmations)
    print(f"\n{'DRY RUN: ' if dry_run else ''}Push to {config.remote}")

    if not dry_run and not skip_confirmation:
        print("\nThis will FORCE PUSH and replace the target repository!")
        print("Type the target URL to confirm:")
        if input("> ").strip() != config.remote:
            print("Aborted - URLs don't match")
            sys.exit(1)

        if input("Final confirmation [y/N]: ").lower() != 'y':
            print("Aborted")
            sys.exit(1)

    # Do the push
    try:
        add_remote("production", config.remote, filtered_path)

        push_cmd = ["push", "--force", "production", "HEAD:main"]
        if dry_run:
            push_cmd.insert(2, "--dry-run")

        output = run_git_in_dir(filtered_path, *push_cmd)
        if output:
            print(output)

        print(f"\n{'Dry run' if dry_run else 'Push'} completed!")
        print(f"Filtered repo at: {filtered_path}")

    except Exception as e:
        print(f"Push failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('config', help='Repository config name')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be pushed')
    parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompts')

    args = parser.parse_args()

    try:
        sync_repo(args.config, args.dry_run, args.yes)
    except KeyboardInterrupt:
        print("\nAborted")
        sys.exit(1)


if __name__ == "__main__":
    main()
