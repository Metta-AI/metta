#!/usr/bin/env python3
"""
Extract the `mettagrid/` subdirectory as a standalone repo (history preserved)
and push it to a target repository.

Usage:
    # Dry run (no push)
    uv run devops/git/push_child_repo.py Metta-AI/mettagrid --dry-run

    # Force push to target (with confirmations unless -y)
    uv run devops/git/push_child_repo.py Metta-AI/mettagrid -y

Target can be either an HTTPS remote URL or an owner/repo slug.
Only HTTPS is supported (no SSH).
"""

import argparse
import sys
from pathlib import Path

import gitta as git

SUBDIR = "mettagrid"


def build_remote_url(target: str) -> str:
    """Build HTTPS remote URL from input target.

    - If target is an HTTPS URL, return as-is
    - If target is an owner/repo slug, return https://github.com/owner/repo.git
    - Otherwise, exit with an error (SSH and non-HTTPS URLs are not supported)
    """
    # HTTPS URL provided
    if target.startswith("https://"):
        return target

    # Disallow SSH and non-HTTPS URLs
    if target.startswith("git@") or target.startswith("ssh://") or target.startswith("http://"):
        raise SystemExit("Only HTTPS remotes are supported. Provide an HTTPS URL or 'owner/repo' slug.")

    # Expect owner/repo slug
    if "/" not in target:
        raise SystemExit("Target must be an HTTPS URL or an 'owner/repo' slug")

    owner_repo = target.strip("/")
    return f"https://github.com/{owner_repo}.git"


def sync_repo(target: str, dry_run: bool = False, skip_confirmation: bool = False):
    """Filter `SUBDIR/` to repo root and push to the target remote."""

    remote_url = build_remote_url(target)
    print(f"Syncing target: {target}")
    print(f"Subdir: {SUBDIR}/")
    print(f"Target: {remote_url}")

    # Step 1: Filter
    print("\nFiltering repository...")
    try:
        # Make the configured subdirectory the repository root in the filtered repo
        filtered_path = git.filter_repo(Path.cwd(), [SUBDIR], root_subdir=SUBDIR)
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
    parser.add_argument("target", help="Target GitHub repository (owner/repo) or full remote URL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pushed")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    args = parser.parse_args()

    try:
        sync_repo(args.target, args.dry_run, args.yes)
    except KeyboardInterrupt:
        print("\nAborted")
        sys.exit(1)


if __name__ == "__main__":
    main()
