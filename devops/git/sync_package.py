#!/usr/bin/env python3
"""
Simple package sync workflow for filtering and pushing child repositories.

Usage:
    # Filter repository
    ./sync_package.py filter . mettagrid/ mettascope/
    
    # Inspect filtered result
    ./sync_package.py inspect /tmp/filtered-repo-xyz/filtered
    
    # Push to production
    ./sync_package.py push /tmp/filtered-repo-xyz/filtered git@github.com:org/public.git
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add parent directories to path to import GitRepo
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.src.metta.common.util.git import GitRepo


def create_filtered_package(source_repo_path: Path, target_paths: List[str]) -> Path:
    """One-liner to create filtered package from local repo.
    
    Returns path to filtered repository ready for pushing.
    """
    repo = GitRepo(source_repo_path)
    return repo.filter_repo(target_paths)


def inspect_filtered_repo(filtered_path: Path):
    """Show what's in the filtered repo."""
    try:
        repo = GitRepo(filtered_path)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
        
    files = repo.get_file_list()
    print(f"\n📊 Repository Statistics:")
    print(f"   Files: {len(files)}")
    print(f"   Commits: {repo.get_commit_count()}")
    print(f"   Location: {filtered_path}")
    
    # Group by top-level directory
    print(f"\n📁 Files by directory:")
    dirs = {}
    for f in files:
        parts = f.split('/')
        if len(parts) > 1:
            top_dir = parts[0] + '/'
        else:
            top_dir = '(root)'
        dirs[top_dir] = dirs.get(top_dir, 0) + 1
    
    for path, count in sorted(dirs.items()):
        print(f"   {path:<20} {count:>5} files")
    
    # Show sample files
    print(f"\n📄 Sample files:")
    for f in files[:10]:
        print(f"   {f}")
    if len(files) > 10:
        print(f"   ... and {len(files) - 10} more")


def push_to_production(filtered_path: Path, remote_url: str, dry_run: bool = False, skip_confirmation: bool = False):
    """Push filtered repository to production remote."""
    try:
        repo = GitRepo(filtered_path)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    print(f"\n📤 Pushing to: {remote_url}")
    print(f"   From: {filtered_path}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    
    if not dry_run and not skip_confirmation:
        response = input("\n⚠️  This will FORCE push. Continue? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Aborted")
            sys.exit(1)
    
    # Add remote
    repo.add_remote("production", remote_url)
    
    # Push
    push_cmd = ["push", "--force"]
    if dry_run:
        push_cmd.append("--dry-run")
    push_cmd.extend(["production", "HEAD:main"])
    
    print(f"\n{'🔔 DRY RUN: ' if dry_run else ''}Pushing...")
    
    try:
        result = repo.run_git(push_cmd)
        if result.stdout:
            print(result.stdout)
        print(f"\n✅ {'Dry run' if dry_run else 'Push'} completed successfully!")
    except RuntimeError as e:
        print(f"\n❌ Push failed: {e}")
        sys.exit(1)


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Simple package sync workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Filter repository locally
    %(prog)s filter . mettagrid/ mettascope/
    
    # Inspect the result
    %(prog)s inspect /tmp/filtered-repo-xyz/filtered
    
    # Push to production (dry run first)
    %(prog)s push /tmp/filtered-repo-xyz/filtered git@github.com:org/public.git --dry-run
    %(prog)s push /tmp/filtered-repo-xyz/filtered git@github.com:org/public.git
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Filter subcommand
    filter_cmd = subparsers.add_parser("filter", help="Filter repository locally")
    filter_cmd.add_argument("source", help="Source repository path")
    filter_cmd.add_argument("paths", nargs="+", help="Paths to include")
    
    # Inspect subcommand
    inspect_cmd = subparsers.add_parser("inspect", help="Inspect filtered repository")
    inspect_cmd.add_argument("filtered_repo", help="Filtered repository path")
    
    # Push subcommand
    push_cmd = subparsers.add_parser("push", help="Push to production")
    push_cmd.add_argument("filtered_repo", help="Filtered repository path")
    push_cmd.add_argument("remote_url", help="Production repository URL")
    push_cmd.add_argument("--dry-run", action="store_true", help="Show what would be pushed")
    push_cmd.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt (use with caution!)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "filter":
            source_path = Path(args.source).absolute()
            if not source_path.exists():
                print(f"❌ Source path not found: {source_path}")
                sys.exit(1)
                
            print(f"🔧 Filtering repository: {source_path}")
            print(f"📁 Paths to extract: {', '.join(args.paths)}")
            
            filtered_path = create_filtered_package(source_path, args.paths)
            
            print(f"\n✅ Success! Filtered repository at:")
            print(f"   {filtered_path}")
            print(f"\nNext steps:")
            print(f"   1. Inspect: {sys.argv[0]} inspect {filtered_path}")
            print(f"   2. Push:    {sys.argv[0]} push {filtered_path} <remote-url>")
            
        elif args.command == "inspect":
            filtered_path = Path(args.filtered_repo).absolute()
            if not filtered_path.exists():
                print(f"❌ Path not found: {filtered_path}")
                sys.exit(1)
            
            inspect_filtered_repo(filtered_path)
            
        elif args.command == "push":
            filtered_path = Path(args.filtered_repo).absolute()
            if not filtered_path.exists():
                print(f"❌ Path not found: {filtered_path}")
                sys.exit(1)
                
            push_to_production(filtered_path, args.remote_url, args.dry_run, args.yes)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()