#!/usr/bin/env python3
"""
Extract specific directories from a monorepo and push them to a separate repository 
with preserved git history.

Prerequisites:
    brew install git-filter-repo
    
Run with:
    uv run --with PyYAML ./sync_package.py [command]

Usage with config file:
    # Run all steps at once
    ./sync_package.py sync filter_repo_test
    
    # Or run individual steps  
    ./sync_package.py filter --config filter_repo_test
    ./sync_package.py inspect /tmp/filtered-repo-xyz/filtered
    ./sync_package.py push /tmp/filtered-repo-xyz/filtered --config filter_repo_test

Usage without config (original):
    ./sync_package.py filter . mettagrid/ mettascope/
    ./sync_package.py inspect /tmp/filtered-repo-xyz/filtered
    ./sync_package.py push /tmp/filtered-repo-xyz/filtered git@github.com:org/repo.git

Config files are stored in child_repos/ directory as YAML files.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directories to path to import git utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.src.metta.common.util.git_filter import (
    filter_repo,
    get_file_list,
    get_commit_count,
    add_remote,
    run_git_in_repo
)
from common.src.metta.common.util.git import run_git


def load_config(config_name: str) -> Dict[str, any]:
    """Load configuration from YAML file."""
    config_dir = Path(__file__).parent / "child_repos"
    
    # Try with .yaml extension first
    config_path = config_dir / f"{config_name}.yaml"
    if not config_path.exists():
        # Try without extension
        config_path = config_dir / config_name
        if not config_path.exists():
            # List available configs
            available = [f.stem for f in config_dir.glob("*.yaml")] if config_dir.exists() else []
            if available:
                print(f"❌ Config '{config_name}' not found. Available configs:")
                for cfg in available:
                    print(f"   - {cfg}")
            else:
                print(f"❌ No config files found in {config_dir}")
            sys.exit(1)
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        if 'paths' not in config:
            print(f"❌ Config missing required field 'paths'")
            sys.exit(1)
        if 'remote' not in config:
            print(f"❌ Config missing required field 'remote'")
            sys.exit(1)
            
        return config
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config file: {e}")
        sys.exit(1)


def create_filtered_package(source_repo_path: Path, target_paths: List[str]) -> Path:
    """One-liner to create filtered package from local repo.
    
    Returns path to filtered repository ready for pushing.
    """
    return filter_repo(source_repo_path, target_paths)


def inspect_filtered_repo(filtered_path: Path):
    """Show what's in the filtered repo."""
    if not (filtered_path / '.git').exists():
        print(f"❌ Not a git repository: {filtered_path}")
        sys.exit(1)
        
    files = get_file_list(filtered_path)
    print(f"\n📊 Repository Statistics:")
    print(f"   Files: {len(files)}")
    print(f"   Commits: {get_commit_count(filtered_path)}")
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


def get_main_repo_remote() -> str:
    """Get the remote URL of the current repository's origin."""
    try:
        result = run_git_in_repo(Path.cwd(), "remote", "get-url", "origin")
        return result.stdout.strip()
    except:
        return ""


def push_to_production(filtered_path: Path, remote_url: str, dry_run: bool = False, skip_confirmation: bool = False):
    """Push filtered repository to production remote."""
    if not (filtered_path / '.git').exists():
        print(f"❌ Not a git repository: {filtered_path}")
        sys.exit(1)
    
    # Safety check: prevent pushing to main metta repository
    dangerous_urls = [
        "git@github.com:Metta-AI/metta.git",
        "https://github.com/Metta-AI/metta.git",
        "git@github.com:Metta-AI/metta",
        "https://github.com/Metta-AI/metta"
    ]
    
    # Also check current repo's origin
    main_repo_url = get_main_repo_remote()
    if main_repo_url:
        dangerous_urls.append(main_repo_url)
    
    # Normalize the URL for comparison
    normalized_remote = remote_url.rstrip('/').rstrip('.git')
    
    for dangerous in dangerous_urls:
        if dangerous and normalized_remote == dangerous.rstrip('/').rstrip('.git'):
            print("\n" + "=" * 80)
            print("🚨🚨🚨 NO WAY! 🚨🚨🚨")
            print("=" * 80)
            print(f"\nYou're trying to push to the MAIN METTA REPOSITORY!")
            print(f"Target: {remote_url}")
            print(f"\nThis would DESTROY the main repository!")
            print("\nIf you really meant to do this (you probably didn't), use a different tool.")
            print("=" * 80)
            sys.exit(1)
    
    print(f"\n📤 Target repository: {remote_url}")
    print(f"   From: {filtered_path}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    
    if not dry_run and not skip_confirmation:
        print("\n⚠️  This will FORCE PUSH and completely replace the target repository!")
        print(f"\nPlease type the target repository URL to confirm:")
        print(f"Expected: {remote_url}")
        typed_url = input("Your input: ").strip()
        
        if typed_url != remote_url:
            print("\n❌ URLs don't match. Aborted for safety.")
            sys.exit(1)
        
        response = input("\nFinal confirmation - proceed with force push? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Aborted")
            sys.exit(1)
    
    # Add remote
    add_remote(filtered_path, "production", remote_url)
    
    # Push
    push_cmd = ["push", "--force"]
    if dry_run:
        push_cmd.append("--dry-run")
    push_cmd.extend(["production", "HEAD:main"])
    
    print(f"\n{'🔔 DRY RUN: ' if dry_run else ''}Pushing...")
    
    try:
        result = run_git_in_repo(filtered_path, *push_cmd)
        if result.stdout:
            print(result.stdout)
        print(f"\n✅ {'Dry run' if dry_run else 'Push'} completed successfully!")
    except RuntimeError as e:
        print(f"\n❌ Push failed: {e}")
        sys.exit(1)


def sync_all(config_name: str, source_path: Path, dry_run: bool = False):
    """Run all steps: filter, inspect, and push."""
    config = load_config(config_name)
    
    print(f"🔄 Syncing {config_name}")
    print(f"   Paths: {', '.join(config['paths'])}")
    print(f"   Remote: {config['remote']}")
    
    # Step 1: Filter
    print(f"\n📁 Step 1/3: Filtering repository...")
    filtered_path = create_filtered_package(source_path, config['paths'])
    
    # Step 2: Inspect
    print(f"\n🔍 Step 2/3: Inspecting filtered repository...")
    inspect_filtered_repo(filtered_path)
    
    # Step 3: Push
    print(f"\n🚀 Step 3/3: Pushing to remote...")
    push_to_production(filtered_path, config['remote'], dry_run=dry_run, skip_confirmation=dry_run)
    
    print(f"\n✅ Sync complete! Filtered repo at: {filtered_path}")
    

def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Sync child repositories from monorepo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using config files (recommended):
    %(prog)s sync filter_repo_test                    # Run all steps
    %(prog)s sync filter_repo_test --dry-run          # Dry run
    
    # Individual steps with config:
    %(prog)s filter --config filter_repo_test
    %(prog)s push /tmp/filtered-repo-xyz/filtered --config filter_repo_test
    
    # Original usage (without config):
    %(prog)s filter . mettagrid/ mettascope/
    %(prog)s inspect /tmp/filtered-repo-xyz/filtered
    %(prog)s push /tmp/filtered-repo-xyz/filtered git@github.com:org/repo.git
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Sync subcommand (all steps at once)
    sync_cmd = subparsers.add_parser("sync", help="Run all steps: filter, inspect, and push")
    sync_cmd.add_argument("config", help="Config name (e.g., filter_repo_test)")
    sync_cmd.add_argument("--source", default=".", help="Source repository path (default: current dir)")
    sync_cmd.add_argument("--dry-run", action="store_true", help="Show what would be pushed without pushing")
    
    # Filter subcommand
    filter_cmd = subparsers.add_parser("filter", help="Filter repository locally")
    filter_cmd.add_argument("source", nargs="?", default=".", help="Source repository path")
    filter_cmd.add_argument("paths", nargs="*", help="Paths to include (if not using --config)")
    filter_cmd.add_argument("--config", help="Config name to load paths from")
    
    # Inspect subcommand
    inspect_cmd = subparsers.add_parser("inspect", help="Inspect filtered repository")
    inspect_cmd.add_argument("filtered_repo", help="Filtered repository path")
    
    # Push subcommand
    push_cmd = subparsers.add_parser("push", help="Push to production")
    push_cmd.add_argument("filtered_repo", help="Filtered repository path")
    push_cmd.add_argument("remote_url", nargs="?", help="Production repository URL (if not using --config)")
    push_cmd.add_argument("--config", help="Config name to load remote URL from")
    push_cmd.add_argument("--dry-run", action="store_true", help="Show what would be pushed")
    push_cmd.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "sync":
            source_path = Path(args.source).absolute()
            if not source_path.exists():
                print(f"❌ Source path not found: {source_path}")
                sys.exit(1)
            sync_all(args.config, source_path, args.dry_run)
            
        elif args.command == "filter":
            source_path = Path(args.source).absolute()
            if not source_path.exists():
                print(f"❌ Source path not found: {source_path}")
                sys.exit(1)
            
            # Get paths from config or command line
            if args.config:
                config = load_config(args.config)
                paths = config['paths']
                print(f"📋 Using config: {args.config}")
            elif args.paths:
                paths = args.paths
            else:
                print("❌ Either specify paths or use --config")
                sys.exit(1)
                
            print(f"🔧 Filtering repository: {source_path}")
            print(f"📁 Paths to extract: {', '.join(paths)}")
            
            filtered_path = create_filtered_package(source_path, paths)
            
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
            
            # Get remote URL from config or command line
            if args.config:
                config = load_config(args.config)
                remote_url = config['remote']
                print(f"📋 Using config: {args.config}")
            elif args.remote_url:
                remote_url = args.remote_url
            else:
                print("❌ Either specify remote URL or use --config")
                sys.exit(1)
                
            push_to_production(filtered_path, remote_url, args.dry_run, args.yes)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()