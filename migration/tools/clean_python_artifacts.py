#!/usr/bin/env uv run
"""Clean Python artifacts that can cause issues during migration."""

import shutil
import sys
from pathlib import Path
import argparse

class PythonCleaner:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.removed_items = []
        
    def clean_pycache(self, directory: Path) -> int:
        """Remove all __pycache__ directories."""
        pycache_dirs = list(directory.rglob("__pycache__"))
        
        for cache_dir in pycache_dirs:
            if not self.dry_run:
                shutil.rmtree(cache_dir)
            self.removed_items.append(('__pycache__', cache_dir))
            
        return len(pycache_dirs)
    
    def clean_pyc_files(self, directory: Path) -> int:
        """Remove all .pyc files."""
        pyc_files = list(directory.rglob("*.pyc"))
        
        for pyc_file in pyc_files:
            if not self.dry_run:
                pyc_file.unlink()
            self.removed_items.append(('.pyc file', pyc_file))
            
        return len(pyc_files)
    
    def clean_pyo_files(self, directory: Path) -> int:
        """Remove all .pyo files."""
        pyo_files = list(directory.rglob("*.pyo"))
        
        for pyo_file in pyo_files:
            if not self.dry_run:
                pyo_file.unlink()
            self.removed_items.append(('.pyo file', pyo_file))
            
        return len(pyo_files)
    
    def clean_egg_info(self, directory: Path) -> int:
        """Remove .egg-info directories."""
        egg_dirs = list(directory.rglob("*.egg-info"))
        
        for egg_dir in egg_dirs:
            if not self.dry_run:
                shutil.rmtree(egg_dir)
            self.removed_items.append(('.egg-info', egg_dir))
            
        return len(egg_dirs)
    
    def clean_dist_build(self, directory: Path) -> int:
        """Remove dist/ and build/ directories."""
        count = 0
        
        for pattern in ["dist", "build"]:
            for found_dir in directory.rglob(pattern):
                if found_dir.is_dir() and found_dir.name == pattern:
                    # Only remove if it looks like a Python build directory
                    parent_has_setup = (found_dir.parent / "setup.py").exists() or \
                                     (found_dir.parent / "pyproject.toml").exists()
                    if parent_has_setup:
                        if not self.dry_run:
                            shutil.rmtree(found_dir)
                        self.removed_items.append((f'{pattern}/', found_dir))
                        count += 1
                        
        return count
    
    def clean_pytest_cache(self, directory: Path) -> int:
        """Remove .pytest_cache directories."""
        pytest_dirs = list(directory.rglob(".pytest_cache"))
        
        for cache_dir in pytest_dirs:
            if not self.dry_run:
                shutil.rmtree(cache_dir)
            self.removed_items.append(('.pytest_cache', cache_dir))
            
        return len(pytest_dirs)
    
    def find_orphaned_inits(self, directory: Path) -> list:
        """Find __init__.py files that might be orphaned after migration."""
        init_files = []
        
        for init_file in directory.rglob("__init__.py"):
            # Check if this is in a src/ directory that will be removed
            if 'src/metta' in str(init_file):
                init_files.append(init_file)
                
        return init_files
    
    def clean_all(self, directory: Path) -> dict:
        """Clean all Python artifacts."""
        results = {
            '__pycache__': self.clean_pycache(directory),
            '.pyc files': self.clean_pyc_files(directory),
            '.pyo files': self.clean_pyo_files(directory),
            '.egg-info': self.clean_egg_info(directory),
            'dist/build': self.clean_dist_build(directory),
            '.pytest_cache': self.clean_pytest_cache(directory),
        }
        
        # Also find potential issues
        orphaned_inits = self.find_orphaned_inits(directory)
        if orphaned_inits:
            results['orphaned_inits'] = len(orphaned_inits)
            
        return results
    
    def report(self) -> None:
        """Generate cleaning report."""
        print("\n" + "="*60)
        print("PYTHON ARTIFACTS CLEANING REPORT")
        print("="*60)
        
        if self.dry_run:
            print("\n⚠ DRY RUN MODE - No files were actually removed")
        else:
            print("\n✓ CLEANING COMPLETE - Files have been removed")
        
        # Group by type
        type_counts = {}
        for artifact_type, path in self.removed_items:
            type_counts[artifact_type] = type_counts.get(artifact_type, 0) + 1
        
        print(f"\nTotal items to remove: {len(self.removed_items)}")
        print("\nBy type:")
        for artifact_type, count in sorted(type_counts.items()):
            print(f"  {artifact_type}: {count}")
        
        # Show sample of items
        if self.removed_items:
            print("\nSample of items (first 10):")
            for artifact_type, path in self.removed_items[:10]:
                try:
                    rel_path = path.relative_to(Path.cwd())
                except ValueError:
                    rel_path = path
                print(f"  [{artifact_type}] {rel_path}")
            
            if len(self.removed_items) > 10:
                print(f"  ... and {len(self.removed_items) - 10} more")

def main():
    parser = argparse.ArgumentParser(
        description='Clean Python artifacts before migration'
    )
    parser.add_argument('--path', default='.', help='Path to clean')
    parser.add_argument('--dry-run', action='store_true', default=True,
                      help='Show what would be removed without removing')
    parser.add_argument('--clean', action='store_true',
                      help='Actually remove files (disables dry-run)')
    parser.add_argument('--type', choices=['all', 'pycache', 'pyc', 'egg', 'dist', 'pytest'],
                      default='all', help='Type of artifacts to clean')
    
    args = parser.parse_args()
    
    # If --clean is set, disable dry run
    if args.clean:
        args.dry_run = False
    
    cleaner = PythonCleaner(dry_run=args.dry_run)
    directory = Path(args.path)
    
    if args.type == 'all':
        results = cleaner.clean_all(directory)
        print("\nCleaning summary:")
        for artifact_type, count in results.items():
            if count > 0:
                print(f"  {artifact_type}: {count} items")
    elif args.type == 'pycache':
        count = cleaner.clean_pycache(directory)
        print(f"Found {count} __pycache__ directories")
    elif args.type == 'pyc':
        count = cleaner.clean_pyc_files(directory)
        print(f"Found {count} .pyc files")
    elif args.type == 'egg':
        count = cleaner.clean_egg_info(directory)
        print(f"Found {count} .egg-info directories")
    elif args.type == 'dist':
        count = cleaner.clean_dist_build(directory)
        print(f"Found {count} dist/build directories")
    elif args.type == 'pytest':
        count = cleaner.clean_pytest_cache(directory)
        print(f"Found {count} .pytest_cache directories")
    
    cleaner.report()
    
    if args.dry_run:
        print("\n💡 To actually remove these files, run with --clean flag")

if __name__ == "__main__":
    main()