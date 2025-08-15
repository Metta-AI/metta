#!/usr/bin/env uv run
"""Tool to rewrite import paths during the migration process."""

import ast
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import argparse
from dataclasses import dataclass
from enum import Enum

class MigrationPhase(Enum):
    """Migration phases corresponding to the reorganization plan."""
    PHASE_2_COMMON = "phase2_common"
    PHASE_3_METTAGRID = "phase3_mettagrid"
    PHASE_4_COGWORKS = "phase4_cogworks"
    PHASE_5_CLEANUP = "phase5_cleanup"

@dataclass
class ImportMapping:
    """Represents a single import path transformation."""
    pattern: str
    replacement: str
    phase: MigrationPhase
    description: str

class ImportRewriter:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.mappings: List[ImportMapping] = []
        self.changes_made: List[Tuple[Path, str, str]] = []
        self.backup_dir = Path("migration/backups")
        self._initialize_mappings()
        
    def _initialize_mappings(self):
        """Initialize import mappings for each migration phase."""
        # Phase 2: Common and Backend-Shared
        self.mappings.extend([
            ImportMapping(
                r'from common\.src\.metta\.common',
                'from metta.common',
                MigrationPhase.PHASE_2_COMMON,
                "Flatten common package structure"
            ),
            ImportMapping(
                r'import common\.src\.metta\.common',
                'import metta.common',
                MigrationPhase.PHASE_2_COMMON,
                "Flatten common package structure"
            ),
            ImportMapping(
                r'from app_backend\.src\.metta\.app_backend\.clients\.stats_client',
                'from metta.backend_shared.stats_client',
                MigrationPhase.PHASE_2_COMMON,
                "Move stats_client to backend_shared"
            ),
        ])
        
        # Phase 3: MettagGrid
        self.mappings.extend([
            ImportMapping(
                r'from mettagrid\.src\.metta\.mettagrid',
                'from metta.mettagrid',
                MigrationPhase.PHASE_3_METTAGRID,
                "Flatten mettagrid structure"
            ),
            ImportMapping(
                r'import mettagrid\.src\.metta\.mettagrid',
                'import metta.mettagrid',
                MigrationPhase.PHASE_3_METTAGRID,
                "Flatten mettagrid structure"
            ),
        ])
        
        # Phase 4: Cogworks (metta only) and Agent flattening
        self.mappings.extend([
            ImportMapping(
                r'from metta\.rl',
                'from metta.cogworks.rl',
                MigrationPhase.PHASE_4_COGWORKS,
                "Move RL to cogworks"
            ),
            ImportMapping(
                r'from agent\.src\.metta\.agent',
                'from metta.agent',
                MigrationPhase.PHASE_4_COGWORKS,
                "Flatten agent package structure"
            ),
            ImportMapping(
                r'from metta\.eval',
                'from metta.cogworks.eval',
                MigrationPhase.PHASE_4_COGWORKS,
                "Move eval to cogworks"
            ),
            ImportMapping(
                r'from metta\.sim',
                'from metta.cogworks.sim',
                MigrationPhase.PHASE_4_COGWORKS,
                "Move sim to cogworks"
            ),
            ImportMapping(
                r'from metta\.sweep',
                'from metta.cogworks.sweep',
                MigrationPhase.PHASE_4_COGWORKS,
                "Move sweep to cogworks"
            ),
            ImportMapping(
                r'from metta\.map',
                'from metta.cogworks.mapgen',
                MigrationPhase.PHASE_4_COGWORKS,
                "Rename map to mapgen in cogworks"
            ),
            ImportMapping(
                r'import metta\.rl',
                'import metta.cogworks.rl',
                MigrationPhase.PHASE_4_COGWORKS,
                "Move RL to cogworks"
            ),
            ImportMapping(
                r'import agent\.src\.metta\.agent',
                'import metta.agent',
                MigrationPhase.PHASE_4_COGWORKS,
                "Flatten agent package structure"
            ),
        ])
    
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file before modification."""
        backup_path = self.backup_dir / file_path.relative_to(Path.cwd())
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def rewrite_file(self, file_path: Path, phase: MigrationPhase) -> bool:
        """Rewrite imports in a single file for the specified phase."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            original_content = content
            relevant_mappings = [m for m in self.mappings if m.phase == phase]
            
            for mapping in relevant_mappings:
                new_content = re.sub(mapping.pattern, mapping.replacement, content)
                if new_content != content:
                    self.changes_made.append((file_path, mapping.pattern, mapping.replacement))
                    content = new_content
            
            if content != original_content:
                if not self.dry_run:
                    # Backup original file
                    self.backup_file(file_path)
                    # Write updated content
                    with open(file_path, 'w') as f:
                        f.write(content)
                return True
            return False
            
        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            return False
    
    def rewrite_directory(self, directory: Path, phase: MigrationPhase,
                         include_tests: bool = True, include_configs: bool = True) -> int:
        """Rewrite all Python files in a directory for the specified phase."""
        patterns = ['*.py']
        if include_configs:
            patterns.extend(['*.yaml', '*.yml'])
            
        files_to_process = []
        for pattern in patterns:
            files_to_process.extend(directory.rglob(pattern))
        
        # Filter out unwanted directories
        files_to_process = [
            f for f in files_to_process
            if '.venv' not in str(f)
            and '__pycache__' not in str(f)
            and '.git/' not in str(f)
            and 'migration/' not in str(f)
            and 'wandb/' not in str(f)
            and (include_tests or 'tests/' not in str(f))
        ]
        
        modified_count = 0
        print(f"Processing {len(files_to_process)} files for {phase.value}...")
        
        for file_path in files_to_process:
            if self.rewrite_file(file_path, phase):
                modified_count += 1
                if not self.dry_run:
                    print(f"  Modified: {file_path}")
        
        return modified_count
    
    def rewrite_config_targets(self, config_path: Path, phase: MigrationPhase) -> bool:
        """Special handling for Hydra config files with _target_ fields."""
        if not config_path.suffix in ['.yaml', '.yml']:
            return False
            
        try:
            import yaml
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            if not config:
                return False
                
            original_config = yaml.dump(config)
            relevant_mappings = [m for m in self.mappings if m.phase == phase]
            
            def update_targets(obj):
                """Recursively update _target_ fields."""
                if isinstance(obj, dict):
                    if '_target_' in obj:
                        for mapping in relevant_mappings:
                            obj['_target_'] = re.sub(
                                mapping.pattern.replace('from ', '').replace('import ', ''),
                                mapping.replacement.replace('from ', '').replace('import ', ''),
                                obj['_target_']
                            )
                    for value in obj.values():
                        update_targets(value)
                elif isinstance(obj, list):
                    for item in obj:
                        update_targets(item)
            
            update_targets(config)
            
            new_config = yaml.dump(config)
            if new_config != original_config:
                if not self.dry_run:
                    self.backup_file(config_path)
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                return True
                
        except Exception as e:
            print(f"Error processing config {config_path}: {e}")
            
        return False
    
    def verify_rewrites(self, directory: Path) -> bool:
        """Verify that rewrites don't break imports."""
        print("\nVerifying rewrites...")
        
        # Try to import key modules
        test_imports = [
            "metta.common" if any(m.phase == MigrationPhase.PHASE_2_COMMON for m in self.mappings) else None,
            "metta.mettagrid" if any(m.phase == MigrationPhase.PHASE_3_METTAGRID for m in self.mappings) else None,
            "metta.cogworks" if any(m.phase == MigrationPhase.PHASE_4_COGWORKS for m in self.mappings) else None,
        ]
        
        failures = []
        for module in test_imports:
            if module:
                try:
                    __import__(module)
                    print(f"  ✓ {module} imports successfully")
                except ImportError as e:
                    failures.append((module, str(e)))
                    print(f"  ✗ {module} failed: {e}")
        
        return len(failures) == 0
    
    def generate_compatibility_layer(self, phase: MigrationPhase) -> None:
        """Generate compatibility imports for gradual migration."""
        compatibility_code = {
            MigrationPhase.PHASE_2_COMMON: '''
# Temporary compatibility layer for common package
import warnings
import sys

# Redirect old imports to new locations
class CompatibilityImporter:
    def find_module(self, fullname, path=None):
        if fullname.startswith('common.src.metta.common'):
            return self
        return None
    
    def load_module(self, fullname):
        warnings.warn(
            f"Import '{fullname}' is deprecated. Use 'metta.common' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Redirect to new module
        new_name = fullname.replace('common.src.metta.common', 'metta.common')
        return __import__(new_name)

sys.meta_path.insert(0, CompatibilityImporter())
''',
            MigrationPhase.PHASE_4_COGWORKS: '''
# Temporary compatibility layer for cogworks migration
import warnings
import sys

class CogworksCompatibilityImporter:
    MAPPINGS = {
        'metta.rl': 'metta.cogworks.rl',
        'metta.agent': 'metta.cogworks.agent',
        'metta.eval': 'metta.cogworks.eval',
        'metta.sim': 'metta.cogworks.sim',
        'metta.sweep': 'metta.cogworks.sweep',
        'metta.map': 'metta.cogworks.mapgen',
    }
    
    def find_module(self, fullname, path=None):
        for old, new in self.MAPPINGS.items():
            if fullname.startswith(old):
                return self
        return None
    
    def load_module(self, fullname):
        for old, new in self.MAPPINGS.items():
            if fullname.startswith(old):
                warnings.warn(
                    f"Import '{fullname}' is deprecated. Use '{fullname.replace(old, new)}' instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                new_name = fullname.replace(old, new)
                return __import__(new_name)

sys.meta_path.insert(0, CogworksCompatibilityImporter())
'''
        }
        
        if phase in compatibility_code:
            compat_file = Path(f"migration/compatibility/{phase.value}_compat.py")
            compat_file.parent.mkdir(parents=True, exist_ok=True)
            compat_file.write_text(compatibility_code[phase])
            print(f"Generated compatibility layer: {compat_file}")
    
    def report(self) -> None:
        """Generate report of changes made."""
        print("\n" + "="*60)
        print("IMPORT REWRITE REPORT")
        print("="*60)
        
        if self.dry_run:
            print("\n⚠ DRY RUN MODE - No files were actually modified")
        
        print(f"\nTotal changes: {len(self.changes_made)}")
        
        # Group changes by pattern
        pattern_counts = {}
        for _, pattern, replacement in self.changes_made:
            key = f"{pattern} → {replacement}"
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        
        print("\nChanges by pattern:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {count:3d} occurrences: {pattern}")
        
        # Show sample of affected files
        if self.changes_made:
            print("\nSample of affected files:")
            for path, _, _ in self.changes_made[:10]:
                print(f"  - {path}")
            
            if len(self.changes_made) > 10:
                print(f"  ... and {len(self.changes_made) - 10} more")

def main():
    parser = argparse.ArgumentParser(description='Rewrite import paths for migration')
    parser.add_argument('--phase', type=str, required=True,
                      choices=[p.value for p in MigrationPhase],
                      help='Migration phase to apply')
    parser.add_argument('--path', default='.', help='Path to process')
    parser.add_argument('--dry-run', action='store_true', default=True,
                      help='Perform dry run without modifying files (default: True)')
    parser.add_argument('--apply', action='store_true',
                      help='Actually apply changes (disables dry-run)')
    parser.add_argument('--include-tests', action='store_true', default=True,
                      help='Include test files in rewriting')
    parser.add_argument('--include-configs', action='store_true', default=True,
                      help='Include config files in rewriting')
    parser.add_argument('--generate-compat', action='store_true',
                      help='Generate compatibility layer for the phase')
    
    args = parser.parse_args()
    
    # If --apply is set, disable dry run
    if args.apply:
        args.dry_run = False
    
    rewriter = ImportRewriter(dry_run=args.dry_run)
    phase = MigrationPhase(args.phase)
    
    # Generate compatibility layer if requested
    if args.generate_compat:
        rewriter.generate_compatibility_layer(phase)
    
    # Perform rewrites
    modified = rewriter.rewrite_directory(
        Path(args.path),
        phase,
        include_tests=args.include_tests,
        include_configs=args.include_configs
    )
    
    # Generate report
    rewriter.report()
    
    # Verify if not dry run
    if not args.dry_run:
        success = rewriter.verify_rewrites(Path(args.path))
        if not success:
            print("\n⚠ WARNING: Some imports may be broken after rewriting")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())