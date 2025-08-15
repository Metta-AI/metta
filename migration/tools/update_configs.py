#!/usr/bin/env uv run
"""Update Hydra config _target_ fields to match new import paths."""

import yaml
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import json

class ConfigUpdater:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.changes_made: List[Tuple[Path, str, str]] = []
        self.backup_dir = Path("migration/backups/configs")
        
        # Define phase-specific mappings
        self.phase_mappings = {
            'phase2': {
                'common.src.metta.common': 'metta.common',
            },
            'phase3': {
                'agent.src.metta.agent': 'metta.agent',
                'mettagrid.src.metta.mettagrid': 'metta.mettagrid',
            },
            'phase4': {
                # Cogworks migrations (metta components only)
                'metta.rl': 'metta.cogworks.rl',
                'metta.sim': 'metta.cogworks.sim',
                'metta.eval': 'metta.cogworks.eval',
                'metta.sweep': 'metta.cogworks.sweep',
                'metta.map': 'metta.cogworks.mapgen',
                # Note: metta.agent stays as is (not moving to cogworks)
            },
            'phase5': {
                # Any final cleanup mappings
            }
        }
    
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the config file."""
        if not self.dry_run:
            backup_path = self.backup_dir / file_path.relative_to(Path.cwd())
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)
            return backup_path
        return file_path
    
    def update_config_file(self, yaml_file: Path, mappings: Dict[str, str]) -> bool:
        """Update _target_ fields in a single YAML file."""
        try:
            with open(yaml_file) as f:
                content = f.read()
                config = yaml.safe_load(content)
            
            if not config:
                return False
            
            original_config = yaml.dump(config, default_flow_style=False, sort_keys=False)
            modified = False
            
            def update_targets(obj, path=""):
                nonlocal modified
                if isinstance(obj, dict):
                    if '_target_' in obj:
                        original_target = obj['_target_']
                        for old_pattern, new_pattern in mappings.items():
                            if obj['_target_'].startswith(old_pattern):
                                obj['_target_'] = obj['_target_'].replace(old_pattern, new_pattern)
                                if obj['_target_'] != original_target:
                                    self.changes_made.append((yaml_file, original_target, obj['_target_']))
                                    modified = True
                                break
                    
                    for key, value in obj.items():
                        update_targets(value, f"{path}.{key}" if path else key)
                        
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        update_targets(item, f"{path}[{i}]")
            
            update_targets(config)
            
            if modified and not self.dry_run:
                self.backup_file(yaml_file)
                # Preserve YAML formatting as much as possible
                with open(yaml_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False, 
                             allow_unicode=True, width=120)
            
            return modified
            
        except yaml.YAMLError as e:
            print(f"YAML Error in {yaml_file}: {e}")
            return False
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")
            return False
    
    def update_configs_for_phase(self, config_dir: Path, phase: str) -> int:
        """Update all configs for a specific migration phase."""
        if phase not in self.phase_mappings:
            raise ValueError(f"Unknown phase: {phase}. Valid phases: {list(self.phase_mappings.keys())}")
        
        mappings = self.phase_mappings[phase]
        if not mappings:
            print(f"No mappings defined for {phase}")
            return 0
        
        yaml_files = list(config_dir.rglob("*.yaml")) + list(config_dir.rglob("*.yml"))
        modified_count = 0
        
        print(f"Processing {len(yaml_files)} config files for {phase}...")
        
        for yaml_file in yaml_files:
            if self.update_config_file(yaml_file, mappings):
                modified_count += 1
                if not self.dry_run:
                    print(f"  Modified: {yaml_file.relative_to(config_dir)}")
        
        return modified_count
    
    def verify_configs(self, config_dir: Path) -> bool:
        """Verify that all _target_ fields point to valid modules."""
        print("\nVerifying config targets...")
        invalid_targets = []
        
        for yaml_file in config_dir.rglob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                
                if not config:
                    continue
                
                def check_targets(obj, path=""):
                    if isinstance(obj, dict):
                        if '_target_' in obj:
                            target = obj['_target_']
                            # Skip checking if it's a built-in or third-party module
                            if target.startswith(('metta.', 'agent.', 'common.', 'mettagrid.')):
                                try:
                                    # Try to import the module
                                    module_path = '.'.join(target.split('.')[:-1])
                                    __import__(module_path)
                                except ImportError:
                                    invalid_targets.append((yaml_file, target))
                        
                        for key, value in obj.items():
                            check_targets(value, f"{path}.{key}" if path else key)
                            
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            check_targets(item, f"{path}[{i}]")
                
                check_targets(config)
                
            except Exception as e:
                print(f"Error checking {yaml_file}: {e}")
        
        if invalid_targets:
            print(f"\n⚠ Found {len(invalid_targets)} invalid targets:")
            for file_path, target in invalid_targets[:10]:
                print(f"  {file_path.name}: {target}")
            if len(invalid_targets) > 10:
                print(f"  ... and {len(invalid_targets) - 10} more")
            return False
        else:
            print("✓ All config targets appear valid")
            return True
    
    def report(self) -> None:
        """Generate update report."""
        print("\n" + "="*60)
        print("CONFIG UPDATE REPORT")
        print("="*60)
        
        if self.dry_run:
            print("\n⚠ DRY RUN MODE - No files were actually modified")
        else:
            print("\n✓ UPDATE COMPLETE - Config files have been updated")
        
        print(f"\nTotal changes: {len(self.changes_made)}")
        
        # Group changes by pattern
        pattern_counts = {}
        for _, old_target, new_target in self.changes_made:
            # Extract the base module change
            old_base = '.'.join(old_target.split('.')[:3])
            new_base = '.'.join(new_target.split('.')[:3])
            key = f"{old_base} → {new_base}"
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        
        if pattern_counts:
            print("\nChanges by pattern:")
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {count:3d} occurrences: {pattern}")
        
        # Show affected files
        affected_files = set(path for path, _, _ in self.changes_made)
        if affected_files:
            print(f"\nAffected config files: {len(affected_files)}")
            for file_path in sorted(affected_files)[:10]:
                try:
                    rel_path = file_path.relative_to(Path.cwd())
                except ValueError:
                    rel_path = file_path
                print(f"  - {rel_path}")
            
            if len(affected_files) > 10:
                print(f"  ... and {len(affected_files) - 10} more")
    
    def save_report(self, output_path: Path) -> None:
        """Save detailed report to JSON."""
        report = {
            'dry_run': self.dry_run,
            'total_changes': len(self.changes_made),
            'changes': [
                {
                    'file': str(path),
                    'old_target': old,
                    'new_target': new
                }
                for path, old, new in self.changes_made
            ],
            'affected_files': sorted(list(set(str(path) for path, _, _ in self.changes_made)))
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\nDetailed report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Update Hydra config targets for migration phases'
    )
    parser.add_argument('phase', 
                      choices=['phase2', 'phase3', 'phase4', 'phase5', 'all'],
                      help='Migration phase to apply')
    parser.add_argument('--config-dir', default='configs',
                      help='Directory containing config files (default: configs)')
    parser.add_argument('--dry-run', action='store_true', default=True,
                      help='Show what would be changed without modifying files')
    parser.add_argument('--apply', action='store_true',
                      help='Actually apply changes (disables dry-run)')
    parser.add_argument('--verify', action='store_true',
                      help='Verify that config targets are valid after update')
    parser.add_argument('--report', default='migration/reports/config-update.json',
                      help='Path for detailed report')
    
    args = parser.parse_args()
    
    # If --apply is set, disable dry run
    if args.apply:
        args.dry_run = False
    
    updater = ConfigUpdater(dry_run=args.dry_run)
    config_dir = Path(args.config_dir)
    
    if not config_dir.exists():
        print(f"Error: Config directory '{config_dir}' does not exist")
        return 1
    
    # Apply updates
    if args.phase == 'all':
        total_modified = 0
        for phase in ['phase2', 'phase3', 'phase4', 'phase5']:
            print(f"\n--- Processing {phase} ---")
            modified = updater.update_configs_for_phase(config_dir, phase)
            total_modified += modified
            print(f"Modified {modified} files in {phase}")
    else:
        modified = updater.update_configs_for_phase(config_dir, args.phase)
        print(f"Modified {modified} files")
    
    # Generate report
    updater.report()
    updater.save_report(Path(args.report))
    
    # Verify if requested
    if args.verify and not args.dry_run:
        success = updater.verify_configs(config_dir)
        if not success:
            print("\n⚠ WARNING: Some config targets may be invalid")
            return 1
    
    if args.dry_run:
        print("\n💡 To actually update configs, run with --apply flag")
    
    return 0

if __name__ == "__main__":
    exit(main())