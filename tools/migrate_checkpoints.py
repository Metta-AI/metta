#!/usr/bin/env python3
"""
Migrate Metta checkpoints from old format to new format.

This tool converts checkpoints from the old pickle-based format (entire PolicyRecord)
to the new standard PyTorch format (separate .pt and .json files).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from metta.agent.policy_store import PolicyStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_single_checkpoint(policy_store: PolicyStore, checkpoint_path: str, replace: bool = False) -> bool:
    """Migrate a single checkpoint file.
    
    Args:
        policy_store: PolicyStore instance
        checkpoint_path: Path to checkpoint to migrate
        replace: If True, replace original file. If False, create .new.pt file
        
    Returns:
        True if migration successful, False otherwise
    """
    try:
        # Check if already in new format
        base_path = checkpoint_path[:-3] if checkpoint_path.endswith('.pt') else checkpoint_path
        metadata_path = base_path + '.json'
        
        if os.path.exists(metadata_path):
            logger.info(f"Skipping {checkpoint_path} - already in new format")
            return True
        
        # Migrate the checkpoint
        if replace:
            # Create temporary new path
            temp_path = base_path + '.tmp.pt'
            policy_store.migrate_checkpoint(checkpoint_path, temp_path)
            
            # Move files to final locations
            os.rename(temp_path, checkpoint_path)
            os.rename(base_path + '.tmp.json', base_path + '.json')
            logger.info(f"Migrated {checkpoint_path} in place")
        else:
            # Create .new.pt file
            new_path = policy_store.migrate_checkpoint(checkpoint_path)
            logger.info(f"Created migrated checkpoint at {new_path}")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to migrate {checkpoint_path}: {e}")
        return False


def migrate_directory(policy_store: PolicyStore, directory: str, replace: bool = False, recursive: bool = False) -> tuple[int, int]:
    """Migrate all checkpoints in a directory.
    
    Args:
        policy_store: PolicyStore instance
        directory: Directory to search for checkpoints
        replace: If True, replace original files
        recursive: If True, search subdirectories
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    success_count = 0
    failed_count = 0
    
    pattern = '**/*.pt' if recursive else '*.pt'
    checkpoint_files = list(Path(directory).glob(pattern))
    
    logger.info(f"Found {len(checkpoint_files)} checkpoint files")
    
    for checkpoint_path in checkpoint_files:
        if migrate_single_checkpoint(policy_store, str(checkpoint_path), replace):
            success_count += 1
        else:
            failed_count += 1
            
    return success_count, failed_count


def main():
    parser = argparse.ArgumentParser(description='Migrate Metta checkpoints to new format')
    parser.add_argument('paths', nargs='+', help='Checkpoint files or directories to migrate')
    parser.add_argument('--replace', action='store_true', help='Replace original files (default: create .new.pt files)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursively search directories')
    parser.add_argument('--device', default='cpu', help='Device to use for loading (default: cpu)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated without doing it')
    
    args = parser.parse_args()
    
    # Create minimal config for PolicyStore
    cfg = OmegaConf.create({
        'device': args.device,
        'data_dir': '/tmp/metta_migration',
        'policy_cache_size': 1,  # Minimal cache
    })
    
    policy_store = PolicyStore(cfg, wandb_run=None)
    
    total_success = 0
    total_failed = 0
    
    for path in args.paths:
        if os.path.isfile(path):
            if path.endswith('.pt'):
                if args.dry_run:
                    base_path = path[:-3] if path.endswith('.pt') else path
                    metadata_path = base_path + '.json'
                    if os.path.exists(metadata_path):
                        logger.info(f"Would skip {path} - already in new format")
                    else:
                        logger.info(f"Would migrate {path}")
                else:
                    if migrate_single_checkpoint(policy_store, path, args.replace):
                        total_success += 1
                    else:
                        total_failed += 1
            else:
                logger.warning(f"Skipping {path} - not a .pt file")
                
        elif os.path.isdir(path):
            if args.dry_run:
                pattern = '**/*.pt' if args.recursive else '*.pt'
                checkpoint_files = list(Path(path).glob(pattern))
                for cp in checkpoint_files:
                    base_path = str(cp)[:-3]
                    metadata_path = base_path + '.json'
                    if os.path.exists(metadata_path):
                        logger.info(f"Would skip {cp} - already in new format")
                    else:
                        logger.info(f"Would migrate {cp}")
            else:
                success, failed = migrate_directory(policy_store, path, args.replace, args.recursive)
                total_success += success
                total_failed += failed
        else:
            logger.error(f"Path does not exist: {path}")
            total_failed += 1
    
    if not args.dry_run:
        logger.info(f"\nMigration complete: {total_success} successful, {total_failed} failed")
    
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())