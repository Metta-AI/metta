#!/usr/bin/env python3
"""
Utility to migrate old Metta checkpoints to the new standard PyTorch format.

Old format: Single file containing entire PolicyRecord object (brittle pickle)
New format: Two files - .pt (state_dict only) + .json (metadata sidecar)

Usage:
    python -m metta.agent.migrate_checkpoints path/to/checkpoint.pt
    python -m metta.agent.migrate_checkpoints path/to/checkpoint/dir --recursive
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore

logger = logging.getLogger(__name__)


def migrate_checkpoint(checkpoint_path: str, output_path: Optional[str] = None, 
                      backup: bool = True, dry_run: bool = False) -> bool:
    """
    Migrate a single checkpoint from old format to new format.
    
    Args:
        checkpoint_path: Path to the old format checkpoint
        output_path: Optional path for the migrated checkpoint (defaults to same location)
        backup: Whether to create a backup of the original file
        dry_run: If True, only show what would be done without making changes
        
    Returns:
        True if migration was successful, False otherwise
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    # Check if already in new format
    base_path = checkpoint_path[:-3] if checkpoint_path.endswith('.pt') else checkpoint_path
    metadata_path = base_path + '.json'
    
    if os.path.exists(metadata_path):
        logger.info(f"Checkpoint already in new format: {checkpoint_path}")
        return True
    
    logger.info(f"Migrating checkpoint: {checkpoint_path}")
    
    try:
        # Load old format checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if not isinstance(checkpoint, PolicyRecord):
            logger.warning(f"Not a PolicyRecord checkpoint: {checkpoint_path}")
            return False
        
        pr = checkpoint
        
        # Extract necessary information
        if pr._cached_policy is None:
            logger.error(f"No cached policy in checkpoint: {checkpoint_path}")
            return False
        
        model = pr._cached_policy
        metadata = pr.metadata
        
        if dry_run:
            logger.info(f"[DRY RUN] Would migrate {checkpoint_path}")
            logger.info(f"[DRY RUN] Would create {checkpoint_path} (state_dict)")
            logger.info(f"[DRY RUN] Would create {metadata_path} (metadata)")
            return True
        
        # Create backup if requested
        if backup:
            backup_path = checkpoint_path + '.old'
            logger.info(f"Creating backup: {backup_path}")
            shutil.copy2(checkpoint_path, backup_path)
        
        # Determine output path
        output_model_path = output_path if output_path else checkpoint_path
        output_base = output_model_path[:-3] if output_model_path.endswith('.pt') else output_model_path
        output_metadata_path = output_base + '.json'
        
        # Save in new format
        # Save model state_dict
        torch.save(model.state_dict(), output_model_path + '.tmp')
        
        # Prepare metadata dict
        metadata_dict = dict(metadata)
        metadata_dict['run_name'] = pr.run_name
        metadata_dict['uri'] = pr.uri
        
        # Add model architecture info
        metadata_dict['model_info'] = {
            'type': type(model).__name__,
            'hidden_size': getattr(model, 'hidden_size', None),
            'core_num_layers': getattr(model, 'core_num_layers', None),
            'agent_attributes': getattr(model, 'agent_attributes', {}),
        }
        
        # Add action names if available
        if 'action_names' not in metadata_dict and hasattr(model, '_action_names'):
            metadata_dict['action_names'] = model._action_names
        
        # Save metadata JSON
        with open(output_metadata_path + '.tmp', 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        # Atomically replace files
        os.replace(output_model_path + '.tmp', output_model_path)
        os.replace(output_metadata_path + '.tmp', output_metadata_path)
        
        logger.info(f"Successfully migrated to:")
        logger.info(f"  - Model: {output_model_path}")
        logger.info(f"  - Metadata: {output_metadata_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to migrate {checkpoint_path}: {e}")
        # Clean up temp files if they exist
        for temp_path in [output_model_path + '.tmp', output_metadata_path + '.tmp']:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        return False


def migrate_directory(directory: str, recursive: bool = False, backup: bool = True, 
                     dry_run: bool = False) -> tuple[int, int]:
    """
    Migrate all checkpoints in a directory.
    
    Args:
        directory: Directory containing checkpoints
        recursive: Whether to search subdirectories
        backup: Whether to create backups
        dry_run: If True, only show what would be done
        
    Returns:
        Tuple of (successful_migrations, failed_migrations)
    """
    success_count = 0
    fail_count = 0
    
    pattern = '**/*.pt' if recursive else '*.pt'
    
    for checkpoint_path in Path(directory).glob(pattern):
        checkpoint_str = str(checkpoint_path)
        
        # Skip backup files
        if checkpoint_str.endswith('.old'):
            continue
            
        # Skip if already has metadata sidecar
        base_path = checkpoint_str[:-3]
        if os.path.exists(base_path + '.json'):
            logger.debug(f"Skipping (already migrated): {checkpoint_str}")
            continue
        
        if migrate_checkpoint(checkpoint_str, backup=backup, dry_run=dry_run):
            success_count += 1
        else:
            fail_count += 1
    
    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Metta checkpoints to new standard PyTorch format"
    )
    parser.add_argument(
        "path", 
        help="Path to checkpoint file or directory"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively search subdirectories"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files (.old)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    path = Path(args.path)
    
    if path.is_file():
        # Migrate single file
        success = migrate_checkpoint(
            str(path), 
            backup=not args.no_backup,
            dry_run=args.dry_run
        )
        sys.exit(0 if success else 1)
        
    elif path.is_dir():
        # Migrate directory
        success, fail = migrate_directory(
            str(path),
            recursive=args.recursive,
            backup=not args.no_backup,
            dry_run=args.dry_run
        )
        
        print(f"\nMigration complete:")
        print(f"  Successful: {success}")
        print(f"  Failed: {fail}")
        
        sys.exit(0 if fail == 0 else 1)
        
    else:
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()