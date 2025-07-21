#!/usr/bin/env python3
"""
Migrate Metta checkpoints from old format to new standard PyTorch format.

This tool converts checkpoints from the old format (single file with pickled PolicyRecord)
to the new format (separate .pt file with state_dict and .json file with metadata).

Usage:
    # Dry run to see what would be migrated
    python tools/migrate_checkpoints.py checkpoints/ --dry-run
    
    # Actually migrate all checkpoints
    python tools/migrate_checkpoints.py checkpoints/
    
    # Migrate a specific checkpoint
    python tools/migrate_checkpoints.py checkpoints/model_0000.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from omegaconf import DictConfig

# Add parent directory to path so we can import from metta
sys.path.insert(0, str(Path(__file__).parent.parent))

from metta.agent.policy_store import PolicyStore, migrate_all_checkpoints_in_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_minimal_config(device: str = "cpu") -> DictConfig:
    """Create minimal config for PolicyStore initialization."""
    return DictConfig({
        "device": device,
        "run": "migration",
        "run_dir": os.getcwd(),
        "trainer": {
            "checkpoint": {"checkpoint_dir": os.getcwd()},
            "num_workers": 1,
        },
        "data_dir": os.getcwd(),
        "agent": {
            "type": "metta",
            "hidden_size": 256,
            "num_layers": 2,
        },
    })


def migrate_single_checkpoint(checkpoint_path: str, policy_store: PolicyStore) -> str:
    """Migrate a single checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    # Check if already in new format
    base_path = checkpoint_path[:-3] if checkpoint_path.endswith('.pt') else checkpoint_path
    if os.path.exists(base_path + '.json'):
        logger.info(f"Checkpoint {checkpoint_path} is already in new format")
        return checkpoint_path
        
    # Migrate
    new_path = policy_store.migrate_checkpoint(checkpoint_path)
    return new_path


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Metta checkpoints to new PyTorch standard format"
    )
    parser.add_argument(
        "path",
        help="Path to checkpoint file or directory containing checkpoints"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually doing it"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for loading checkpoints (default: cpu)"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace original files instead of creating .new.pt files"
    )
    
    args = parser.parse_args()
    
    # Create PolicyStore with minimal config
    config = create_minimal_config(args.device)
    policy_store = PolicyStore(config, wandb_run=None)
    
    path = Path(args.path)
    
    if path.is_file():
        # Migrate single file
        if args.dry_run:
            base_path = str(path)[:-3] if str(path).endswith('.pt') else str(path)
            new_path = base_path + '.new.pt'
            logger.info(f"Would migrate: {path} -> {new_path}")
        else:
            try:
                new_path = migrate_single_checkpoint(str(path), policy_store)
                logger.info(f"Successfully migrated {path} to {new_path}")
                
                if args.replace:
                    # Move new files to original names
                    base_path = str(path)[:-3]
                    os.rename(new_path, str(path))
                    os.rename(base_path + '.new.json', base_path + '.json')
                    logger.info(f"Replaced original files")
                    
            except Exception as e:
                logger.error(f"Failed to migrate {path}: {e}")
                sys.exit(1)
                
    elif path.is_dir():
        # Migrate all checkpoints in directory
        logger.info(f"Scanning directory: {path}")
        migrated = migrate_all_checkpoints_in_dir(str(path), policy_store, dry_run=args.dry_run)
        
        if args.dry_run:
            logger.info(f"Would migrate {len(migrated)} checkpoint(s)")
        else:
            logger.info(f"Migrated {len(migrated)} checkpoint(s)")
            
            if args.replace and migrated:
                # Replace original files
                for old_path, new_path in migrated:
                    base_path = old_path[:-3]
                    os.rename(new_path, old_path)
                    os.rename(base_path + '.new.json', base_path + '.json')
                logger.info(f"Replaced {len(migrated)} original files")
                
    else:
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)
        
    logger.info("Migration complete!")


if __name__ == "__main__":
    main()