#!/usr/bin/env python3
"""
Migration tool for converting old PolicyRecord checkpoints to new MettaAgent format.

Usage:
    python -m metta.agent.migrate_checkpoints <input_path> [output_path]

Examples:
    # Migrate a single checkpoint
    python -m metta.agent.migrate_checkpoints checkpoints/old_model.pt

    # Migrate all checkpoints in a directory
    python -m metta.agent.migrate_checkpoints checkpoints/

    # Migrate to a different output directory
    python -m metta.agent.migrate_checkpoints checkpoints/ migrated_checkpoints/
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointMigrator:
    """Handles migration of old checkpoint formats to the new MettaAgent format."""

    def __init__(self, backup: bool = True):
        self.backup = backup
        self.migration_stats = {
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
        }

    def detect_checkpoint_type(self, checkpoint_data: any) -> str:
        """Detect the type/format of the checkpoint."""
        if hasattr(checkpoint_data, "_policy") and hasattr(checkpoint_data, "metadata"):
            return "PolicyRecord"
        elif isinstance(checkpoint_data, dict):
            if "model_state_dict" in checkpoint_data:
                return "MettaAgent_v2"  # New format
            elif "checkpoint_format_version" in checkpoint_data:
                return f"MettaAgent_v{checkpoint_data['checkpoint_format_version']}"
            elif any(k.startswith("components.") for k in checkpoint_data.keys()):
                return "raw_state_dict"
            elif "model" in checkpoint_data:
                return "MettaAgent_training"  # save_for_training format
        elif isinstance(checkpoint_data, torch.nn.Module):
            return "raw_model"
        return "unknown"

    def migrate_policy_record(self, path: str, checkpoint_data: any) -> Optional[Dict]:
        """Migrate a PolicyRecord checkpoint to MettaAgent format."""
        logger.info(f"Migrating PolicyRecord checkpoint: {path}")

        try:
            # Extract data from PolicyRecord
            policy = checkpoint_data._policy
            metadata = checkpoint_data.metadata
            name = checkpoint_data.name
            uri = checkpoint_data.uri

            # Determine model type
            from metta.agent.brain_policy import BrainPolicy
            from metta.rl.policy import PytorchAgent

            if isinstance(policy, BrainPolicy):
                model_type = "brain"
            elif isinstance(policy, PytorchAgent):
                model_type = "pytorch"
            else:
                model_type = "unknown"

            # Create new format checkpoint
            new_checkpoint = {
                "model_state_dict": policy.state_dict() if policy else None,
                "model_type": model_type,
                "name": name,
                "uri": uri,
                "metadata": metadata,
                "observation_space_version": None,
                "action_space_version": None,
                "layer_version": None,
                "checkpoint_format_version": 2,
                "migration_info": {
                    "migrated_from": "PolicyRecord",
                    "original_path": path,
                    "migration_date": str(torch.datetime.datetime.now()),
                },
            }

            # Try to save component config for BrainPolicy
            if model_type == "brain" and hasattr(policy, "agent_attributes"):
                new_checkpoint["agent_attributes"] = policy.agent_attributes

            return new_checkpoint

        except Exception as e:
            logger.error(f"Failed to migrate PolicyRecord: {e}")
            return None

    def migrate_raw_state_dict(self, path: str, checkpoint_data: Dict) -> Optional[Dict]:
        """Migrate a raw state dict to MettaAgent format."""
        logger.info(f"Migrating raw state dict checkpoint: {path}")

        # Extract epoch from filename if possible
        epoch = 0
        filename = os.path.basename(path)
        if "model_" in filename:
            try:
                epoch = int(filename.split("_")[1].split(".")[0])
            except:
                pass

        return {
            "model_state_dict": checkpoint_data,
            "model_type": "brain",  # Assume brain for raw state dicts
            "name": filename,
            "uri": f"file://{path}",
            "metadata": {
                "epoch": epoch,
                "warning": "Migrated from raw state dict; some metadata unavailable",
            },
            "observation_space_version": None,
            "action_space_version": None,
            "layer_version": None,
            "checkpoint_format_version": 2,
            "migration_info": {
                "migrated_from": "raw_state_dict",
                "original_path": path,
                "migration_date": str(torch.datetime.datetime.now()),
            },
        }

    def migrate_checkpoint(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """Migrate a single checkpoint file."""
        if not input_path.endswith(".pt"):
            logger.debug(f"Skipping non-.pt file: {input_path}")
            return False

        self.migration_stats["total"] += 1

        try:
            # Load checkpoint
            checkpoint_data = torch.load(input_path, map_location="cpu", weights_only=False)
            checkpoint_type = self.detect_checkpoint_type(checkpoint_data)

            logger.info(f"Detected checkpoint type: {checkpoint_type} for {input_path}")

            # Skip if already in new format
            if checkpoint_type.startswith("MettaAgent_v"):
                logger.info(f"Checkpoint already in MettaAgent format: {input_path}")
                self.migration_stats["skipped"] += 1
                return True

            # Migrate based on type
            new_checkpoint = None
            if checkpoint_type == "PolicyRecord":
                new_checkpoint = self.migrate_policy_record(input_path, checkpoint_data)
            elif checkpoint_type == "raw_state_dict":
                new_checkpoint = self.migrate_raw_state_dict(input_path, checkpoint_data)
            else:
                logger.warning(f"Cannot migrate checkpoint type: {checkpoint_type}")
                self.migration_stats["failed"] += 1
                return False

            if new_checkpoint is None:
                self.migration_stats["failed"] += 1
                return False

            # Backup original if requested
            if self.backup:
                backup_path = input_path + ".backup"
                shutil.copy2(input_path, backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Save migrated checkpoint
            save_path = output_path or input_path
            torch.save(new_checkpoint, save_path)
            logger.info(f"Saved migrated checkpoint: {save_path}")

            self.migration_stats["succeeded"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to migrate {input_path}: {e}")
            self.migration_stats["failed"] += 1
            return False

    def migrate_directory(self, input_dir: str, output_dir: Optional[str] = None) -> None:
        """Migrate all checkpoints in a directory."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Find all .pt files
        pt_files = sorted(input_path.glob("*.pt"))
        logger.info(f"Found {len(pt_files)} .pt files in {input_dir}")

        for pt_file in pt_files:
            if output_dir:
                output_file = Path(output_dir) / pt_file.name
            else:
                output_file = None

            self.migrate_checkpoint(str(pt_file), str(output_file) if output_file else None)

    def print_summary(self):
        """Print migration summary."""
        print("\nMigration Summary:")
        print(f"  Total files processed: {self.migration_stats['total']}")
        print(f"  Successfully migrated: {self.migration_stats['succeeded']}")
        print(f"  Already in new format: {self.migration_stats['skipped']}")
        print(f"  Failed to migrate: {self.migration_stats['failed']}")


def main():
    parser = argparse.ArgumentParser(description="Migrate old PolicyRecord checkpoints to new MettaAgent format")
    parser.add_argument("input_path", help="Path to checkpoint file or directory")
    parser.add_argument("output_path", nargs="?", help="Optional output path")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create migrator
    migrator = CheckpointMigrator(backup=not args.no_backup)

    # Perform migration
    input_path = Path(args.input_path)
    if input_path.is_file():
        success = migrator.migrate_checkpoint(args.input_path, args.output_path)
        if not success:
            sys.exit(1)
    elif input_path.is_dir():
        migrator.migrate_directory(args.input_path, args.output_path)
    else:
        print(f"Error: {args.input_path} is neither a file nor a directory")
        sys.exit(1)

    # Print summary
    migrator.print_summary()

    # Exit with error if any migrations failed
    if migrator.migration_stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
