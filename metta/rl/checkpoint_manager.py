"""Simple checkpoint manager for training and evaluation."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from metta.rl.wandb_policy_loader import get_wandb_artifact_metadata, load_policy_from_wandb_uri

logger = logging.getLogger(__name__)


def parse_checkpoint_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parse checkpoint metadata from filename.

    Format: {run_name}---e{epoch}_s{agent_step}_t{total_time}s.pt
    Example: kickstart_test---e5_s5280_t18s.pt
    """
    pattern = r"(.+)---e(\d+)_s(\d+)_t(\d+)s\.pt"
    match = re.match(pattern, filename)
    if match:
        return {
            "run": match.group(1),
            "epoch": int(match.group(2)),
            "agent_step": int(match.group(3)),
            "total_time": int(match.group(4)),
            "checkpoint_file": filename,
        }
    return None


class CheckpointManager:
    """Simple checkpoint manager: torch.save/load + filename-embedded metadata."""

    def __init__(self, run_name: str, run_dir: str = "./train_dir"):
        self.run_name = self._validate_run_name(run_name)
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / self.run_name / "checkpoints"

    def exists(self) -> bool:
        return self.checkpoint_dir.exists() and any(self.checkpoint_dir.glob(f"{self.run_name}---e*_s*_t*s.pt"))

    def load_latest_agent(self):
        """Load the latest agent using torch.load(weights_only=False)."""
        agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}---e*_s*_t*s.pt"))
        if not agent_files:
            return None

        # Get latest by epoch number from filename
        latest_file = max(agent_files, key=lambda p: parse_checkpoint_filename(p.name)["epoch"])
        logger.info(f"Loading agent from {latest_file}")
        return torch.load(latest_file, weights_only=False)

    def load_agent(self, epoch: Optional[int] = None):
        if epoch is None:
            return self.load_latest_agent()

        agent_file = self.checkpoint_dir / f"agent_epoch_{epoch}.pt"
        if not agent_file.exists():
            return None

        logger.info(f"Loading agent from {agent_file}")
        return torch.load(agent_file, weights_only=False)

    def load_trainer_state(self, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load trainer state (optimizer state, epoch, agent_step)."""
        if epoch is None:
            epoch = self.get_latest_epoch()
        if epoch is None:
            return None

        trainer_file = self.checkpoint_dir / f"trainer_epoch_{epoch}.pt"
        if not trainer_file.exists():
            return None

        logger.info(f"Loading trainer state from {trainer_file}")
        return torch.load(trainer_file, weights_only=False)

    def save_agent(self, agent, epoch: int, metadata: Dict[str, Any]):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Extract metadata for filename
        agent_step = metadata.get("agent_step", 0)
        total_time = int(metadata.get("total_time", 0))

        # Generate filename with embedded metadata
        filename = f"{self.run_name}---e{epoch}_s{agent_step}_t{total_time}s.pt"
        agent_file = self.checkpoint_dir / filename

        # Save agent with torch.save
        torch.save(agent, agent_file)

        logger.info(f"Saved agent: {agent_file}")

    def save_trainer_state(self, optimizer, epoch: int, agent_step: int):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainer_state = {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "agent_step": agent_step,
        }

        trainer_file = self.checkpoint_dir / f"trainer_epoch_{epoch}.pt"
        torch.save(trainer_state, trainer_file)

        logger.info(f"Saved trainer state: {trainer_file}")

    def save_checkpoint(
        self,
        agent,
        epoch: int,
        trainer_state: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
        agent_step: Optional[int] = None,
    ):
        """Save complete checkpoint with torch.save + YAML metadata."""
        metadata = {"score": score or 0.0, "agent_step": agent_step or 0}
        self.save_agent(agent, epoch, metadata)

        if trainer_state:
            # Assume trainer_state contains optimizer
            optimizer = trainer_state.get("optimizer")
            if optimizer:
                self.save_trainer_state(optimizer, epoch, agent_step or 0)

    def list_epochs(self) -> list[int]:
        return sorted(
            [
                metadata["epoch"]
                for f in self.checkpoint_dir.glob(f"{self.run_name}---e*_s*_t*s.pt")
                if (metadata := parse_checkpoint_filename(f.name)) is not None
            ]
        )

    def get_latest_epoch(self) -> Optional[int]:
        epochs = self.list_epochs()
        return epochs[-1] if epochs else None

    def load_metadata(self, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if epoch is None:
            epoch = self.get_latest_epoch()
        if epoch is None:
            return None

        # Find checkpoint file with this epoch
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.run_name}---e{epoch}_s*_t*s.pt"))
        if checkpoint_files:
            return parse_checkpoint_filename(checkpoint_files[0].name)
        return None

    def find_best_checkpoint(self, metric: str = "epoch") -> Optional[Path]:
        """Find checkpoint with highest value for the given metric."""

        def get_score(checkpoint_file):
            metadata = parse_checkpoint_filename(checkpoint_file.name)
            return metadata.get(metric, 0.0) if metadata else float("-inf")

        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.run_name}---e*_s*_t*s.pt"))
        return max(checkpoint_files, key=get_score, default=None) if checkpoint_files else None

    def select_checkpoints(
        self, strategy: str = "latest", count: int = 1, metric: str = "epoch", filters: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """Select checkpoints using different strategies."""
        if not self.checkpoint_dir.exists():
            return []

        # Parse and filter valid checkpoints
        checkpoints = [
            (file, metadata)
            for file in self.checkpoint_dir.glob(f"{self.run_name}---e*_s*_t*s.pt")
            if (metadata := parse_checkpoint_filename(file.name)) is not None
            and (not filters or self._matches_filters(metadata, filters))
        ]

        if not checkpoints:
            return []

        # Apply strategy-specific sorting
        sort_strategies = {
            "latest": lambda x: x[1].get("epoch", 0),
            "best_score": lambda x: x[1].get(metric, float("-inf")),
            "all": lambda x: 0,
        }

        if strategy not in sort_strategies:
            raise ValueError(f"Unknown selection strategy: {strategy}")

        checkpoints.sort(key=sort_strategies[strategy], reverse=True)
        return [cp[0] for cp in checkpoints[: count if strategy != "all" else len(checkpoints)]]

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters."""
        for key, filter_value in filters.items():
            if key not in metadata:
                return False

            metadata_value = metadata[key]

            if isinstance(filter_value, dict):
                # Handle range filters like {"min": 0.5, "max": 1.0}
                if "min" in filter_value and metadata_value < filter_value["min"]:
                    return False
                if "max" in filter_value and metadata_value > filter_value["max"]:
                    return False
            else:
                # Handle exact match filters
                if metadata_value != filter_value:
                    return False

        return True

    def load_policy_from_uri(self, uri: str, device: str = "cpu"):
        """Load a policy from either local file:// or wandb:// URI."""
        if uri.startswith("wandb://"):
            return load_policy_from_wandb_uri(uri, device)

        if uri.startswith("file://"):
            file_path = Path(uri[7:])  # Remove "file://" prefix
            if file_path.exists():
                return torch.load(file_path, map_location=device, weights_only=False)
            logger.error(f"Local file not found: {file_path}")
            return None

        logger.error(f"Unsupported URI format: {uri}. Supported: file://, wandb://")
        return None

    def get_policy_metadata_from_uri(self, uri: str) -> Dict[str, Any]:
        """Get metadata from a policy URI without loading the policy."""
        if uri.startswith("wandb://"):
            return get_wandb_artifact_metadata(uri)

        if uri.startswith("file://"):
            yaml_path = Path(uri[7:]).with_suffix(".yaml")  # Remove "file://" prefix
            if yaml_path.exists():
                try:
                    with open(yaml_path) as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {yaml_path}: {e}")

        return {}

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """Clean up old checkpoints, keeping only the most recent ones."""
        if not self.checkpoint_dir.exists():
            return 0

        agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}---e*_s*_t*s.pt"))
        if len(agent_files) <= keep_last_n:
            return 0

        # Sort by epoch and get files to remove
        agent_files.sort(key=lambda p: parse_checkpoint_filename(p.name)["epoch"])
        files_to_remove = agent_files[:-keep_last_n]

        deleted_count = 0
        for agent_file in files_to_remove:
            try:
                # Remove associated trainer file if it exists
                if metadata := parse_checkpoint_filename(agent_file.name):
                    trainer_file = self.checkpoint_dir / f"trainer_epoch_{metadata['epoch']}.pt"
                    trainer_file.unlink(missing_ok=True)

                agent_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {agent_file}: {e}")

        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} old checkpoints, kept {keep_last_n} most recent")
        return deleted_count

    def _validate_run_name(self, run_name: str) -> str:
        if not run_name or not re.match(r"^[a-zA-Z0-9._-]+$", run_name):
            raise ValueError(f"Invalid run_name: {run_name}")
        return run_name
