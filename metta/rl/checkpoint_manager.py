"""Simple checkpoint manager for training and evaluation."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from metta.rl.wandb_policy_loader import get_wandb_artifact_metadata, load_policy_from_wandb_uri

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Simple checkpoint manager: torch.save/load + basic metadata."""

    def __init__(self, run_name: str, run_dir: str = "./train_dir"):
        self.run_name = self._validate_run_name(run_name)
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / self.run_name / "checkpoints"

    def exists(self) -> bool:
        """Check if this run has any checkpoints."""
        return self.checkpoint_dir.exists() and any(self.checkpoint_dir.glob("agent_epoch_*.pt"))

    def load_latest_agent(self):
        """Load the latest agent using torch.load(weights_only=False)."""
        agent_files = list(self.checkpoint_dir.glob("agent_epoch_*.pt"))
        if not agent_files:
            return None

        # Get latest by epoch number from filename
        latest_file = max(agent_files, key=lambda p: self._extract_epoch(p.name))
        logger.info(f"Loading agent from {latest_file}")
        return torch.load(latest_file, weights_only=False)

    def load_agent(self, epoch: Optional[int] = None):
        """Load specific epoch or latest agent."""
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
        """Save agent with YAML metadata."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save agent with torch.save
        agent_file = self.checkpoint_dir / f"agent_epoch_{epoch}.pt"
        torch.save(agent, agent_file)

        # Extract relevant metadata fields
        score = metadata.get("score", 0.0)
        agent_step = metadata.get("agent_step", 0)

        # Save YAML metadata for integration
        yaml_metadata = {
            "run": self.run_name,
            "epoch": epoch,
            "agent_step": agent_step,
            "score": score,
            "checkpoint_file": agent_file.name,
        }

        yaml_file = self.checkpoint_dir / f"agent_epoch_{epoch}.yaml"
        with open(yaml_file, "w") as f:
            yaml.safe_dump(yaml_metadata, f, default_flow_style=False)

        logger.info(f"Saved agent: {agent_file}, metadata: {yaml_file}")

    def save_trainer_state(self, optimizer, epoch: int, agent_step: int):
        """Save trainer state (optimizer state)."""
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
        """List all available epochs."""
        agent_files = self.checkpoint_dir.glob("agent_epoch_*.pt")
        return sorted([self._extract_epoch(f.name) for f in agent_files])

    def get_latest_epoch(self) -> Optional[int]:
        """Get the latest epoch number."""
        epochs = self.list_epochs()
        return epochs[-1] if epochs else None

    def load_metadata(self, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load YAML metadata."""
        if epoch is None:
            epoch = self.get_latest_epoch()
        if epoch is None:
            return None

        yaml_file = self.checkpoint_dir / f"agent_epoch_{epoch}.yaml"
        if yaml_file.exists():
            with open(yaml_file) as f:
                return yaml.safe_load(f)
        return None

    def find_best_checkpoint(self, metric: str = "score") -> Optional[Path]:
        """Find checkpoint with best score."""
        best_score = float("-inf")
        best_file = None

        for yaml_file in self.checkpoint_dir.glob("agent_epoch_*.yaml"):
            try:
                with open(yaml_file) as f:
                    metadata = yaml.safe_load(f)
                    if metadata is None:
                        continue
                    score = metadata.get(metric, 0.0)
                    if score > best_score:
                        best_score = score
                        epoch = metadata.get("epoch")
                        best_file = self.checkpoint_dir / f"agent_epoch_{epoch}.pt"
            except Exception as e:
                logger.warning(f"Failed to process metadata file {yaml_file}: {e}")

        return best_file if best_file and best_file.exists() else None

    def select_checkpoints(
        self, strategy: str = "latest", count: int = 1, metric: str = "score", filters: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """Select checkpoints using different strategies.

        Supports "latest", "best_score", and "all" selection strategies.
        Optionally filter checkpoints by metadata criteria like minimum score thresholds."""
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []
        for yaml_file in self.checkpoint_dir.glob("agent_epoch_*.yaml"):
            try:
                with open(yaml_file) as f:
                    metadata = yaml.safe_load(f)

                epoch = metadata.get("epoch") if metadata else None
                if not epoch:
                    continue

                checkpoint_path = self.checkpoint_dir / f"agent_epoch_{epoch}.pt"
                if not checkpoint_path.exists():
                    continue

                if filters and not self._matches_filters(metadata, filters):
                    continue

                checkpoints.append((checkpoint_path, metadata))
            except Exception:
                continue

        if not checkpoints:
            return []

        if strategy == "latest":
            checkpoints.sort(key=lambda x: x[1].get("epoch", 0), reverse=True)
            return [cp[0] for cp in checkpoints[:count]]
        elif strategy == "best_score":
            checkpoints.sort(key=lambda x: x[1].get(metric, float("-inf")), reverse=True)
            return [cp[0] for cp in checkpoints[:count]]
        elif strategy == "all":
            return [cp[0] for cp in checkpoints]
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

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
        """Load a policy from either local file:// or wandb:// URI.

        Supports both local checkpoints and wandb artifacts for unified policy access.
        """
        if uri.startswith("wandb://"):
            return load_policy_from_wandb_uri(uri, device)
        elif uri.startswith("file://"):
            # Load from local file path
            file_path = Path(uri[7:])  # Remove "file://" prefix
            if file_path.exists():
                return torch.load(file_path, map_location=device, weights_only=False)
            else:
                logger.error(f"Local file not found: {file_path}")
                return None
        else:
            logger.error(f"Unsupported URI format: {uri}. Supported: file://, wandb://")
            return None

    def get_policy_metadata_from_uri(self, uri: str) -> Dict[str, Any]:
        """Get metadata from a policy URI without loading the policy.

        Returns basic information for selection and filtering purposes.
        """
        if uri.startswith("wandb://"):
            return get_wandb_artifact_metadata(uri)
        elif uri.startswith("file://"):
            # Load metadata from local YAML file
            file_path = Path(uri[7:])  # Remove "file://" prefix
            yaml_path = file_path.with_suffix(".yaml")
            if yaml_path.exists():
                try:
                    with open(yaml_path) as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {yaml_path}: {e}")
            return {}
        else:
            return {}

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """Clean up old checkpoints, keeping only the most recent ones."""
        if not self.checkpoint_dir.exists():
            return 0

        agent_files = list(self.checkpoint_dir.glob("agent_epoch_*.pt"))
        if len(agent_files) <= keep_last_n:
            return 0

        agent_files.sort(key=lambda p: self._extract_epoch(p.name))
        files_to_remove = agent_files[:-keep_last_n]

        deleted_count = 0
        for agent_file in files_to_remove:
            try:
                yaml_file = agent_file.with_suffix(".yaml")
                yaml_file.unlink(missing_ok=True)

                epoch = self._extract_epoch(agent_file.name)
                trainer_file = self.checkpoint_dir / f"trainer_epoch_{epoch}.pt"
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

    def _extract_epoch(self, filename: str) -> int:
        """Extract epoch number from filename."""
        if not filename.endswith(".pt"):
            raise ValueError(f"Expected .pt file: {filename}")

        parts = filename[:-3].split("_")
        if len(parts) < 3 or parts[1] != "epoch":
            raise ValueError(f"Invalid checkpoint filename: {filename}")

        return int(parts[-1])
