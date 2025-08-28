"""Simple checkpoint manager - just the essentials for training and basic evaluation."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

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
        if not yaml_file.exists():
            return None

        try:
            with open(yaml_file) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata from {yaml_file}: {e}")
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

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """Clean up old checkpoints, keeping only the most recent ones."""
        try:
            if not self.checkpoint_dir.exists():
                logger.info(f"Checkpoint directory does not exist: {self.checkpoint_dir}")
                return 0

            # Get all agent checkpoint files
            agent_files = list(self.checkpoint_dir.glob("agent_epoch_*.pt"))
            if len(agent_files) <= keep_last_n:
                logger.info(f"Only {len(agent_files)} checkpoints found, nothing to clean up")
                return 0

            # Sort by epoch number (oldest first)
            try:
                agent_files.sort(key=lambda p: self._extract_epoch(p.name))
            except Exception as e:
                logger.error(f"Failed to sort checkpoint files: {e}")
                return 0

            # Determine which files to remove (all except the last N)
            files_to_remove = agent_files[:-keep_last_n]
            deleted_count = 0

            for agent_file in files_to_remove:
                try:
                    # Remove corresponding YAML metadata file if it exists
                    yaml_file = agent_file.with_suffix(".yaml")
                    if yaml_file.exists():
                        yaml_file.unlink()
                        logger.debug(f"Deleted metadata file: {yaml_file}")

                    # Remove trainer state file if it exists
                    epoch = self._extract_epoch(agent_file.name)
                    trainer_file = self.checkpoint_dir / f"trainer_epoch_{epoch}.pt"
                    if trainer_file.exists():
                        trainer_file.unlink()
                        logger.debug(f"Deleted trainer state file: {trainer_file}")

                    # Remove the agent checkpoint file
                    agent_file.unlink()
                    logger.info(f"Deleted checkpoint: {agent_file}")
                    deleted_count += 1

                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {agent_file}: {e}")

            logger.info(f"Cleanup complete: deleted {deleted_count} old checkpoints, kept {keep_last_n} most recent")
            return deleted_count

        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {e}")
            return 0

    def _validate_run_name(self, run_name: str) -> str:
        """Validate run_name to prevent path traversal attacks."""
        if not run_name:
            raise ValueError("run_name cannot be empty")

        # Allow only alphanumeric characters, underscores, hyphens, and dots
        if not re.match(r"^[a-zA-Z0-9._-]+$", run_name):
            raise ValueError(
                f"Invalid run_name '{run_name}': only alphanumeric characters, underscores, hyphens, and dots allowed"
            )

        # Prevent path traversal patterns
        if ".." in run_name or run_name.startswith(".") or "/" in run_name or "\\" in run_name:
            raise ValueError(f"Invalid run_name '{run_name}': path traversal patterns not allowed")

        # Reasonable length limit
        if len(run_name) > 128:
            raise ValueError(f"run_name too long: {len(run_name)} chars (max 128)")

        return run_name

    def _extract_epoch(self, filename: str) -> int:
        """Extract epoch number from filename with basic error handling."""
        try:
            # Expected format: agent_epoch_123.pt or trainer_epoch_123.pt
            if not filename.endswith(".pt"):
                raise ValueError(f"Expected .pt file, got: {filename}")

            # Remove .pt extension
            name_without_ext = filename[:-3]

            # Split by underscore and get last part (epoch number)
            parts = name_without_ext.split("_")
            if len(parts) < 3 or parts[1] != "epoch":
                raise ValueError(f"Expected format '[agent|trainer]_epoch_<number>.pt', got: {filename}")

            epoch_str = parts[-1]
            epoch = int(epoch_str)

            # Reasonable bounds checking
            if epoch < 0:
                raise ValueError(f"Epoch cannot be negative: {epoch}")
            if epoch > 1000000:  # 1M epochs should be enough for anyone
                raise ValueError(f"Epoch too large: {epoch}")

            return epoch

        except ValueError as e:
            logger.error(f"Failed to extract epoch from filename '{filename}': {e}")
            raise ValueError(f"Invalid filename format '{filename}': {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error extracting epoch from '{filename}': {e}")
            raise RuntimeError(f"Epoch extraction failed: {e}") from e
