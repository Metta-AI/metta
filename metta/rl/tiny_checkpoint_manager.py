"""Ultra-simple checkpoint manager using direct torch.save/load with YAML metadata."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


class TinyCheckpointManager:
    """Minimal checkpoint manager: torch.save/load + YAML metadata for PolicyEvaluator integration."""

    def __init__(self, run_name: str, run_dir: str = "./train_dir"):
        self.run_name = run_name
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / run_name / "checkpoints"

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
        """Save agent with YAML metadata for PolicyEvaluator integration."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save agent with torch.save
        agent_file = self.checkpoint_dir / f"agent_epoch_{epoch}.pt"
        torch.save(agent, agent_file)

        # Extract relevant metadata fields
        score = metadata.get("score", 0.0)
        agent_step = metadata.get("agent_step", 0)

        # Save YAML metadata for PolicyEvaluator and database integration
        yaml_metadata = {
            "run": self.run_name,
            "epoch": epoch,
            "agent_step": agent_step,
            "score": score,
            "checkpoint_file": agent_file.name,
        }

        yaml_file = self.checkpoint_dir / f"agent_epoch_{epoch}.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_metadata, f, default_flow_style=False)

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
        """Load YAML metadata for PolicyEvaluator integration."""
        if epoch is None:
            epoch = self.get_latest_epoch()
        if epoch is None:
            return None

        yaml_file = self.checkpoint_dir / f"agent_epoch_{epoch}.yaml"
        if not yaml_file.exists():
            return None

        with open(yaml_file) as f:
            return yaml.safe_load(f)

    def find_best_checkpoint(self, metric: str = "score") -> Optional[Path]:
        """Find checkpoint with best score for PolicyEvaluator."""
        best_score = float("-inf")
        best_file = None

        for yaml_file in self.checkpoint_dir.glob("agent_epoch_*.yaml"):
            with open(yaml_file) as f:
                metadata = yaml.safe_load(f)
                score = metadata.get(metric, 0.0)
                if score > best_score:
                    best_score = score
                    epoch = metadata.get("epoch")
                    best_file = self.checkpoint_dir / f"agent_epoch_{epoch}.pt"

        return best_file if best_file and best_file.exists() else None

    def _extract_epoch(self, filename: str) -> int:
        """Extract epoch number from filename like 'agent_epoch_123.pt'."""
        return int(filename.split("_")[-1].split(".")[0])
