"""Simple checkpoint manager for training and evaluation."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def parse_checkpoint_filename(filename: str) -> tuple[str, int, int, int]:
    """Parse checkpoint metadata from filename: {run_name}.e{epoch}.s{agent_step}.t{total_time}.pt

    Returns: (run_name, epoch, agent_step, total_time)
    """
    parts = filename.split(".")
    if len(parts) != 5 or parts[-1] != "pt":
        raise ValueError(f"Invalid checkpoint filename format: {filename}")

    run_name = parts[0]
    epoch = int(parts[1][1:])  # Remove 'e' prefix
    agent_step = int(parts[2][1:])  # Remove 's' prefix
    total_time = int(parts[3][1:])  # Remove 't' prefix

    return run_name, epoch, agent_step, total_time


class CheckpointManager:
    """Simple checkpoint manager: torch.save/load + filename-embedded metadata."""

    def __init__(self, run_name: str, run_dir: str = "./train_dir"):
        self.run_name = self._validate_run_name(run_name)
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / self.run_name / "checkpoints"

    def exists(self) -> bool:
        return self.checkpoint_dir.exists() and any(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.pt"))

    def load_agent(self, epoch: Optional[int] = None):
        """Load agent from checkpoint by epoch (or latest if None)."""
        if epoch is None:
            agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.pt"))
            if not agent_files:
                return None
            agent_file = max(agent_files, key=lambda p: parse_checkpoint_filename(p.name)[1])
        else:
            agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e{epoch}.s*.t*.pt"))
            if not agent_files:
                return None
            agent_file = agent_files[0]

        logger.info(f"Loading agent from {agent_file}")
        return torch.load(agent_file, weights_only=False)

    def load_trainer_state(self, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load trainer state (optimizer state, epoch, agent_step)."""
        epoch = epoch or self.get_latest_epoch()
        if epoch is None:
            return None

        trainer_file = self.checkpoint_dir / f"trainer_epoch_{epoch}.pt"
        if not trainer_file.exists():
            return None

        logger.info(f"Loading trainer state from {trainer_file}")
        return torch.load(trainer_file, weights_only=False)

    def save_checkpoint(
        self,
        agent,
        epoch: int,
        trainer_state: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
        agent_step: Optional[int] = None,
    ):
        """Save complete checkpoint with filename-embedded metadata."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save agent with embedded metadata in filename
        filename = f"{self.run_name}.e{epoch}.s{agent_step or 0}.t{int(score or 0)}.pt"
        torch.save(agent, self.checkpoint_dir / filename)
        logger.info(f"Saved agent: {filename}")

        # Save trainer state if provided
        if trainer_state and trainer_state.get("optimizer"):
            trainer_file = self.checkpoint_dir / f"trainer_epoch_{epoch}.pt"
            torch.save(
                {
                    "optimizer": trainer_state["optimizer"].state_dict(),
                    "epoch": epoch,
                    "agent_step": agent_step or 0,
                },
                trainer_file,
            )
            logger.info(f"Saved trainer state: {trainer_file}")

    def get_latest_epoch(self) -> Optional[int]:
        agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.pt"))
        if not agent_files:
            return None
        latest_file = max(agent_files, key=lambda p: parse_checkpoint_filename(p.name)[1])
        return parse_checkpoint_filename(latest_file.name)[1]

    def find_best_checkpoint(self, metric: str = "epoch") -> Optional[Path]:
        """Find checkpoint with highest value for the given metric."""
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.pt"))
        if not checkpoint_files:
            return None
        metric_idx = {"epoch": 1, "agent_step": 2, "total_time": 3}.get(metric, 1)
        return max(checkpoint_files, key=lambda f: parse_checkpoint_filename(f.name)[metric_idx])

    def select_checkpoints(self, strategy: str = "latest", count: int = 1, metric: str = "epoch") -> List[Path]:
        """Select checkpoints using different strategies."""
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.pt"))
        if not checkpoint_files:
            return []

        if strategy == "latest":
            checkpoint_files.sort(key=lambda f: parse_checkpoint_filename(f.name)[1], reverse=True)
        elif strategy in ["best_score", "top"]:
            metric_idx = {"epoch": 1, "agent_step": 2, "total_time": 3}.get(metric, 1)
            checkpoint_files.sort(key=lambda f: parse_checkpoint_filename(f.name)[metric_idx], reverse=True)
        elif strategy == "all":
            count = len(checkpoint_files)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

        return checkpoint_files[:count]

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """Clean up old checkpoints, keeping only the most recent ones."""
        if not self.checkpoint_dir.exists():
            return 0

        agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.pt"))
        if len(agent_files) <= keep_last_n:
            return 0

        agent_files.sort(key=lambda p: parse_checkpoint_filename(p.name)[1])
        files_to_remove = agent_files[:-keep_last_n]

        deleted_count = 0
        for agent_file in files_to_remove:
            try:
                _, epoch, _, _ = parse_checkpoint_filename(agent_file.name)
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
