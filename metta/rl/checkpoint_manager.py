"""Simple checkpoint manager for training and evaluation."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def name_from_uri(uri: str) -> str:
    """Extract run name from any checkpoint URI."""
    if uri.startswith("file://"):
        path = Path(uri[7:])
        if path.suffix == ".pt":
            return parse_checkpoint_filename(path.name)[0]
        elif path.is_dir() or not path.suffix:
            if path.name == "checkpoints":
                return path.parent.name
            return path.name
        else:
            return path.stem
    elif uri.startswith("wandb://"):
        parts = uri[8:].split("/")
        if len(parts) >= 3:
            run_name = parts[2].split(":")[0]
            return run_name
    return "unknown"


def epoch_from_uri(uri: str) -> int:
    """Extract epoch directly from URI."""
    if uri.startswith("file://") and uri.endswith(".pt"):
        _, epoch, _, _ = parse_checkpoint_filename(Path(uri[7:]).name)
        return epoch
    return 0


def key_and_version(uri: str) -> tuple[str, int]:
    """Extract key (run name) and version (epoch) from a policy URI."""
    return name_from_uri(uri), epoch_from_uri(uri)


def parse_checkpoint_filename(filename: str) -> tuple[str, int, int, int]:
    """Parse checkpoint metadata from filename: {run_name}.e{epoch}.s{agent_step}.t{total_time}.pt"""
    parts = filename.split(".")
    if len(parts) != 5 or parts[-1] != "pt":
        raise ValueError(f"Invalid checkpoint filename format: {filename}")

    run_name = parts[0]
    epoch = int(parts[1][1:])
    agent_step = int(parts[2][1:])
    total_time = int(parts[3][1:])

    return run_name, epoch, agent_step, total_time


def get_checkpoint_uri_from_dir(checkpoint_dir: str) -> Optional[str]:
    """Get URI of latest checkpoint from directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    run_name = checkpoint_path.parent.name if checkpoint_path.parent else "unknown"
    checkpoints = list(checkpoint_path.glob(f"{run_name}.e*.s*.t*.pt"))
    if not checkpoints:
        return None

    latest_file = max(checkpoints, key=lambda p: parse_checkpoint_filename(p.name)[1])
    return f"file://{latest_file}"


def load_policy_from_dir(checkpoint_dir: str) -> Optional[Any]:
    """Load latest policy from directory."""
    uri = get_checkpoint_uri_from_dir(checkpoint_dir)
    if not uri:
        return None

    path = Path(uri[7:])
    return torch.load(path, weights_only=False)


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

        trainer_file = self.checkpoint_dir / f"{self.run_name}.e{epoch}.trainer.pt"
        if not trainer_file.exists():
            return None

        logger.info(f"Loading trainer state from {trainer_file}")
        return torch.load(trainer_file, weights_only=False)

    def save_agent(self, agent, epoch: int, metadata: Dict[str, Any]):
        """Save agent with metadata embedded in filename."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        agent_step = metadata.get("agent_step", 0)
        score = metadata.get("score", 0)

        filename = f"{self.run_name}.e{epoch}.s{agent_step}.t{int(score)}.pt"
        torch.save(agent, self.checkpoint_dir / filename)
        logger.info(f"Saved agent: {filename}")

    def save_trainer_state(self, optimizer, epoch: int, agent_step: int):
        """Save trainer optimizer state."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainer_file = self.checkpoint_dir / f"{self.run_name}.e{epoch}.trainer.pt"
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "agent_step": agent_step,
            },
            trainer_file,
        )
        logger.info(f"Saved trainer state: {trainer_file}")

    def get_checkpoint_uri(self, epoch: Optional[int] = None) -> Optional[str]:
        """Get URI for checkpoint at given epoch (or latest if None)."""
        if epoch is None:
            latest_file = self.find_best_checkpoint("epoch")
            if not latest_file:
                return None
        else:
            agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e{epoch}.s*.t*.pt"))
            if not agent_files:
                return None
            latest_file = agent_files[0]
        return f"file://{latest_file}"

    def get_latest_epoch(self) -> Optional[int]:
        """Get the latest epoch number."""
        latest_file = self.find_best_checkpoint("epoch")
        return parse_checkpoint_filename(latest_file.name)[1] if latest_file else None

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
                trainer_file = self.checkpoint_dir / f"{self.run_name}.e{epoch}.trainer.pt"
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
