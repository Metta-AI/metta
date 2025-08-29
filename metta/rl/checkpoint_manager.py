"""Simple checkpoint manager for training and evaluation."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def key_and_version(uri: str) -> tuple[str, int]:
    """Extract key (run name) and version (epoch) from a policy URI.
    
    Since all checkpoints are .pt files with metadata in filenames,
    we can simplify this to handle just the common cases.
    """
    if uri.startswith("file://"):
        path = Path(uri[7:])
        if path.suffix == ".pt":
            # All checkpoints are .pt files with embedded metadata
            parsed = parse_checkpoint_filename(path.name)
            return parsed[0], parsed[1]  # run_name, epoch
        # For directory URIs, extract run name from path
        return path.stem if path.suffix else path.name, 0
    elif uri.startswith("wandb://"):
        # Extract run name from wandb URI format
        parts = uri[8:].split("/")
        run_name = parts[2].split(":")[0] if len(parts) >= 3 else "unknown"
        return run_name, 0
    return "unknown", 0


def parse_checkpoint_filename(filename: str) -> tuple[str, int, int, int, float]:
    """Parse checkpoint metadata from filename.
    
    Format: {run_name}.e{epoch}.s{agent_step}.t{total_time}.sc{score}.pt
    - e: epoch
    - s: agent_step  
    - t: total_time
    - sc: score (evaluation score, 0 if not evaluated)
    """
    parts = filename.split(".")
    if len(parts) != 6 or parts[-1] != "pt":
        raise ValueError(f"Invalid checkpoint filename format: {filename}")

    run_name = parts[0]
    epoch = int(parts[1][1:])  # Remove 'e' prefix
    agent_step = int(parts[2][1:])  # Remove 's' prefix
    total_time = int(parts[3][1:])  # Remove 't' prefix
    score_int = int(parts[4][2:])  # Remove 'sc' prefix (stored as int * 10000)
    score = score_int / 10000.0  # Convert back to float

    return run_name, epoch, agent_step, total_time, score


def get_checkpoint_uri_from_dir(checkpoint_dir: str) -> Optional[str]:
    """Get URI of latest checkpoint from directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # Find all checkpoint files (any run name)
    checkpoints = list(checkpoint_path.glob("*.e*.s*.t*.sc*.pt"))
    if not checkpoints:
        return None

    # Return the latest by epoch
    latest_file = max(checkpoints, key=lambda p: parse_checkpoint_filename(p.name)[1])
    return f"file://{latest_file}"


class CheckpointManager:
    """Simple checkpoint manager: torch.save/load + filename-embedded metadata."""

    def __init__(self, run_name: str, run_dir: str = "./train_dir"):
        self.run_name = self._validate_run_name(run_name)
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / self.run_name / "checkpoints"

    def exists(self) -> bool:
        return self.checkpoint_dir.exists() and any(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.sc*.pt"))

    def load_agent(self, epoch: Optional[int] = None):
        """Load agent from checkpoint by epoch (or latest if None)."""
        if epoch is None:
            agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.sc*.pt"))
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
        if epoch is None:
            latest_file = self.find_best_checkpoint("epoch")
            epoch = parse_checkpoint_filename(latest_file.name)[1] if latest_file else None
        if epoch is None:
            return None

        trainer_file = self.checkpoint_dir / f"{self.run_name}.e{epoch}.trainer.pt"
        if not trainer_file.exists():
            return None

        logger.info(f"Loading trainer state from {trainer_file}")
        return torch.load(trainer_file, weights_only=False)

    def save_agent(self, agent, epoch: int, metadata: Dict[str, Any]):
        """Save agent with metadata embedded in filename.
        
        Filename format: {run_name}.e{epoch}.s{agent_step}.t{total_time}.sc{score}.pt
        Score defaults to 0 if not provided (e.g., before evaluation).
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        agent_step = metadata.get("agent_step", 0)
        total_time = metadata.get("total_time", 0)
        score = metadata.get("score", 0.0)  # Default to 0 if no evaluation done yet

        # Format score as integer (multiply by 10000 to preserve 4 decimal places)
        score_int = int(score * 10000)
        filename = f"{self.run_name}.e{epoch}.s{agent_step}.t{int(total_time)}.sc{score_int}.pt"
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
            return f"file://{latest_file}" if latest_file else None
        
        # Find specific epoch
        agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e{epoch}.s*.t*.pt"))
        return f"file://{agent_files[0]}" if agent_files else None

    def get_latest_epoch(self) -> Optional[int]:
        """Get the latest epoch number."""
        latest_file = self.find_best_checkpoint("epoch")
        return parse_checkpoint_filename(latest_file.name)[1] if latest_file else None

    def find_best_checkpoint(self, metric: str = "epoch") -> Optional[Path]:
        """Find checkpoint with highest value for the given metric."""
        checkpoints = self.select_checkpoints(strategy="latest", count=1, metric=metric)
        return checkpoints[0] if checkpoints else None

    def select_checkpoints(self, strategy: str = "latest", count: int = 1, metric: str = "epoch") -> List[Path]:
        """Select checkpoints based on strategy. Simplified since all metadata is in filenames."""
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.sc*.pt"))
        if not checkpoint_files:
            return []

        # Simple metric index mapping (parse returns: run_name, epoch, agent_step, total_time, score)
        metric_idx = {"epoch": 1, "agent_step": 2, "total_time": 3, "score": 4}.get(metric, 1)
        
        # Sort by the selected metric (descending)
        checkpoint_files.sort(key=lambda f: parse_checkpoint_filename(f.name)[metric_idx], reverse=True)
        
        # Return all files if strategy is "all", otherwise return count
        return checkpoint_files if strategy == "all" else checkpoint_files[:count]

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """Clean up old checkpoints, keeping only the most recent ones."""
        if not self.checkpoint_dir.exists():
            return 0

        agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.sc*.pt"))
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
