import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def key_and_version(uri: str) -> tuple[str, int]:
    """Extract key (run name) and version (epoch) from a policy URI."""
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


def get_checkpoint_uri_from_dir(checkpoint_dir: str, epoch: int | None = None) -> str | None:
    """Get a checkpoint URI from a directory, loading latest if epoch not specified.

    Returns the file:// URI of the checkpoint, or None if not found.
    """
    from pathlib import Path

    dir_path = Path(checkpoint_dir)

    if not dir_path.exists():
        return None

    # Find all checkpoint files
    checkpoint_files = list(dir_path.glob("*.pt"))
    if not checkpoint_files:
        return None

    if epoch is None:
        # Get latest by epoch number
        latest = max(checkpoint_files, key=lambda p: parse_checkpoint_filename(p.name)[1] if p.suffix == ".pt" else -1)
        return f"file://{latest.absolute()}"
    else:
        # Find specific epoch
        for ckpt in checkpoint_files:
            try:
                _, ckpt_epoch, _, _, _ = parse_checkpoint_filename(ckpt.name)
                if ckpt_epoch == epoch:
                    return f"file://{ckpt.absolute()}"
            except ValueError:
                continue
        return None


def parse_checkpoint_filename(filename: str) -> tuple[str, int, int, int, float]:
    """Parse checkpoint metadata from filename: {run_name}.e{epoch}.s{agent_step}.t{total_time}.sc{score}.pt."""
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


class CheckpointManager:
    """Simple checkpoint manager: torch.save/load + filename-embedded metadata with LRU cache."""

    def __init__(self, run_name: str, run_dir: str = "./train_dir", cache_size: int = 3):
        if not run_name or not re.match(r"^[a-zA-Z0-9._-]+$", run_name):
            raise ValueError(f"Invalid run_name: {run_name}")
        self.run_name = run_name
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / self.run_name / "checkpoints"

        # Simple LRU cache using OrderedDict
        self.cache_size = max(0, cache_size)  # 0 means no caching
        self._cache = OrderedDict()  # path -> agent object

    def exists(self) -> bool:
        return self.checkpoint_dir.exists() and any(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.sc*.pt"))

    def _get_from_cache(self, path: Path):
        """Get agent from cache if available, maintaining LRU order."""
        if self.cache_size == 0:
            return None
        path_str = str(path)
        if path_str in self._cache:
            self._cache.move_to_end(path_str)  # Move to end (most recently used)
            return self._cache[path_str]
        return None

    def _add_to_cache(self, path: Path, agent):
        """Add agent to cache, evicting oldest if needed."""
        if self.cache_size == 0:
            return
        path_str = str(path)
        if path_str in self._cache:
            del self._cache[path_str]  # Remove if already exists (to update position)
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)  # Remove oldest (first item)
        self._cache[path_str] = agent

    def clear_cache(self):
        self._cache.clear()

    def load_agent(self, epoch: Optional[int] = None):
        """Load agent from checkpoint by epoch (or latest if None), with caching.

        Returns None if no checkpoints exist."""
        if epoch is None:
            agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.sc*.pt"))
            if not agent_files:
                return None
            agent_file = max(agent_files, key=lambda p: parse_checkpoint_filename(p.name)[1])
        else:
            agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e{epoch}.s*.t*.sc*.pt"))
            if not agent_files:
                logger.warning(f"No checkpoint found for {self.run_name} at epoch {epoch}")
                return None
            agent_file = agent_files[0]

        # Check cache first
        cached_agent = self._get_from_cache(agent_file)
        if cached_agent is not None:
            return cached_agent

        agent = torch.load(agent_file, weights_only=False)
        self._add_to_cache(agent_file, agent)
        return agent

    def load_trainer_state(self, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load trainer state (optimizer state, epoch, agent_step). Returns None if no trainer state exists."""
        if epoch is None:
            latest_file = self.find_best_checkpoint("epoch")
            if not latest_file:
                return None
            epoch = parse_checkpoint_filename(latest_file.name)[1]

        trainer_file = self.checkpoint_dir / f"{self.run_name}.e{epoch}.trainer.pt"
        if not trainer_file.exists():
            return None
        return torch.load(trainer_file, weights_only=False)

    def save_agent(self, agent, epoch: int, metadata: Dict[str, Any]):
        """Save agent with metadata embedded in filename."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        agent_step = metadata.get("agent_step", 0)
        total_time = metadata.get("total_time", 0)
        score = metadata.get("score", 0.0)  # Default to 0 if no evaluation done yet

        score_int = int(score * 10000)  # Format score as integer (preserve 4 decimal places)
        filename = f"{self.run_name}.e{epoch}.s{agent_step}.t{int(total_time)}.sc{score_int}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(agent, checkpoint_path)

        # Invalidate cache entry if it exists
        if str(checkpoint_path) in self._cache:
            del self._cache[str(checkpoint_path)]

    def save_trainer_state(self, optimizer, epoch: int, agent_step: int):
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

    def get_checkpoint_uri(self, epoch: Optional[int] = None) -> str:
        """Get URI for checkpoint at given epoch (or latest if None)."""
        if epoch is None:
            latest_file = self.find_best_checkpoint("epoch")
            if not latest_file:
                raise FileNotFoundError(f"No checkpoints found for {self.run_name}")
            return f"file://{latest_file}"

        # Find specific epoch
        agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e{epoch}.s*.t*.sc*.pt"))
        if not agent_files:
            raise FileNotFoundError(f"No checkpoint found for {self.run_name} at epoch {epoch}")
        return f"file://{agent_files[0]}"

    def get_latest_epoch(self) -> Optional[int]:
        checkpoints = self.select_checkpoints(strategy="latest", count=1, metric="epoch")
        return parse_checkpoint_filename(checkpoints[0].name)[1] if checkpoints else None

    def select_checkpoints(self, strategy: str = "latest", count: int = 1, metric: str = "epoch") -> List[Path]:
        """Select checkpoints based on strategy. Simplified since all metadata is in filenames."""
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.sc*.pt"))
        if not checkpoint_files:
            return []

        metric_idx = {"epoch": 1, "agent_step": 2, "total_time": 3, "score": 4}.get(metric, 1)
        checkpoint_files.sort(key=lambda f: parse_checkpoint_filename(f.name)[metric_idx], reverse=True)
        return checkpoint_files if strategy == "all" else checkpoint_files[:count]

    def find_best_checkpoint(self, metric: str = "epoch") -> Optional[Path]:
        """Find single checkpoint with highest value for the given metric."""
        checkpoints = self.select_checkpoints(count=1, metric=metric)
        return checkpoints[0] if checkpoints else None

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
            _, epoch, _, _, _ = parse_checkpoint_filename(agent_file.name)
            trainer_file = self.checkpoint_dir / f"{self.run_name}.e{epoch}.trainer.pt"
            trainer_file.unlink(missing_ok=True)
            agent_file.unlink()
            deleted_count += 1

        return deleted_count

    def upload_to_wandb(self, epoch: Optional[int] = None, wandb_run=None) -> Optional[str]:
        """Upload checkpoint to wandb as an artifact."""
        from metta.rl.wandb import upload_checkpoint_as_artifact

        if epoch is None:
            epoch = self.get_latest_epoch()
            if epoch is None:
                return None

        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e{epoch}.s*.t*.sc*.pt"))
        if not checkpoint_files:
            return None

        checkpoint_file = checkpoint_files[0]
        _, epoch_num, agent_step, total_time, score = parse_checkpoint_filename(checkpoint_file.name)

        return upload_checkpoint_as_artifact(
            checkpoint_path=str(checkpoint_file),
            artifact_name=self.run_name,
            artifact_type="model",
            metadata={
                "epoch": epoch_num,
                "agent_step": agent_step,
                "total_time": total_time,
                "score": score,
                "run_name": self.run_name,
            },
            wandb_run=wandb_run,
        )
