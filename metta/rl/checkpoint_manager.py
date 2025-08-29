import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from metta.mettagrid.util.file import WandbURI, local_copy
from metta.rl.wandb import get_wandb_checkpoint_metadata, load_policy_from_wandb_uri, upload_checkpoint_as_artifact

logger = logging.getLogger(__name__)


def expand_wandb_uri(uri: str, default_project: str = "metta") -> str:
    """Expand short wandb URI formats to full format."""
    if not uri.startswith("wandb://"):
        return uri

    path = uri[8:]  # Remove "wandb://"

    # Check for short format patterns
    if path.startswith("run/"):
        run_name = path[4:]  # Remove "run/"
        if ":" in run_name:
            run_name, version = run_name.split(":", 1)
        else:
            version = "latest"
        return f"wandb://{default_project}/model/{run_name}:{version}"

    elif path.startswith("sweep/"):
        sweep_name = path[6:]  # Remove "sweep/"
        if ":" in sweep_name:
            sweep_name, version = sweep_name.split(":", 1)
        else:
            version = "latest"
        return f"wandb://{default_project}/sweep_model/{sweep_name}:{version}"

    # Already in full format or unrecognized pattern - return as-is
    return uri


def key_and_version(uri: str) -> tuple[str, int]:
    """Extract key (run name) and version (epoch) from a policy URI."""
    if uri.startswith("file://"):
        path = Path(uri[7:])
        if path.suffix == ".pt" and is_valid_checkpoint_filename(path.name):
            return parse_checkpoint_filename(path.name)[:2]
        return path.stem if path.suffix else path.name, 0

    if uri.startswith("wandb://"):
        expanded_uri = expand_wandb_uri(uri)
        metadata = get_wandb_checkpoint_metadata(expanded_uri)
        if metadata:
            return metadata["run_name"], metadata["epoch"]
        wandb_uri = WandbURI.parse(expanded_uri)
        artifact_name = wandb_uri.artifact_path.split("/")[-1].split(":")[0]
        return artifact_name, 0

    if uri.startswith("s3://"):
        filename = uri.split("/")[-1]
        if filename.endswith(".pt") and is_valid_checkpoint_filename(filename):
            return parse_checkpoint_filename(filename)[:2]
        path = Path(filename)
        return path.stem if path.suffix else path.name, 0

    return "unknown", 0


def is_valid_checkpoint_filename(filename: str) -> bool:
    """Check if filename matches format: run_name__e{epoch}__s{step}__t{time}__sc{score}.pt"""
    if not filename.endswith(".pt"):
        return False

    parts = filename[:-3].split("__")
    if len(parts) != 5:
        return False
    return (
        parts[1].startswith("e")
        and parts[1][1:].isdigit()
        and parts[2].startswith("s")
        and parts[2][1:].isdigit()
        and parts[3].startswith("t")
        and parts[3][1:].isdigit()
        and parts[4].startswith("sc")
        and parts[4][2:].isdigit()
    )


def parse_checkpoint_filename(filename: str) -> tuple[str, int, int, int, float]:
    """Parse checkpoint metadata from filename."""
    if not is_valid_checkpoint_filename(filename):
        raise ValueError(f"Invalid checkpoint filename format: {filename}")
    parts = filename[:-3].split("__")
    run_name = parts[0]
    epoch = int(parts[1][1:])
    agent_step = int(parts[2][1:])
    total_time = int(parts[3][1:])
    score = int(parts[4][2:]) / 10000.0
    return (run_name, epoch, agent_step, total_time, score)


class CheckpointManager:
    """Checkpoint manager with filename-embedded metadata and LRU cache."""

    def __init__(self, run_name: str, run_dir: str = "./train_dir", cache_size: int = 3):
        self.run_name = run_name
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / self.run_name / "checkpoints"
        self.cache_size = cache_size
        self._cache = OrderedDict()

    @staticmethod
    def load_from_uri(uri: str):
        """Load a policy from file://, s3://, or wandb:// URI."""
        try:
            if uri.startswith("file://"):
                path = Path(uri[7:])
                if path.is_file() and path.suffix == ".pt":
                    return torch.load(path, weights_only=False)
                if path.is_dir():
                    if path.name != "checkpoints":
                        path = path / "checkpoints"
                    checkpoint_files = list(path.glob("*.pt"))
                    if not checkpoint_files:
                        return None
                    valid_checkpoints = [
                        (ckpt, parse_checkpoint_filename(ckpt.name)[1])
                        for ckpt in checkpoint_files
                        if is_valid_checkpoint_filename(ckpt.name)
                    ]
                    if not valid_checkpoints:
                        return torch.load(checkpoint_files[0], weights_only=False)
                    latest_checkpoint = max(valid_checkpoints, key=lambda x: x[1])[0]
                    return torch.load(latest_checkpoint, weights_only=False)
                return None
            if uri.startswith("s3://"):
                with local_copy(uri) as local_path:
                    return torch.load(local_path, weights_only=False)
            if uri.startswith("wandb://"):
                expanded_uri = expand_wandb_uri(uri)
                return load_policy_from_wandb_uri(expanded_uri, device="cpu")
            return None
        except Exception:
            return None

    @staticmethod
    def normalize_uri(path_or_uri: str) -> str:
        """Convert path to URI format and expand short wandb URIs."""
        if not path_or_uri.startswith(("file://", "wandb://", "s3://")):
            return f"file://{Path(path_or_uri).resolve()}"
        if path_or_uri.startswith("wandb://"):
            return expand_wandb_uri(path_or_uri)
        return path_or_uri

    @staticmethod
    def get_policy_metadata(uri: str) -> dict[str, Any]:
        """Extract metadata from policy URI."""
        original_uri = uri
        if uri.startswith("wandb://"):
            uri = expand_wandb_uri(uri)  # Expand wandb URI before normalization
        uri = CheckpointManager.normalize_uri(uri)
        run_name, epoch = key_and_version(uri)
        metadata = {"run_name": run_name, "epoch": epoch, "uri": uri, "original_uri": original_uri}

        if uri.startswith("file://"):
            path = Path(uri[7:])
            if path.is_file() and is_valid_checkpoint_filename(path.name):
                try:
                    run_name, epoch, agent_step, total_time, score = parse_checkpoint_filename(path.name)
                    metadata.update(
                        {
                            "run_name": run_name,
                            "epoch": epoch,
                            "agent_step": agent_step,
                            "total_time": total_time,
                            "score": score,
                        }
                    )
                except ValueError:
                    pass
        return metadata

    def _find_checkpoint_files(self, epoch: Optional[int] = None) -> List[Path]:
        """Find checkpoint files, optionally for specific epoch."""
        pattern = f"{self.run_name}__e{epoch}__s*__t*__sc*.pt" if epoch else f"{self.run_name}__e*__s*__t*__sc*.pt"
        return list(self.checkpoint_dir.glob(pattern))

    def _get_checkpoint_file(self, epoch: Optional[int] = None) -> Optional[Path]:
        """Get single checkpoint file, latest if epoch not specified."""
        files = self._find_checkpoint_files(epoch)
        if not files:
            return None
        return files[0] if epoch else max(files, key=lambda p: parse_checkpoint_filename(p.name)[1])

    def exists(self) -> bool:
        return self.checkpoint_dir.exists() and bool(self._find_checkpoint_files())

    def load_agent(self, epoch: Optional[int] = None):
        """Load agent with caching."""
        agent_file = self._get_checkpoint_file(epoch)
        if not agent_file:
            return None
        path_str = str(agent_file)

        if path_str in self._cache:
            self._cache.move_to_end(path_str)
            return self._cache[path_str]

        agent = torch.load(agent_file, weights_only=False)
        
        # Only cache if cache size > 0
        if self.cache_size > 0:
            if path_str in self._cache:
                del self._cache[path_str]
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[path_str] = agent
        return agent

    def load_trainer_state(self, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load trainer state for resuming training."""
        if epoch is None:
            checkpoint_files = self._find_checkpoint_files()
            if not checkpoint_files:
                return None
            epoch = max(parse_checkpoint_filename(f.name)[1] for f in checkpoint_files)

        trainer_file = self.checkpoint_dir / f"{self.run_name}__e{epoch}__trainer.pt"
        if not trainer_file.exists():
            return None

        state = torch.load(trainer_file, weights_only=False)
        result = {
            "optimizer_state": state.get("optimizer", state.get("optimizer_state")),
            "epoch": state.get("epoch", epoch),
            "agent_step": state.get("agent_step", 0),
        }
        if "stopwatch_state" in state:
            result["stopwatch_state"] = state["stopwatch_state"]
        return result

    def save_agent(self, agent, epoch: int, metadata: Dict[str, Any]):
        """Save agent with metadata in filename."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        agent_step = metadata.get("agent_step", 0)
        total_time = int(metadata.get("total_time", 0))
        score = int(metadata.get("score", 0.0) * 10000)
        filename = f"{self.run_name}__e{epoch}__s{agent_step}__t{total_time}__sc{score}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(agent, checkpoint_path)
        self._cache.pop(str(checkpoint_path), None)

    def save_trainer_state(
        self, optimizer, epoch: int, agent_step: int, stopwatch_state: Optional[Dict[str, Any]] = None
    ):
        """Save trainer state."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_file = self.checkpoint_dir / f"{self.run_name}__e{epoch}__trainer.pt"
        state = {"optimizer": optimizer.state_dict(), "epoch": epoch, "agent_step": agent_step}
        if stopwatch_state:
            state["stopwatch_state"] = stopwatch_state
        torch.save(state, trainer_file)

    def get_checkpoint_uri(self, epoch: Optional[int] = None) -> str:
        """Get URI for checkpoint."""
        checkpoint = self._get_checkpoint_file(epoch)
        if not checkpoint:
            msg = f"No checkpoint found for {self.run_name}" + (f" at epoch {epoch}" if epoch else "")
            raise FileNotFoundError(msg)
        return f"file://{checkpoint}"

    def select_checkpoints(self, strategy: str = "latest", count: int = 1, metric: str = "epoch") -> List[Path]:
        """Select checkpoints."""
        checkpoint_files = self._find_checkpoint_files()
        if not checkpoint_files:
            return []
        metric_idx = {"epoch": 1, "agent_step": 2, "total_time": 3, "score": 4}.get(metric, 1)
        checkpoint_files.sort(key=lambda f: parse_checkpoint_filename(f.name)[metric_idx], reverse=True)
        return checkpoint_files if strategy == "all" else checkpoint_files[:count]

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """Clean up old checkpoints."""
        agent_files = self._find_checkpoint_files()
        if len(agent_files) <= keep_last_n:
            return 0
        agent_files.sort(key=lambda p: parse_checkpoint_filename(p.name)[1])
        files_to_remove = agent_files if keep_last_n == 0 else agent_files[:-keep_last_n]
        for agent_file in files_to_remove:
            epoch = parse_checkpoint_filename(agent_file.name)[1]
            trainer_file = self.checkpoint_dir / f"{self.run_name}__e{epoch}__trainer.pt"
            trainer_file.unlink(missing_ok=True)
            agent_file.unlink()
        return len(files_to_remove)

    def upload_to_wandb(self, epoch: Optional[int] = None, wandb_run=None) -> Optional[str]:
        """Upload checkpoint to wandb."""
        if epoch is None:
            checkpoint_files = self._find_checkpoint_files()
            if not checkpoint_files:
                return None
            epoch = max(parse_checkpoint_filename(f.name)[1] for f in checkpoint_files)

        checkpoint_files = self._find_checkpoint_files(epoch)
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
