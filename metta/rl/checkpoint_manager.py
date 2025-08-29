import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from metta.mettagrid.util.file import WandbURI, local_copy
from metta.rl.wandb import get_wandb_checkpoint_metadata, load_policy_from_wandb_uri, upload_checkpoint_as_artifact

logger = logging.getLogger(__name__)


def expand_wandb_uri(uri: str, default_project: str = "metta") -> str:
    """Expand short wandb URI formats to full format.

    Supports:
    - wandb://run/<run_name> -> wandb://metta/model/<run_name>:latest
    - wandb://sweep/<sweep_name> -> wandb://metta/sweep_model/<sweep_name>:latest
    - wandb://project/artifact:version -> unchanged
    """
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
    """Check if a filename matches the checkpoint naming convention."""
    parts = filename.split(".")
    if len(parts) != 6 or parts[-1] != "pt":
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
    """Parse checkpoint metadata from filename: {run_name}.e{epoch}.s{agent_step}.t{total_time}.sc{score}.pt."""
    if not is_valid_checkpoint_filename(filename):
        raise ValueError(f"Invalid checkpoint filename format: {filename}")
    parts = filename.split(".")
    return (parts[0], int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:]), int(parts[4][2:]) / 10000.0)


class CheckpointManager:
    """Checkpoint manager with filename-embedded metadata and LRU cache."""

    def __init__(self, run_name: str, run_dir: str = "./train_dir", cache_size: int = 3):
        if not run_name or not re.match(r"^[a-zA-Z0-9._-]+$", run_name):
            raise ValueError(f"Invalid run_name: {run_name}")
        self.run_name = run_name
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / self.run_name / "checkpoints"
        self.cache_size = max(0, cache_size)
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
                        logger.warning(f"No checkpoints found in {path}")
                        return None
                    valid_checkpoints = [
                        (ckpt, parse_checkpoint_filename(ckpt.name)[1])
                        for ckpt in checkpoint_files
                        if is_valid_checkpoint_filename(ckpt.name)
                    ]
                    if not valid_checkpoints:
                        logger.info(f"No standard checkpoint files found, loading {checkpoint_files[0].name}")
                        try:
                            return torch.load(checkpoint_files[0], weights_only=False)
                        except Exception as e:
                            logger.warning(f"Failed to load fallback checkpoint {checkpoint_files[0]}: {e}")
                            return None
                    latest_checkpoint = max(valid_checkpoints, key=lambda x: x[1])[0]
                    return torch.load(latest_checkpoint, weights_only=False)
                logger.warning(f"File not found: {path}")
                return None
            if uri.startswith("s3://"):
                with local_copy(uri) as local_path:
                    return torch.load(local_path, weights_only=False)
            if uri.startswith("wandb://"):
                expanded_uri = expand_wandb_uri(uri)
                return load_policy_from_wandb_uri(expanded_uri, device="cpu")
            logger.warning(f"Unsupported URI format: {uri}. Supported: file://, s3://, wandb://")
            return None
        except Exception as e:
            logger.warning(f"Failed to load policy from {uri}: {e}")
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

    def exists(self) -> bool:
        return self.checkpoint_dir.exists() and any(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.sc*.pt"))

    def clear_cache(self):
        self._cache.clear()

    def load_agent(self, epoch: Optional[int] = None):
        """Load agent with caching."""
        pattern = f"{self.run_name}.e{epoch}.s*.t*.sc*.pt" if epoch else f"{self.run_name}.e*.s*.t*.sc*.pt"
        agent_files = list(self.checkpoint_dir.glob(pattern))
        if not agent_files:
            if epoch:
                logger.warning(f"No checkpoint found for {self.run_name} at epoch {epoch}")
            return None

        agent_file = agent_files[0] if epoch else max(agent_files, key=lambda p: parse_checkpoint_filename(p.name)[1])
        path_str = str(agent_file)

        if self.cache_size > 0 and path_str in self._cache:
            self._cache.move_to_end(path_str)
            return self._cache[path_str]

        agent = torch.load(agent_file, weights_only=False)

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
            checkpoints = self.select_checkpoints(count=1, metric="epoch")
            if not checkpoints:
                return None
            epoch = parse_checkpoint_filename(checkpoints[0].name)[1]

        trainer_file = self.checkpoint_dir / f"{self.run_name}.e{epoch}.trainer.pt"
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
        filename = f"{self.run_name}.e{epoch}.s{agent_step}.t{total_time}.sc{score}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(agent, checkpoint_path)
        if str(checkpoint_path) in self._cache:
            del self._cache[str(checkpoint_path)]

    def save_trainer_state(
        self, optimizer, epoch: int, agent_step: int, stopwatch_state: Optional[Dict[str, Any]] = None
    ):
        """Save trainer state."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_file = self.checkpoint_dir / f"{self.run_name}.e{epoch}.trainer.pt"
        state = {"optimizer": optimizer.state_dict(), "epoch": epoch, "agent_step": agent_step}
        if stopwatch_state:
            state["stopwatch_state"] = stopwatch_state
        torch.save(state, trainer_file)

    def get_checkpoint_uri(self, epoch: Optional[int] = None) -> str:
        """Get URI for checkpoint."""
        pattern = f"{self.run_name}.e{epoch}.s*.t*.sc*.pt" if epoch else f"{self.run_name}.e*.s*.t*.sc*.pt"
        files = list(self.checkpoint_dir.glob(pattern))
        if not files:
            msg = f"No checkpoint found for {self.run_name}" + (f" at epoch {epoch}" if epoch else "")
            raise FileNotFoundError(msg)
        checkpoint = files[0] if epoch else max(files, key=lambda p: parse_checkpoint_filename(p.name)[1])
        return f"file://{checkpoint}"

    def select_checkpoints(self, strategy: str = "latest", count: int = 1, metric: str = "epoch") -> List[Path]:
        """Select checkpoints."""
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.sc*.pt"))
        if not checkpoint_files:
            return []
        metric_idx = {"epoch": 1, "agent_step": 2, "total_time": 3, "score": 4}.get(metric, 1)
        checkpoint_files.sort(key=lambda f: parse_checkpoint_filename(f.name)[metric_idx], reverse=True)
        return checkpoint_files if strategy == "all" else checkpoint_files[:count]

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """Clean up old checkpoints."""
        if not self.checkpoint_dir.exists():
            return 0
        agent_files = list(self.checkpoint_dir.glob(f"{self.run_name}.e*.s*.t*.sc*.pt"))
        if len(agent_files) <= keep_last_n:
            return 0
        agent_files.sort(key=lambda p: parse_checkpoint_filename(p.name)[1])
        files_to_remove = agent_files[:-keep_last_n]
        for agent_file in files_to_remove:
            epoch = parse_checkpoint_filename(agent_file.name)[1]
            trainer_file = self.checkpoint_dir / f"{self.run_name}.e{epoch}.trainer.pt"
            trainer_file.unlink(missing_ok=True)
            agent_file.unlink()
        return len(files_to_remove)

    def upload_to_wandb(self, epoch: Optional[int] = None, wandb_run=None) -> Optional[str]:
        """Upload checkpoint to wandb."""
        if epoch is None:
            checkpoints = self.select_checkpoints(count=1, metric="epoch")
            if not checkpoints:
                return None
            epoch = parse_checkpoint_filename(checkpoints[0].name)[1]

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
