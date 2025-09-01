import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from metta.mettagrid.util.file import WandbURI, local_copy
from metta.rl.wandb import get_wandb_checkpoint_metadata, load_policy_from_wandb_uri

logger = logging.getLogger(__name__)

# Global cache for sharing checkpoints across CheckpointManager instances
_global_cache = OrderedDict()


def _parse_uri_path(uri: str, scheme: str) -> str:
    """Extract path from URI, removing the scheme prefix."""
    prefix = f"{scheme}://"
    return uri[len(prefix) :] if uri.startswith(prefix) else uri


def expand_wandb_uri(uri: str, default_project: str = "metta") -> str:
    """Expand short wandb URI formats to full format."""
    if not uri.startswith("wandb://"):
        return uri

    path = _parse_uri_path(uri, "wandb")

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
        path = Path(_parse_uri_path(uri, "file"))
        if path.suffix == ".pt" and is_valid_checkpoint_filename(path.name):
            return parse_checkpoint_filename(path.name)[:2]

        # Handle directory URIs by finding the latest checkpoint inside
        if path.is_dir():
            checkpoint_file = _find_best_checkpoint_in_dir(path)
            if checkpoint_file and is_valid_checkpoint_filename(checkpoint_file.name):
                return parse_checkpoint_filename(checkpoint_file.name)[:2]
            elif checkpoint_file:
                return checkpoint_file.stem, 0

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

    if uri.startswith("mock://"):
        return _parse_uri_path(uri, "mock"), 0

    return "unknown", 0


def is_valid_checkpoint_filename(filename: str) -> bool:
    """Check if filename matches expected checkpoint format."""
    if not filename.endswith(".pt"):
        return False
    parts = filename[:-3].split("__")
    return (
        len(parts) == 5
        and parts[1].startswith("e")
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
    return (parts[0], int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:]), int(parts[4][2:]) / 10000.0)


def _find_best_checkpoint_in_dir(directory: Path) -> Optional[Path]:
    """Find the best checkpoint file in a directory."""
    # Try direct directory first, then checkpoints subdirectory
    search_dirs = [directory]
    if directory.name != "checkpoints":
        checkpoints_subdir = directory / "checkpoints"
        if checkpoints_subdir.is_dir():
            search_dirs.append(checkpoints_subdir)

    for search_dir in search_dirs:
        checkpoint_files = list(search_dir.glob("*.pt"))
        if checkpoint_files:
            # Prefer files with valid checkpoint format, sorted by epoch
            valid_checkpoints = [
                (ckpt, parse_checkpoint_filename(ckpt.name)[1])
                for ckpt in checkpoint_files
                if is_valid_checkpoint_filename(ckpt.name)
            ]
            if valid_checkpoints:
                return max(valid_checkpoints, key=lambda x: x[1])[0]
            return checkpoint_files[0]  # Fallback to any .pt file
    return None


class CheckpointManager:
    """Checkpoint manager with filename-embedded metadata and LRU cache."""

    def __init__(self, run_name: str = "default", run_dir: str = "./train_dir", cache_size: int = 3):
        # Validate run name
        if not run_name or not run_name.strip():
            raise ValueError("Run name cannot be empty")
        if any(char in run_name for char in [" ", "/", "*", "\\", ":", "<", ">", "|", "?", '"']):
            raise ValueError(f"Run name contains invalid characters: {run_name}")

        self.run_name = run_name
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / self.run_name / "checkpoints"
        self.cache_size = cache_size
        self._cache = OrderedDict()

    @staticmethod
    def clear_cache():
        """Clear the global cache used by all CheckpointManager instances."""
        global _global_cache
        _global_cache.clear()

    @staticmethod
    def load_from_uri(uri: str, device: str | torch.device | None = None):
        """Load a policy from file://, s3://, or wandb:// URI.

        Supports loading from local files, S3 buckets, wandb artifacts, or mock URIs.
        Defaults to CPU if no device is specified. Returns None if loading fails.
        """
        # Normalize the URI first (converts plain paths to file:// URIs)
        uri = CheckpointManager.normalize_uri(uri)

        # Convert device to string for consistency
        if device is None:
            device_str = "cpu"
        elif isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = device

        if uri.startswith("file://"):
            path = Path(_parse_uri_path(uri, "file"))
            try:
                if path.is_file() and path.suffix == ".pt":
                    return torch.load(path, weights_only=False, map_location=device_str)
                if path.is_dir():
                    checkpoint_file = _find_best_checkpoint_in_dir(path)
                    return (
                        torch.load(checkpoint_file, weights_only=False, map_location=device_str)
                        if checkpoint_file
                        else None
                    )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {uri}: {e}")
                return None
            return None
        if uri.startswith("s3://"):
            try:
                with local_copy(uri) as local_path:
                    return torch.load(local_path, weights_only=False, map_location=device_str)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {uri}: {e}")
                return None
        if uri.startswith("wandb://"):
            try:
                expanded_uri = expand_wandb_uri(uri)
                return load_policy_from_wandb_uri(expanded_uri, device=device_str)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {uri}: {e}")
                return None
        if uri.startswith("mock://"):
            from metta.agent.mocks import MockAgent

            return MockAgent()
        return None

    @staticmethod
    def normalize_uri(path_or_uri: str) -> str:
        """Convert path to URI format and expand short wandb URIs."""
        if not path_or_uri.startswith(("file://", "wandb://", "s3://", "mock://")):
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
            path = Path(_parse_uri_path(uri, "file"))
            if path.is_file() and is_valid_checkpoint_filename(path.name):
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
        return metadata

    def _find_checkpoint_files(self, epoch: Optional[int] = None) -> List[Path]:
        pattern = f"{self.run_name}__e{epoch}__s*__t*__sc*.pt" if epoch else f"{self.run_name}__e*__s*__t*__sc*.pt"
        return list(self.checkpoint_dir.glob(pattern))

    def _get_checkpoint_file(self, epoch: Optional[int] = None) -> Optional[Path]:
        files = self._find_checkpoint_files(epoch)
        if not files:
            return None
        return files[0] if epoch else max(files, key=lambda p: parse_checkpoint_filename(p.name)[1])

    def exists(self) -> bool:
        return self.checkpoint_dir.exists() and bool(self._find_checkpoint_files())

    def load_agent(self, epoch: Optional[int] = None, device: Optional[torch.device] = None):
        agent_file = self._get_checkpoint_file(epoch)
        if not agent_file:
            return None
        path_str = str(agent_file)

        if path_str in self._cache:
            self._cache.move_to_end(path_str)
            return self._cache[path_str]

        # Load to specified device or CPU by default
        map_location = str(device) if device else "cpu"
        agent = torch.load(agent_file, weights_only=False, map_location=map_location)

        # Only cache if cache size > 0
        if self.cache_size > 0:
            # Remove if already exists (shouldn't happen, but be safe)
            if path_str in self._cache:
                del self._cache[path_str]
            # Evict oldest entry if at capacity
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[path_str] = agent
        return agent

    def load_trainer_state(self) -> Optional[Dict[str, Any]]:
        trainer_file = self.checkpoint_dir / "trainer_state.pt"
        if not trainer_file.exists():
            return None
        state = torch.load(trainer_file, weights_only=False)
        result = {
            "optimizer_state": state.get("optimizer", state.get("optimizer_state")),
            "epoch": state.get("epoch", 0),
            "agent_step": state.get("agent_step", 0),
        }
        if "stopwatch_state" in state:
            result["stopwatch_state"] = state["stopwatch_state"]
        return result

    def save_agent(self, agent, epoch: int, metadata: Dict[str, Any], wandb_run=None) -> Optional[str]:
        """Save agent checkpoint to disk and optionally to wandb.

        Returns:
            Wandb artifact URI if uploaded, None otherwise
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        agent_step = metadata.get("agent_step", 0)
        total_time = int(metadata.get("total_time", 0))
        score = int(metadata.get("score", 0.0) * 10000)
        filename = f"{self.run_name}__e{epoch}__s{agent_step}__t{total_time}__sc{score}.pt"
        checkpoint_path = self.checkpoint_dir / filename

        # Check if we're overwriting an existing checkpoint for this epoch
        existing_files = self._find_checkpoint_files(epoch)

        torch.save(agent, checkpoint_path)

        # Upload to wandb if run is provided
        wandb_uri = None
        if wandb_run and metadata.get("upload_to_wandb", True):
            from metta.rl.wandb import upload_checkpoint_as_artifact

            # For final checkpoint, append "_final" to distinguish it
            name = self.run_name + "_final" if metadata.get("is_final", False) else self.run_name

            wandb_metadata = {
                "run_name": self.run_name,
                "epoch": epoch,
                "agent_step": agent_step,
                "total_time": total_time,
                "score": metadata.get("score", 0.0),
            }

            wandb_uri = upload_checkpoint_as_artifact(
                checkpoint_path=str(checkpoint_path),
                artifact_name=name,
                metadata=wandb_metadata,
                wandb_run=wandb_run,
            )

        # Only invalidate cache entries if we're overwriting an existing checkpoint
        if existing_files:
            keys_to_remove = []
            for cached_path in self._cache.keys():
                if Path(cached_path).name.startswith(f"{self.run_name}__e{epoch}__"):
                    keys_to_remove.append(cached_path)
            for key in keys_to_remove:
                self._cache.pop(key, None)

        return wandb_uri

    def save_trainer_state(
        self, optimizer, epoch: int, agent_step: int, stopwatch_state: Optional[Dict[str, Any]] = None
    ):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_file = self.checkpoint_dir / "trainer_state.pt"
        state = {"optimizer": optimizer.state_dict(), "epoch": epoch, "agent_step": agent_step}
        if stopwatch_state:
            state["stopwatch_state"] = stopwatch_state
        torch.save(state, trainer_file)

    def select_checkpoints(self, strategy: str = "latest", count: int = 1, metric: str = "epoch") -> List[str]:
        """Select checkpoints and return their URIs.

        Strategy can be "latest" or "all", and metric can be "epoch", "agent_step", "total_time", or "score".
        Returns a list of file:// URIs for the selected checkpoints.
        """
        checkpoint_files = self._find_checkpoint_files()
        if not checkpoint_files:
            return []
        metric_idx = {"epoch": 1, "agent_step": 2, "total_time": 3, "score": 4}.get(metric, 1)
        checkpoint_files.sort(key=lambda f: parse_checkpoint_filename(f.name)[metric_idx], reverse=True)
        selected_files = checkpoint_files if strategy == "all" else checkpoint_files[:count]
        return [f"file://{path.resolve()}" for path in selected_files]

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        agent_files = self._find_checkpoint_files()
        if len(agent_files) <= keep_last_n:
            return 0
        agent_files.sort(key=lambda p: parse_checkpoint_filename(p.name)[1])
        files_to_remove = agent_files if keep_last_n == 0 else agent_files[:-keep_last_n]
        for agent_file in files_to_remove:
            agent_file.unlink()
        # Clean up trainer state if all checkpoints are being removed
        if keep_last_n == 0:
            trainer_file = self.checkpoint_dir / "trainer_state.pt"
            trainer_file.unlink(missing_ok=True)
        return len(files_to_remove)
