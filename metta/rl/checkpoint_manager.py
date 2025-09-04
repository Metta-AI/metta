import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import torch

from metta.agent.mocks import MockAgent
from metta.mettagrid.util.file import WandbURI, local_copy
from metta.rl.wandb import (
    expand_wandb_uri,
    get_wandb_checkpoint_metadata,
    load_policy_from_wandb_uri,
    upload_checkpoint_as_artifact,
)

logger = logging.getLogger(__name__)


class PolicyMetadata(TypedDict, total=False):
    """Type definition for policy metadata returned by get_policy_metadata."""

    run_name: str
    epoch: int
    uri: str
    original_uri: str
    # Optional fields (only present for valid checkpoint files)
    agent_step: int
    total_time: int
    score: float


def _parse_uri_path(uri: str, scheme: str) -> str:
    """Extract path from URI, removing the scheme prefix.
    "file:///tmp/model.pt" -> "/tmp/model.pt"
    "wandb://project/artifact:v1" -> "project/artifact:v1"
    """
    prefix = f"{scheme}://"
    return uri[len(prefix) :] if uri.startswith(prefix) else uri


def key_and_version(uri: str) -> tuple[str, int]:
    """Extract key (run name) and version (epoch) from a policy URI.
    "file:///tmp/my_run__e5__s100__t60__sc5000.pt" -> ("my_run", 5)
    "s3://bucket/my_run__e10__s200__t120__sc8000.pt" -> ("my_run", 10)
    "mock://test_agent" -> ("test_agent", 0)
    """
    if uri.startswith("file://"):
        path = Path(_parse_uri_path(uri, "file"))
        if path.suffix == ".pt" and is_valid_checkpoint_filename(path.name):
            return parse_checkpoint_filename(path.name)[:2]

        # Handle directory URIs by finding the latest checkpoint inside
        if path.is_dir():
            checkpoint_file = _find_latest_checkpoint_in_dir(path)
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
        # Fallback: parse artifact name from URI
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


def _find_latest_checkpoint_in_dir(directory: Path) -> Optional[Path]:
    """Find the latest checkpoint file in a directory (by epoch)."""
    # Try direct directory first, then checkpoints subdirectory
    search_dirs = [directory]
    if directory.name != "checkpoints":
        checkpoints_subdir = directory / "checkpoints"
        if checkpoints_subdir.is_dir():
            search_dirs.append(checkpoints_subdir)

    for search_dir in search_dirs:
        checkpoint_files = list(search_dir.glob("*.pt"))
        if checkpoint_files:
            # Only return files with valid checkpoint format, sorted by epoch
            valid_checkpoints = [
                (ckpt, parse_checkpoint_filename(ckpt.name)[1])
                for ckpt in checkpoint_files
                if is_valid_checkpoint_filename(ckpt.name)
            ]
            if valid_checkpoints:
                return max(valid_checkpoints, key=lambda x: x[1])[0]
    return None


class CheckpointManager:
    """Checkpoint manager with filename-embedded metadata and LRU cache."""

    def __init__(self, run: str = "default", run_dir: str = "./train_dir", cache_size: int = 3):
        # Validate run name
        if not run or not run.strip():
            raise ValueError("Run name cannot be empty")
        if any(char in run for char in [" ", "/", "*", "\\", ":", "<", ">", "|", "?", '"']):
            raise ValueError(f"Run name contains invalid characters: {run}")
        if "__" in run:
            raise ValueError(f"Run name cannot contain '__' as it's used as a delimiter in checkpoint filenames: {run}")

        self.run = run
        self.run_name = run
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / self.run / "checkpoints"
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def clear_cache(self):
        """Clear the instance's LRU cache."""
        self._cache.clear()

    @staticmethod
    def load_from_uri(uri: str, device: str | torch.device = "cpu"):
        """Load a policy from a URI (file://, wandb://, s3://, or mock://)."""
        if uri.startswith("file://"):
            path = Path(_parse_uri_path(uri, "file"))
            if path.is_dir():
                # Find latest checkpoint in directory
                checkpoint_file = _find_latest_checkpoint_in_dir(path)
                if not checkpoint_file:
                    raise FileNotFoundError(f"No checkpoint files in {uri}")
                return torch.load(checkpoint_file, weights_only=False, map_location=device)
            # Load specific file
            return torch.load(path, weights_only=False, map_location=device)

        if uri.startswith("s3://"):
            with local_copy(uri) as local_path:
                return torch.load(local_path, weights_only=False, map_location=device)

        if uri.startswith("wandb://"):
            return load_policy_from_wandb_uri(uri, device=device)

        if uri.startswith("mock://"):
            return MockAgent()

        raise ValueError(f"Invalid URI: {uri}")

    @staticmethod
    def normalize_uri(uri: str) -> str:
        """Convert paths to file:// URIs. Keep other URI schemes as-is."""
        if uri.startswith(("file://", "s3://", "mock://", "wandb://")):
            return uri
        # Assume it's a file path - convert to URI
        return f"file://{Path(uri).resolve()}"

    @staticmethod
    def get_policy_metadata(uri: str) -> PolicyMetadata:
        """Extract metadata from policy URI."""
        normalized_uri = CheckpointManager.normalize_uri(uri)
        run_name, epoch = key_and_version(normalized_uri)  # Use normalized URI for metadata extraction
        metadata: PolicyMetadata = {"run_name": run_name, "epoch": epoch, "uri": normalized_uri, "original_uri": uri}

        # Add extra metadata for file:// URIs with valid checkpoint filenames
        if normalized_uri.startswith("file://"):
            path = Path(_parse_uri_path(normalized_uri, "file"))
            if path.is_file() and is_valid_checkpoint_filename(path.name):
                _, _, agent_step, total_time, score = parse_checkpoint_filename(path.name)
                metadata["agent_step"] = agent_step
                metadata["total_time"] = total_time
                metadata["score"] = score
        return metadata

    def _find_checkpoint_files(self, epoch: Optional[int] = None) -> List[Path]:
        pattern = f"{self.run_name}__e{epoch}__s*__t*__sc*.pt" if epoch else f"{self.run_name}__e*__s*__t*__sc*.pt"
        return list(self.checkpoint_dir.glob(pattern))

    def load_agent(self, epoch: Optional[int] = None, device: Optional[torch.device] = None):
        """Load agent checkpoint from local directory with LRU caching."""
        files = self._find_checkpoint_files(epoch)
        if not files:
            raise FileNotFoundError(f"No checkpoints found for {self.run_name} epoch={epoch}")

        # Select file: first if epoch specified, latest otherwise
        agent_file = files[0] if epoch else max(files, key=lambda p: parse_checkpoint_filename(p.name)[1])
        cache_key = str(agent_file)

        # Check cache
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Load from disk
        file_uri = f"file://{agent_file.resolve()}"
        agent = self.load_from_uri(file_uri, device=device or "cpu")

        # Update cache
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)  # Evict oldest
            self._cache[cache_key] = agent

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

    def save_agent(self, agent, epoch: int, metadata: Dict[str, Any], wandb_run=None) -> str:
        """Save agent checkpoint to disk and optionally to wandb.
        Returns URI of saved checkpoint (file:// for local, wandb:// if uploaded to wandb).
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

        # Return wandb URI if uploaded, otherwise return local file URI
        if wandb_uri:
            return wandb_uri
        else:
            return f"file://{checkpoint_path.resolve()}"

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
