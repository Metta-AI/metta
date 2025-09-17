import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import torch

from metta.agent.mocks import MockAgent
from metta.mettagrid.util.file import local_copy, write_file
from metta.mettagrid.util.uri import ParsedURI

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


def key_and_version(uri: str) -> tuple[str, int]:
    """Extract key (run name) and version (epoch) from a policy URI.
    "file:///tmp/my_run__e5__s100__t60__sc5000.pt" -> ("my_run", 5)
    "s3://bucket/my_run__e10__s200__t120__sc8000.pt" -> ("my_run", 10)
    "mock://test_agent" -> ("test_agent", 0)
    """
    parsed = ParsedURI.parse(uri)

    if parsed.scheme == "file" and parsed.local_path is not None:
        path = parsed.local_path

        if path.suffix == ".pt":
            return _extract_run_and_epoch(path)

        if path.is_dir():
            checkpoint_file = _find_latest_checkpoint_in_dir(path)
            if checkpoint_file:
                return _extract_run_and_epoch(checkpoint_file)
        return (path.stem if path.suffix else path.name, 0)

    if parsed.scheme == "s3" and parsed.key:
        key_path = Path(parsed.key)
        if key_path.suffix == ".pt":
            try:
                return _extract_run_and_epoch(key_path)
            except ValueError:
                pass
        return (key_path.stem if key_path.suffix else key_path.name, 0)

    if parsed.scheme == "mock":
        return (parsed.path or "mock"), 0

    return "unknown", 0


def _extract_run_and_epoch(path: Path) -> tuple[str, int]:
    """Infer run name and epoch from a checkpoint path."""

    stem = path.stem
    # Determine run name based on directory structure (…/<run>/checkpoints/v{epoch}.pt)
    if path.parent.name == "checkpoints":
        run_name = path.parent.parent.name
    else:
        run_name = path.parent.name

    if stem.startswith("v") and stem[1:].isdigit():
        return run_name, int(stem[1:])

    # Legacy fallback: run__e{epoch}[__…]
    parts = stem.split("__")
    if len(parts) >= 2 and parts[1].startswith("e") and parts[1][1:].isdigit():
        return parts[0], int(parts[1][1:])

    raise ValueError(f"Cannot determine run/epoch from checkpoint path: {path}")


def _find_latest_checkpoint_in_dir(directory: Path) -> Optional[Path]:
    """Find the latest checkpoint file in a directory (by epoch)."""
    # Try direct directory first, then checkpoints subdirectory
    search_dirs = [directory]
    if directory.name != "checkpoints":
        checkpoints_subdir = directory / "checkpoints"
        if checkpoints_subdir.is_dir():
            search_dirs.append(checkpoints_subdir)

    for search_dir in search_dirs:
        checkpoint_files = [ckpt for ckpt in search_dir.glob("*.pt") if ckpt.stem]
        if checkpoint_files:
            try:
                return max(checkpoint_files, key=lambda p: _extract_run_and_epoch(p)[1])
            except ValueError:
                continue
    return None


class CheckpointManager:
    """Checkpoint manager with filename-embedded metadata and LRU cache."""

    def __init__(
        self,
        run: str = "default",
        run_dir: str = "./train_dir",
        cache_size: int = 3,
        remote_prefix: str | None = None,
    ):
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
        self._remote_prefix = None
        if remote_prefix:
            parsed = ParsedURI.parse(remote_prefix)
            if parsed.scheme != "s3" or not parsed.bucket or not parsed.key:
                raise ValueError("remote_prefix must be an s3:// URI with bucket and key prefix")
            # Remove trailing slash from prefix for deterministic joins
            key_prefix = parsed.key.rstrip("/")
            self._remote_prefix = f"s3://{parsed.bucket}/{key_prefix}" if key_prefix else f"s3://{parsed.bucket}"

    def clear_cache(self):
        """Clear the instance's LRU cache."""
        self._cache.clear()

    @staticmethod
    def load_from_uri(uri: str, device: str | torch.device = "cpu"):
        """Load a policy from a URI (file://, s3://, or mock://)."""
        if uri.startswith(("http://", "https://", "ftp://", "gs://")):
            raise ValueError(f"Invalid URI: {uri}")
        parsed = ParsedURI.parse(uri)

        if parsed.scheme == "file" and parsed.local_path is not None:
            path = parsed.local_path
            if path.is_dir():
                checkpoint_file = _find_latest_checkpoint_in_dir(path)
                if not checkpoint_file:
                    raise FileNotFoundError(f"No checkpoint files in {uri}")
                return torch.load(checkpoint_file, weights_only=False, map_location=device)
            return torch.load(path, weights_only=False, map_location=device)

        if parsed.scheme == "s3":
            with local_copy(parsed.canonical) as local_path:
                return torch.load(local_path, weights_only=False, map_location=device)

        if parsed.scheme == "mock":
            return MockAgent()

        raise ValueError(f"Invalid URI: {uri}")

    @staticmethod
    def normalize_uri(uri: str) -> str:
        """Convert paths to file:// URIs. Keep other URI schemes as-is."""
        parsed = ParsedURI.parse(uri)
        return parsed.canonical

    @staticmethod
    def get_policy_metadata(uri: str) -> PolicyMetadata:
        """Extract metadata from policy URI."""
        normalized_uri = CheckpointManager.normalize_uri(uri)
        run_name, epoch = key_and_version(normalized_uri)
        return {
            "run_name": run_name,
            "epoch": epoch,
            "uri": normalized_uri,
            "original_uri": uri,
        }

    def _find_checkpoint_files(self, epoch: Optional[int] = None) -> List[Path]:
        pattern = f"v{epoch}.pt" if epoch is not None else "v*.pt"
        return list(self.checkpoint_dir.glob(pattern))

    def load_agent(self, epoch: Optional[int] = None, device: Optional[torch.device] = None):
        """Load agent checkpoint from local directory with LRU caching."""
        files = self._find_checkpoint_files(epoch)
        if not files:
            raise FileNotFoundError(f"No checkpoints found for {self.run_name} epoch={epoch}")

        # Select file: first if epoch specified, latest otherwise
        agent_file = files[0] if epoch else max(files, key=lambda p: _extract_run_and_epoch(p)[1])
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

    def save_agent(self, agent, epoch: int, metadata: Dict[str, Any]) -> str:
        """Save agent checkpoint to disk and upload to remote storage if configured.

        Returns URI of saved checkpoint (s3:// if remote prefix configured, otherwise file://).
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.run_name}__e{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / filename

        # Check if we're overwriting an existing checkpoint for this epoch
        existing_files = self._find_checkpoint_files(epoch)

        torch.save(agent, checkpoint_path)

        remote_uri = None
        if self._remote_prefix:
            remote_uri = f"{self._remote_prefix}/{filename}"
            write_file(remote_uri, str(checkpoint_path))

        # Only invalidate cache entries if we're overwriting an existing checkpoint
        if existing_files:
            keys_to_remove = []
            for cached_path in self._cache.keys():
                if Path(cached_path).name.startswith(f"{self.run_name}__e{epoch}"):
                    keys_to_remove.append(cached_path)
            for key in keys_to_remove:
                self._cache.pop(key, None)

        if remote_uri:
            return remote_uri
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

    def select_checkpoints(self, strategy: str = "latest", count: int = 1) -> List[str]:
        """Select checkpoints and return their URIs.

        Strategy can be "latest" or "all". Checkpoints are ordered purely by epoch.
        """
        checkpoint_files = self._find_checkpoint_files()
        if not checkpoint_files:
            return []
        checkpoint_files.sort(key=lambda f: _extract_run_and_epoch(f)[1], reverse=True)
        selected_files = checkpoint_files if strategy == "all" else checkpoint_files[:count]
        if self._remote_prefix:
            return [f"{self._remote_prefix}/{path.name}" for path in selected_files]
        return [f"file://{path.resolve()}" for path in selected_files]

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        agent_files = self._find_checkpoint_files()
        if len(agent_files) <= keep_last_n:
            return 0
        agent_files.sort(key=lambda p: _extract_run_and_epoch(p)[1])
        files_to_remove = agent_files if keep_last_n == 0 else agent_files[:-keep_last_n]
        for agent_file in files_to_remove:
            agent_file.unlink()
        # Clean up trainer state if all checkpoints are being removed
        if keep_last_n == 0:
            trainer_file = self.checkpoint_dir / "trainer_state.pt"
            trainer_file.unlink(missing_ok=True)
        return len(files_to_remove)
