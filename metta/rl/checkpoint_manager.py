import logging
import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import torch

from metta.agent.mocks import MockAgent
from metta.agent.policy import Policy
from metta.rl.puffer_policy import _is_puffer_state_dict, load_pufferlib_checkpoint
from metta.rl.system_config import SystemConfig
from metta.tools.utils.auto_config import auto_policy_storage_decision
from metta.utils.file import local_copy, write_file
from metta.utils.uri import ParsedURI

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

    Examples:
        "file:///tmp/my_run/checkpoints/my_run:v5.pt" -> ("my_run", 5)
        "s3://bucket/policies/my_run/checkpoints/my_run:v10.pt" -> ("my_run", 10)
        "file:///tmp/my_run/checkpoints/my_run:latest.pt" -> ("my_run", latest_epoch)
        "s3://bucket/policies/my_run/checkpoints/my_run:latest.pt" -> ("my_run", latest_epoch)
        "mock://test_agent" -> ("test_agent", 0)

    The :latest selector automatically resolves to the highest epoch number
    available in the checkpoint directory.
    """
    parsed = ParsedURI.parse(uri)

    if parsed.scheme == "file" and parsed.local_path is not None:
        path = parsed.local_path

        if path.suffix == ".pt":
            run_name, epoch = _extract_run_and_epoch(path)
            if epoch == -1:  # :latest selector
                epoch = _resolve_latest_epoch(path, run_name)
            return (run_name, epoch)

        if path.is_dir():
            checkpoint_file = _find_latest_checkpoint_in_dir(path)
            if checkpoint_file:
                return _extract_run_and_epoch(checkpoint_file)
        return (path.stem if path.suffix else path.name, 0)

    if parsed.scheme == "s3" and parsed.key:
        key_path = Path(parsed.key)
        if key_path.suffix == ".pt":
            try:
                run_name, epoch = _extract_run_and_epoch(Path(key_path.name))
                if epoch == -1:  # :latest selector
                    epoch = _resolve_latest_epoch_s3(uri, run_name)
                return (run_name, epoch)
            except ValueError:
                pass
        return (key_path.stem if key_path.suffix else key_path.name, 0)

    if parsed.scheme == "mock":
        return (parsed.path or "mock"), 0

    return "unknown", 0


def _extract_run_and_epoch(path: Path) -> tuple[str, int]:
    """Infer run name and epoch from a checkpoint path.

    The parser is intentionally permissive: it understands the new
    ``<run_name>:v{epoch}.pt`` format while falling back to directory
    structure, leading ``v{epoch}.pt}``, or legacy ``run__e{epoch}``
    filenames. Unexpected filenames return epoch ``0`` with a best-effort
    run name instead of failing.

    Special handling for :latest selector - returns epoch -1 to indicate
    latest epoch resolution is needed.
    """

    stem = path.stem

    # Prefer run from filename (<run>:v{epoch}.pt) when present.
    run_name: str | None = None
    epoch = 0

    if ":v" in stem:
        candidate_run, suffix = stem.rsplit(":v", 1)
        if candidate_run:
            run_name = candidate_run
        if suffix.isdigit():
            epoch = int(suffix)
    elif ":latest" in stem:
        candidate_run = stem.replace(":latest", "")
        if candidate_run:
            run_name = candidate_run
            epoch = -1  # Special marker for latest resolution

    # Fall back to directory structure (…/<run>/checkpoints/<file>.pt)
    if run_name is None:
        if path.parent.name == "checkpoints" and path.parent.parent.name:
            run_name = path.parent.parent.name
        elif path.parent.name not in {"", "."}:
            run_name = path.parent.name
        else:
            run_name = stem

    # Handle filenames like v{epoch}.pt where run name comes from directories
    if epoch == 0 and stem.startswith("v") and stem[1:].isdigit():
        epoch = int(stem[1:])

    # Legacy ``run__e{epoch}`` filenames
    if epoch == 0:
        parts = stem.split("__")
        if len(parts) >= 2 and parts[1].startswith("e") and parts[1][1:].isdigit():
            run_name = parts[0]
            epoch = int(parts[1][1:])

    # Last resort: try to parse trailing digits after 'v'
    if epoch == 0 and "v" in stem:
        trailing = stem.rsplit("v", 1)[-1]
        if trailing.isdigit():
            epoch = int(trailing)

    return run_name or "unknown", epoch


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


def _resolve_latest_epoch(path: Path, run_name: str) -> int:
    """Resolve :latest to actual epoch number by finding latest checkpoint."""
    checkpoint_dir = path.parent
    if checkpoint_dir.name != "checkpoints":
        # Try to find checkpoints directory
        potential_checkpoints = checkpoint_dir / "checkpoints"
        if potential_checkpoints.is_dir():
            checkpoint_dir = potential_checkpoints

    latest_checkpoint = _find_latest_checkpoint_in_dir(checkpoint_dir)
    if latest_checkpoint:
        _, epoch = _extract_run_and_epoch(latest_checkpoint)
        return epoch
    return 0


def _resolve_latest_epoch_s3(uri: str, run_name: str) -> int:
    """Resolve :latest for S3 URIs by listing available checkpoints."""
    try:
        import boto3

        from metta.utils.uri import ParsedURI

        # Extract base path without the filename
        base_uri = uri.rsplit("/", 1)[0] + "/"
        parsed = ParsedURI.parse(base_uri)

        if parsed.scheme != "s3" or not parsed.bucket:
            return 0

        # List objects in S3
        s3_client = boto3.client("s3")
        prefix = parsed.key or ""

        response = s3_client.list_objects_v2(Bucket=parsed.bucket, Prefix=prefix)

        if "Contents" not in response:
            return 0

        # Filter for checkpoint files
        checkpoint_files = []
        for obj in response["Contents"]:
            key = obj["Key"]
            filename = key.split("/")[-1]  # Get just the filename
            if filename.endswith(".pt") and not filename.endswith("trainer_state.pt"):
                checkpoint_files.append(filename)

        if not checkpoint_files:
            return 0

        # Extract epochs from filenames and find maximum
        max_epoch = 0
        for filename in checkpoint_files:
            try:
                _, epoch = _extract_run_and_epoch(Path(filename))
                if epoch > max_epoch:
                    max_epoch = epoch
            except ValueError:
                continue

        return max_epoch
    except Exception as e:
        logger.warning(f"Failed to resolve :latest for S3 URI {uri}: {e}, defaulting to epoch 0")
        return 0


def _load_checkpoint_file(path: str, device: str | torch.device) -> Policy:
    """Load a checkpoint file, raising FileNotFoundError on corruption."""
    try:
        checkpoint_data = torch.load(path, weights_only=False, map_location=device)

        if _is_puffer_state_dict(checkpoint_data):
            return load_pufferlib_checkpoint(checkpoint_data, device)

        return checkpoint_data

    except FileNotFoundError:
        raise
    except (pickle.UnpicklingError, RuntimeError, OSError) as err:
        raise FileNotFoundError(f"Invalid or corrupted checkpoint file: {path}") from err


class CheckpointManager:
    """Checkpoint manager with filename-embedded metadata and LRU cache."""

    def __init__(
        self,
        run: str,
        system_cfg: SystemConfig,
        cache_size: int = 3,
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
        self.run_dir = system_cfg.data_dir / self.run
        self.checkpoint_dir = self.run_dir / "checkpoints"

        os.makedirs(system_cfg.data_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.cache_size = cache_size
        self._cache = OrderedDict()

        self._remote_prefix = None
        if not system_cfg.local_only:
            if system_cfg.remote_prefix:
                parsed = ParsedURI.parse(system_cfg.remote_prefix)
                if parsed.scheme != "s3" or not parsed.bucket or not parsed.key:
                    raise ValueError("remote_prefix must be an s3:// URI with bucket and key prefix")
                # Remove trailing slash from prefix for deterministic joins
                key_prefix = parsed.key.rstrip("/")
                self._remote_prefix = f"s3://{parsed.bucket}/{key_prefix}" if key_prefix else f"s3://{parsed.bucket}"

            if self._remote_prefix is None:
                self._setup_remote_prefix()

    def _setup_remote_prefix(self) -> None:
        """Determine and set the remote prefix for policy storage if needed."""
        if self._remote_prefix is None:
            storage_decision = auto_policy_storage_decision(self.run)
            if storage_decision.remote_prefix:
                self._remote_prefix = storage_decision.remote_prefix
                if storage_decision.reason == "env_override":
                    logger.info("Using POLICY_REMOTE_PREFIX for policy storage: %s", storage_decision.remote_prefix)
                else:
                    logger.info(
                        "Policies will sync to %s (Softmax AWS profile detected).",
                        storage_decision.remote_prefix,
                    )
            elif storage_decision.reason == "not_connected":
                logger.info(
                    "Softmax AWS SSO not detected; policies will remain local. "
                    "Run 'aws sso login --profile softmax' then 'metta status --components=aws' to enable uploads."
                )
            elif storage_decision.reason == "aws_not_enabled":
                logger.info(
                    "AWS component disabled; policies will remain local. Run 'metta configure aws' to set up S3."
                )
            elif storage_decision.reason == "no_base_prefix":
                logger.info(
                    "Remote policy prefix unset; policies will remain local. Configure POLICY_REMOTE_PREFIX or run "
                    "'metta configure aws'."
                )

    @property
    def remote_checkpoints_enabled(self) -> bool:
        return self._remote_prefix is not None

    def clear_cache(self):
        """Clear the instance's LRU cache."""
        self._cache.clear()

    @staticmethod
    def load_from_uri(uri: str, device: str | torch.device = "cpu") -> Policy:
        """Load a policy from a URI (file://, s3://, or mock://).

        Supports :latest selector for automatic resolution to the most recent checkpoint:
            file:///path/to/run/checkpoints/run_name:latest.pt
            s3://bucket/path/run/checkpoints/run_name:latest.pt
        """
        if uri.startswith(("http://", "https://", "ftp://", "gs://")):
            raise ValueError(f"Invalid URI: {uri}")

        # Resolve :latest selector before proceeding
        uri = CheckpointManager._resolve_latest_uri(uri)
        parsed = ParsedURI.parse(uri)

        if parsed.scheme == "file" and parsed.local_path is not None:
            path = parsed.local_path
            if path.is_dir():
                checkpoint_file = _find_latest_checkpoint_in_dir(path)
                if not checkpoint_file:
                    raise FileNotFoundError(f"No checkpoint files in {uri}")
                return _load_checkpoint_file(str(checkpoint_file), device)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            return _load_checkpoint_file(str(path), device)

        if parsed.scheme == "s3":
            with local_copy(parsed.canonical) as local_path:
                return _load_checkpoint_file(str(local_path), device)

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
        def matches_epoch(path: Path) -> bool:
            if epoch is None:
                return True
            stem = path.stem
            if stem.endswith(f":v{epoch}"):
                return True
            _, version = _extract_run_and_epoch(path)
            return version == epoch

        candidates = [
            path for path in self.checkpoint_dir.glob("*.pt") if path.name != "trainer_state.pt" and matches_epoch(path)
        ]

        candidates.sort(
            key=lambda p: (_extract_run_and_epoch(p)[1], p.stat().st_mtime),
            reverse=True,
        )
        return candidates

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
        if "curriculum_state" in state:
            result["curriculum_state"] = state["curriculum_state"]
        if "loss_states" in state:
            result["loss_states"] = state["loss_states"]
        return result

    def save_agent(self, agent, epoch: int, metadata: Dict[str, Any]) -> str:
        """Save agent checkpoint to disk and upload to remote storage if configured.

        Returns URI of saved checkpoint (s3:// if remote prefix configured, otherwise file://).
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.run_name}:v{epoch}.pt"
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
                if Path(cached_path).name.startswith(f"{self.run_name}:v{epoch}"):
                    keys_to_remove.append(cached_path)
            for key in keys_to_remove:
                self._cache.pop(key, None)

        if remote_uri:
            return remote_uri
        return f"file://{checkpoint_path.resolve()}"

    def save_trainer_state(
        self,
        optimizer,
        epoch: int,
        agent_step: int,
        stopwatch_state: Optional[Dict[str, Any]] = None,
        curriculum_state: Optional[Dict[str, Any]] = None,
        loss_states: Optional[Dict[str, Any]] = None,
    ):
        from metta.rl.training.optimizer import is_schedulefree_optimizer

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_file = self.checkpoint_dir / "trainer_state.pt"

        # For ScheduleFree optimizers, ensure we're in eval mode to get correct state
        if is_schedulefree_optimizer(optimizer):
            optimizer.eval()

        state = {"optimizer": optimizer.state_dict(), "epoch": epoch, "agent_step": agent_step}
        if stopwatch_state:
            state["stopwatch_state"] = stopwatch_state
        if curriculum_state:
            state["curriculum_state"] = curriculum_state
        if loss_states is not None:
            state["loss_states"] = loss_states
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

    @staticmethod
    def _resolve_latest_uri(uri: str) -> str:
        """Resolve :latest in URI to actual epoch number."""
        if ":latest" not in uri:
            return uri

        try:
            # Parse the URI to handle it properly
            parsed = ParsedURI.parse(uri)

            if parsed.scheme == "file" and parsed.local_path:
                # For file URIs, find the latest checkpoint in the directory
                checkpoint_dir = parsed.local_path.parent
                latest_checkpoint = _find_latest_checkpoint_in_dir(checkpoint_dir)
                if latest_checkpoint:
                    return f"file://{latest_checkpoint.resolve()}"
                else:
                    logger.warning(f"No checkpoints found in directory for :latest URI {uri}")
                    return uri
            elif parsed.scheme == "s3":
                # For S3 URIs, extract run name and resolve latest epoch
                run_name = parsed.local_path.stem.replace(":latest", "") if parsed.local_path else "unknown"
                actual_epoch = _resolve_latest_epoch_s3(uri, run_name)
                return uri.replace(":latest", f":v{actual_epoch}")
            else:
                logger.warning(f"Unsupported scheme for :latest resolution: {parsed.scheme}")
                return uri
        except Exception as e:
            logger.warning(f"Failed to resolve :latest in URI {uri}: {e}")

        return uri
