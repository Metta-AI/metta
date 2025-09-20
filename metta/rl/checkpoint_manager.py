import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import torch

from metta.agent.mocks import MockAgent
from mettagrid.util.file import local_copy, write_file
from mettagrid.util.uri import ParsedURI

logger = logging.getLogger(__name__)


def _is_state_dict(loaded_obj: Any) -> bool:
    """
    Check if the loaded object is a state_dict (PufferLib format) rather than a full agent.
    A state_dict is typically a dict with tensor values and parameter names as keys.
    """
    if not isinstance(loaded_obj, dict):
        return False

    if not loaded_obj:
        return False

    # State dicts typically have keys ending with .weight, .bias, etc.
    # and values that are tensors
    sample_items = list(loaded_obj.items())[:5]  # Check first 5 items
    for key, value in sample_items:
        if isinstance(key, str) and torch.is_tensor(value):
            # This looks like a parameter name -> tensor mapping
            continue
        else:
            return False
    return True


def _create_agent_from_state_dict(state_dict: Dict[str, torch.Tensor], device: str | torch.device = "cpu"):
    """
    Create a MettaAgent from a state_dict (PufferLib checkpoint format).
    This requires creating the agent architecture and loading the weights.
    """
    from metta.agent.agent_config import AgentConfig
    from metta.agent.metta_agent import MettaAgent
    from metta.mettagrid.builder.envs import make_arena
    from metta.mettagrid.mettagrid_env import MettaGridEnv
    from metta.rl.system_config import SystemConfig

    logger.info("Loading PufferLib checkpoint format (state_dict) - creating MettaAgent")

    # Create a minimal environment for agent initialization

    env_cfg = make_arena(num_agents=60)
    temp_env = MettaGridEnv(env_cfg, render_mode="rgb_array")

    system_cfg = SystemConfig(device=str(device))
    agent_cfg = AgentConfig(name="pytorch/fast")

    # Create the agent
    agent = MettaAgent(temp_env, system_cfg, agent_cfg)

    temp_env.close()
    # Try to load only compatible parameters
    agent_state = agent.policy.state_dict()
    compatible_state = {}

    new_state_dict = {}

    for state_dict_key, state_dict_value in state_dict.items():
        if state_dict_key.startswith("fast_policy."):
            new_key = state_dict_key[len("fast_policy.") :]
            new_state_dict[new_key] = state_dict_value
        else:
            new_state_dict[state_dict_key] = state_dict_value

    keys_matched = 0
    for key, value in new_state_dict.items():
        if key in agent_state:
            agent_state[key] = value
            keys_matched += 1
        else:
            logger.info(f"Skipping incompatible parameter: {key}")

    logger.info(f"Loaded {keys_matched}/{len(new_state_dict)} compatible parameters")

    if compatible_state:
        agent.load_state_dict(compatible_state, strict=True)
        logger.info(f"Loaded {len(compatible_state)}/{len(state_dict)} compatible parameters")
    else:
        logger.warning("No compatible parameters found - proceeding with random initialization")

    return agent


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
    "file:///tmp/my_run/checkpoints/my_run:v5.pt" -> ("my_run", 5)
    "s3://bucket/policies/my_run/checkpoints/my_run:v10.pt" -> ("my_run", 10)
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
                return _extract_run_and_epoch(Path(key_path.name))
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


def _load_checkpoint_file(path: str, device: str | torch.device):
    """Load a checkpoint file, raising FileNotFoundError on corruption."""
    try:
        return torch.load(path, weights_only=False, map_location=device)
    except FileNotFoundError:
        raise
    except (pickle.UnpicklingError, RuntimeError, OSError) as err:
        raise FileNotFoundError(f"Invalid or corrupted checkpoint file: {path}") from err


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
        """Load a policy from a URI (file://, wandb://, s3://, or mock://).
        Supports both Metta format (full agent) and PufferLib format (state_dict)."""

        def _load_and_handle_format(checkpoint_path_or_obj):
            """Helper to load checkpoint and handle both formats."""
            if isinstance(checkpoint_path_or_obj, (str, Path)):
                loaded_obj = torch.load(checkpoint_path_or_obj, weights_only=False, map_location=device)
            else:
                loaded_obj = checkpoint_path_or_obj

            # Check if this is a PufferLib checkpoint (state_dict format)
            if _is_state_dict(loaded_obj):
                logger.info(f"Detected PufferLib checkpoint format in {uri}")
                return _create_agent_from_state_dict(loaded_obj, device)
            else:
                # Standard Metta format (full agent object)
                return loaded_obj

        if uri.startswith("file://"):
            path = Path(_parse_uri_path(uri, "file"))
            if path.is_dir():
                checkpoint_file = _find_latest_checkpoint_in_dir(path)
                if not checkpoint_file:
                    raise FileNotFoundError(f"No checkpoint files in {uri}")
                return _load_and_handle_format(checkpoint_file)
            # Load specific file
            return _load_and_handle_format(path)

        if uri.startswith("s3://"):
            with local_copy(uri) as local_path:
                return _load_and_handle_format(local_path)

        if uri.startswith("wandb://"):
            # For wandb, we need to handle the result after loading
            loaded_obj = load_policy_from_wandb_uri(uri, device=device)
            if _is_state_dict(loaded_obj):
                logger.info(f"Detected PufferLib checkpoint format in wandb URI {uri}")
                return _create_agent_from_state_dict(loaded_obj, device)
            return loaded_obj

        if uri.startswith("mock://"):
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
