"""Simple URI resolution for policy loading."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from metta.rl.checkpoint_manager import CheckpointManager, parse_checkpoint_filename
from metta.rl.wandb_policy_loader import get_wandb_artifact_metadata, load_policy_from_wandb_uri


class PolicyUriResolver:
    """URI dispatcher for file:// and wandb:// schemes."""

    def __init__(self):
        self._checkpoint_managers = {}

    def resolve_policy(self, uri: str, device: str = "cpu") -> torch.nn.Module:
        """Load a policy from file:// or wandb:// URI."""
        if uri.startswith("file://"):
            return self._resolve_file_policy(uri, device)
        elif uri.startswith("wandb://"):
            return load_policy_from_wandb_uri(uri, device)
        else:
            raise ValueError(f"Unsupported URI scheme: {uri}")

    def get_policy_metadata(self, uri: str) -> Dict[str, Any]:
        """Get metadata from a policy URI without loading the full policy."""
        if uri.startswith("file://"):
            return self._get_file_metadata(uri)
        elif uri.startswith("wandb://"):
            return get_wandb_artifact_metadata(uri)
        else:
            raise ValueError(f"Unsupported URI scheme for metadata: {uri}")

    def discover_policies(
        self, base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "score"
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Discover policies from a base URI with selection strategy."""
        if base_uri.startswith("file://"):
            return self._discover_file_policies(base_uri, strategy, count, metric)
        elif base_uri.startswith("wandb://"):
            metadata = self.get_policy_metadata(base_uri)
            return [(base_uri, metadata)]
        else:
            raise ValueError(f"Unsupported URI scheme for discovery: {base_uri}")

    def _resolve_file_policy(self, uri: str, device: str) -> torch.nn.Module:
        """Resolve policy from file:// URI."""
        file_path = Path(uri[7:])

        if file_path.is_file():
            return torch.load(file_path, map_location=device, weights_only=False)
        elif file_path.is_dir():
            checkpoint_manager = self._get_or_create_checkpoint_manager(file_path)
            return checkpoint_manager.load_latest_agent()
        else:
            raise FileNotFoundError(f"Path does not exist: {file_path}")

    def _get_file_metadata(self, uri: str) -> Dict[str, Any]:
        """Get metadata from file:// URI."""
        file_path = Path(uri[7:])

        if file_path.is_file():
            return parse_checkpoint_filename(file_path.name) or {}
        elif file_path.is_dir():
            checkpoint_manager = self._get_or_create_checkpoint_manager(file_path)
            return checkpoint_manager.load_metadata() or {}
        else:
            raise FileNotFoundError(f"Path does not exist: {file_path}")

    def _discover_file_policies(
        self, base_uri: str, strategy: str, count: int, metric: str
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Discover policies from file:// base URI."""
        dir_path = Path(base_uri[7:])
        checkpoint_manager = self._get_or_create_checkpoint_manager(dir_path)

        strategy_map = {"latest": "latest", "best_score": "best_score", "top": "best_score", "all": "all"}
        checkpoint_strategy = strategy_map.get(strategy, "latest")

        selected_paths = checkpoint_manager.select_checkpoints(strategy=checkpoint_strategy, count=count, metric=metric)

        results = []
        for path in selected_paths:
            uri = f"file://{path}"
            metadata = self._get_file_metadata(uri)
            results.append((uri, metadata))

        return results

    def _get_or_create_checkpoint_manager(self, dir_path: Path) -> CheckpointManager:
        """Get or create CheckpointManager for a directory path."""
        dir_key = str(dir_path.resolve())

        if dir_key not in self._checkpoint_managers:
            if dir_path.name == "checkpoints":
                run_name = dir_path.parent.name
                run_dir = str(dir_path.parent.parent)
            else:
                run_name = dir_path.name
                run_dir = str(dir_path.parent)

            self._checkpoint_managers[dir_key] = CheckpointManager(run_name=run_name, run_dir=run_dir)

        return self._checkpoint_managers[dir_key]


# Global resolver instance for convenience
policy_resolver = PolicyUriResolver()


def resolve_policy_from_uri(uri: str, device: str = "cpu") -> torch.nn.Module:
    """Convenience function to resolve policy from URI."""
    return policy_resolver.resolve_policy(uri, device)


def get_policy_metadata_from_uri(uri: str) -> Dict[str, Any]:
    """Convenience function to get policy metadata from URI."""
    return policy_resolver.get_policy_metadata(uri)


def discover_policies_from_uri(
    base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "score"
) -> List[Tuple[str, Dict[str, Any]]]:
    """Convenience function to discover policies from base URI."""
    return policy_resolver.discover_policies(base_uri, strategy, count, metric)
