"""Simple URI resolution infrastructure for policy loading.

Provides a clean interface for resolving and loading policies from different sources.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.wandb_policy_loader import load_policy_from_wandb_uri, get_wandb_artifact_metadata

logger = logging.getLogger(__name__)


class PolicyUriResolver:
    """Simple URI dispatcher for file:// and wandb:// schemes.

    Provides unified interface for loading policies regardless of storage location.
    """

    def __init__(self):
        self._checkpoint_managers = {}  # Cache CheckpointManagers for reuse

    def resolve_policy(self, uri: str, device: str = "cpu") -> Optional[torch.nn.Module]:
        """Load a policy from any supported URI scheme.

        Supports file:// and wandb:// URIs with basic error handling.
        """
        try:
            if uri.startswith("file://"):
                return self._resolve_file_policy(uri, device)
            elif uri.startswith("wandb://"):
                return self._resolve_wandb_policy(uri, device)
            else:
                logger.error(f"Unsupported URI scheme: {uri}")
                return None

        except Exception as e:
            logger.error(f"Failed to resolve policy from {uri}: {e}")
            return None

    def get_policy_metadata(self, uri: str) -> Dict[str, Any]:
        """Get metadata from a policy URI without loading the full policy.

        Returns empty dict if metadata cannot be retrieved.
        """
        try:
            if uri.startswith("file://"):
                return self._get_file_metadata(uri)
            elif uri.startswith("wandb://"):
                return get_wandb_artifact_metadata(uri)
            else:
                logger.error(f"Unsupported URI scheme for metadata: {uri}")
                return {}

        except Exception as e:
            logger.error(f"Failed to get metadata from {uri}: {e}")
            return {}

    def discover_policies(
        self, base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "score"
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Discover policies from a base URI with selection strategy.

        Returns list of (uri, metadata) tuples for discovered policies.
        """
        try:
            if base_uri.startswith("file://"):
                return self._discover_file_policies(base_uri, strategy, count, metric)
            elif base_uri.startswith("wandb://"):
                # For wandb, just return the single artifact
                metadata = self.get_policy_metadata(base_uri)
                return [(base_uri, metadata)] if metadata else []
            else:
                logger.error(f"Unsupported URI scheme for discovery: {base_uri}")
                return []

        except Exception as e:
            logger.error(f"Failed to discover policies from {base_uri}: {e}")
            return []

    def _resolve_file_policy(self, uri: str, device: str) -> Optional[torch.nn.Module]:
        """Resolve policy from file:// URI."""
        file_path = Path(uri[7:])  # Remove "file://" prefix

        if file_path.is_file():
            # Direct file path
            return torch.load(file_path, map_location=device, weights_only=False)
        elif file_path.is_dir():
            # Directory with checkpoints - use CheckpointManager
            checkpoint_manager = self._get_or_create_checkpoint_manager(file_path)
            return checkpoint_manager.load_latest_agent()
        else:
            logger.error(f"File path does not exist: {file_path}")
            return None

    def _resolve_wandb_policy(self, uri: str, device: str) -> Optional[torch.nn.Module]:
        """Resolve policy from wandb:// URI."""
        return load_policy_from_wandb_uri(uri, device)

    def _get_file_metadata(self, uri: str) -> Dict[str, Any]:
        """Get metadata from file:// URI."""
        file_path = Path(uri[7:])  # Remove "file://" prefix

        if file_path.is_file():
            # Check for companion YAML file
            yaml_path = file_path.with_suffix(".yaml")
            if yaml_path.exists():
                try:
                    import yaml

                    with open(yaml_path) as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load YAML metadata: {e}")
            return {}
        elif file_path.is_dir():
            # Directory with checkpoints
            checkpoint_manager = self._get_or_create_checkpoint_manager(file_path)
            return checkpoint_manager.load_metadata() or {}
        else:
            return {}

    def _discover_file_policies(
        self, base_uri: str, strategy: str, count: int, metric: str
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Discover policies from file:// base URI."""
        dir_path = Path(base_uri[7:])  # Remove "file://" prefix

        if not dir_path.is_dir():
            return []

        checkpoint_manager = self._get_or_create_checkpoint_manager(dir_path)

        # Map strategy names to CheckpointManager strategies
        strategy_map = {"latest": "latest", "best_score": "best_score", "top": "best_score", "all": "all"}

        checkpoint_strategy = strategy_map.get(strategy, "latest")

        try:
            selected_paths = checkpoint_manager.select_checkpoints(
                strategy=checkpoint_strategy, count=count, metric=metric
            )

            results = []
            for path in selected_paths:
                uri = f"file://{path}"
                metadata = self._get_file_metadata(uri)
                results.append((uri, metadata))

            return results

        except Exception as e:
            logger.error(f"Failed to discover file policies: {e}")
            return []

    def _get_or_create_checkpoint_manager(self, dir_path: Path) -> CheckpointManager:
        """Get or create CheckpointManager for a directory path."""
        dir_key = str(dir_path.resolve())

        if dir_key not in self._checkpoint_managers:
            # Extract run_name and run_dir from path structure
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


def resolve_policy_from_uri(uri: str, device: str = "cpu") -> Optional[torch.nn.Module]:
    """Convenience function to resolve policy from URI.

    Uses global resolver instance for simple policy loading.
    """
    return policy_resolver.resolve_policy(uri, device)


def get_policy_metadata_from_uri(uri: str) -> Dict[str, Any]:
    """Convenience function to get policy metadata from URI.

    Uses global resolver instance for simple metadata access.
    """
    return policy_resolver.get_policy_metadata(uri)


def discover_policies_from_uri(
    base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "score"
) -> List[Tuple[str, Dict[str, Any]]]:
    """Convenience function to discover policies from base URI.

    Uses global resolver instance for simple policy discovery.
    """
    return policy_resolver.discover_policies(base_uri, strategy, count, metric)
