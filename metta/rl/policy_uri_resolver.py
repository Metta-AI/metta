"""Simple URI resolution for policy loading."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from metta.rl.checkpoint_manager import parse_checkpoint_filename
from metta.rl.wandb_policy_loader import get_wandb_artifact_metadata, load_policy_from_wandb_uri


def resolve_policy(uri: str, device: str = "cpu") -> torch.nn.Module:
    """Load a policy from file:// or wandb:// URI."""
    if uri.startswith("file://"):
        file_path = Path(uri[7:])
        if file_path.is_file():
            return torch.load(file_path, map_location=device, weights_only=False)
        elif file_path.is_dir():
            # Find latest checkpoint in directory
            pattern = "*.e*.s*.t*.pt"
            checkpoints = list(file_path.glob(pattern))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint files found in {file_path}")
            latest = max(checkpoints, key=lambda f: parse_checkpoint_filename(f.name)["epoch"])
            return torch.load(latest, map_location=device, weights_only=False)
        else:
            raise FileNotFoundError(f"Path does not exist: {file_path}")
    elif uri.startswith("wandb://"):
        return load_policy_from_wandb_uri(uri, device)
    else:
        raise ValueError(f"Unsupported URI scheme: {uri}")


def get_policy_metadata(uri: str) -> Dict[str, Any]:
    """Get metadata from a policy URI."""
    if uri.startswith("file://"):
        file_path = Path(uri[7:])
        if file_path.is_file():
            return parse_checkpoint_filename(file_path.name) or {}
        elif file_path.is_dir():
            # Find latest checkpoint in directory
            pattern = "*.e*.s*.t*.pt"
            checkpoints = list(file_path.glob(pattern))
            if not checkpoints:
                return {}
            latest = max(checkpoints, key=lambda f: parse_checkpoint_filename(f.name)["epoch"])
            return parse_checkpoint_filename(latest.name) or {}
        else:
            return {}
    elif uri.startswith("wandb://"):
        return get_wandb_artifact_metadata(uri)
    else:
        raise ValueError(f"Unsupported URI scheme: {uri}")


def discover_policies(
    base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "epoch"
) -> List[Tuple[str, Dict[str, Any]]]:
    """Discover policies from a base URI."""
    if base_uri.startswith("file://"):
        dir_path = Path(base_uri[7:])
        pattern = "*.e*.s*.t*.pt"
        checkpoints = list(dir_path.glob(pattern))
        if not checkpoints:
            return []

        # Sort checkpoints based on strategy
        if strategy == "latest":
            checkpoints.sort(key=lambda f: parse_checkpoint_filename(f.name)["epoch"], reverse=True)
        elif strategy in ["best_score", "top"]:
            checkpoints.sort(key=lambda f: parse_checkpoint_filename(f.name).get(metric, float("-inf")), reverse=True)
        elif strategy != "all":
            raise ValueError(f"Unknown strategy: {strategy}")

        # Return requested count
        if strategy == "all":
            count = len(checkpoints)

        results = []
        for checkpoint in checkpoints[:count]:
            uri = f"file://{checkpoint}"
            metadata = parse_checkpoint_filename(checkpoint.name) or {}
            results.append((uri, metadata))

        return results
    elif base_uri.startswith("wandb://"):
        metadata = get_policy_metadata(base_uri)
        return [(base_uri, metadata)]
    else:
        raise ValueError(f"Unsupported URI scheme: {base_uri}")


class PolicyUriResolver:
    """Legacy class for backward compatibility."""

    def resolve_policy(self, uri: str, device: str = "cpu") -> torch.nn.Module:
        return resolve_policy(uri, device)

    def get_policy_metadata(self, uri: str) -> Dict[str, Any]:
        return get_policy_metadata(uri)

    def discover_policies(
        self, base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "score"
    ) -> List[Tuple[str, Dict[str, Any]]]:
        return discover_policies(base_uri, strategy, count, metric)


# Convenience functions (using new direct functions)
def resolve_policy_from_uri(uri: str, device: str = "cpu") -> torch.nn.Module:
    return resolve_policy(uri, device)


def get_policy_metadata_from_uri(uri: str) -> Dict[str, Any]:
    return get_policy_metadata(uri)


def discover_policies_from_uri(
    base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "score"
) -> List[Tuple[str, Dict[str, Any]]]:
    return discover_policies(base_uri, strategy, count, metric)


# Legacy global instance for backward compatibility
policy_resolver = PolicyUriResolver()
