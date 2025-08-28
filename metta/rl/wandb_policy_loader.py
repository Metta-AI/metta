"""Simple wandb policy loading."""

import tempfile
from pathlib import Path
from typing import Optional

import torch

try:
    import wandb
except ImportError:
    wandb = None


def load_policy_from_wandb_uri(wandb_uri: str, device: str = "cpu") -> Optional[torch.nn.Module]:
    """Load policy from wandb://entity/project/artifact_name:version format."""
    if not wandb or not wandb_uri.startswith("wandb://"):
        return None

    try:
        artifact_path = wandb_uri[8:]  # Remove "wandb://" prefix
        artifact = wandb.Api().artifact(artifact_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            artifact.download(root=str(artifact_dir))

            policy_files = list(artifact_dir.rglob("*.pt"))
            if policy_files:
                return torch.load(policy_files[0], map_location=device, weights_only=False)

        return None
    except Exception:
        return None


def get_wandb_artifact_metadata(wandb_uri: str) -> dict:
    """Get metadata from wandb artifact."""
    if not wandb or not wandb_uri.startswith("wandb://"):
        return {}

    try:
        artifact_path = wandb_uri[8:]  # Remove "wandb://" prefix
        artifact = wandb.Api().artifact(artifact_path)
        return {
            "artifact_name": artifact.name,
            "version": artifact.version,
            "type": artifact.type,
            "created_at": artifact.created_at,
        }
    except Exception:
        return {}
