"""Simple wandb policy loading utilities.

Minimal wandb integration for loading policies from wandb artifacts without complex abstractions.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def load_policy_from_wandb_uri(wandb_uri: str, device: str = "cpu") -> Optional[torch.nn.Module]:
    """Load a policy directly from a wandb URI.

    Supports simple wandb URI formats:
    - wandb://run/run_name (loads latest artifact from run)
    - wandb://entity/project/artifact_name:version
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not available. Install with: pip install wandb")
        return None

    try:
        # Parse the wandb URI
        if not wandb_uri.startswith("wandb://"):
            raise ValueError(f"Invalid wandb URI format: {wandb_uri}")

        uri_path = wandb_uri[8:]  # Remove "wandb://" prefix

        # Simple URI parsing - we'll support two formats:
        # 1. wandb://run/run_name (find latest artifact from run)
        # 2. wandb://entity/project/artifact_name:version (direct artifact)

        if uri_path.startswith("run/"):
            # Format: wandb://run/run_name
            run_name = uri_path[4:]  # Remove "run/" prefix
            return _load_from_run(run_name, device)
        else:
            # Format: wandb://entity/project/artifact_name:version
            return _load_from_artifact_path(uri_path, device)

    except Exception as e:
        logger.error(f"Failed to load policy from wandb URI {wandb_uri}: {e}")
        return None


def _load_from_run(run_name: str, device: str) -> Optional[torch.nn.Module]:
    """Load latest policy artifact from a wandb run."""
    import wandb

    api = wandb.Api()

    # Find the run (assume current user/project for simplicity)
    try:
        run = api.run(run_name)
    except Exception as e:
        logger.error(f"Could not find wandb run {run_name}: {e}")
        return None

    # Find the latest policy artifact
    artifacts = list(run.logged_artifacts())
    policy_artifacts = [a for a in artifacts if a.type == "model" or "policy" in a.name.lower()]

    if not policy_artifacts:
        logger.error(f"No policy artifacts found in run {run_name}")
        return None

    # Use the latest artifact
    latest_artifact = sorted(policy_artifacts, key=lambda x: x.version)[-1]
    return _download_and_load_artifact(latest_artifact, device)


def _load_from_artifact_path(artifact_path: str, device: str) -> Optional[torch.nn.Module]:
    """Load policy from direct artifact path."""
    import wandb

    api = wandb.Api()

    try:
        artifact = api.artifact(artifact_path)
        return _download_and_load_artifact(artifact, device)
    except Exception as e:
        logger.error(f"Could not load artifact {artifact_path}: {e}")
        return None


def _download_and_load_artifact(artifact, device: str) -> Optional[torch.nn.Module]:
    """Download wandb artifact and load the policy."""
    try:
        # Download artifact to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir) / "artifact"
            artifact.download(root=str(artifact_dir))

            # Find the policy file (look for .pt files)
            policy_files = list(artifact_dir.glob("*.pt"))
            if not policy_files:
                policy_files = list(artifact_dir.rglob("*.pt"))  # Search recursively

            if not policy_files:
                logger.error(f"No .pt files found in artifact {artifact.name}")
                return None

            # Load the first policy file found
            policy_file = policy_files[0]
            logger.info(f"Loading policy from {policy_file}")

            # Load with torch.load (weights_only=False for compatibility)
            policy = torch.load(policy_file, map_location=device, weights_only=False)
            return policy

    except Exception as e:
        logger.error(f"Failed to download and load artifact {artifact.name}: {e}")
        return None


def get_wandb_artifact_metadata(wandb_uri: str) -> dict:
    """Get metadata from wandb artifact without downloading.

    Returns basic information about the artifact for selection purposes.
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not available")
        return {}

    try:
        if not wandb_uri.startswith("wandb://"):
            return {}

        uri_path = wandb_uri[8:]  # Remove "wandb://" prefix

        if uri_path.startswith("run/"):
            run_name = uri_path[4:]
            api = wandb.Api()
            run = api.run(run_name)

            return {
                "run_name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "tags": run.tags,
                "config": dict(run.config),
                "summary": dict(run.summary),
            }
        else:
            # Direct artifact path
            api = wandb.Api()
            artifact = api.artifact(uri_path)

            return {
                "artifact_name": artifact.name,
                "version": artifact.version,
                "type": artifact.type,
                "size": artifact.size,
                "created_at": artifact.created_at,
                "metadata": artifact.metadata,
            }

    except Exception as e:
        logger.error(f"Failed to get wandb metadata for {wandb_uri}: {e}")
        return {}
