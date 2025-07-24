"""Utility functions for loading policies from checkpoint files."""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def load_policy_from_checkpoint(checkpoint_path: str, device: torch.device) -> Any:
    """Load a policy from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file (local, S3, or WandB URI)
        device: Device to load the policy on

    Returns:
        The loaded policy
    """
    # Handle WandB URIs
    if checkpoint_path.startswith("wandb://"):
        return _load_policy_from_wandb(checkpoint_path, device)

    # Handle S3 paths
    elif checkpoint_path.startswith("s3://"):
        return _load_policy_from_s3(checkpoint_path, device)

    # Handle local files
    else:
        return _load_policy_from_local(checkpoint_path, device)


def _load_policy_from_wandb(wandb_uri: str, device: torch.device) -> Any:
    """Load a policy from a WandB artifact."""
    import wandb

    # Parse WandB URI: wandb://entity/project/artifact_type/name:version
    uri_parts = wandb_uri[8:].split("/")  # Remove "wandb://"
    if len(uri_parts) < 4:
        raise ValueError(f"Invalid WandB URI format: {wandb_uri}")

    entity = uri_parts[0]
    project = uri_parts[1]
    artifact_type = uri_parts[2]
    name_version = uri_parts[3]

    # Split name and version
    if ":" in name_version:
        name, version = name_version.split(":", 1)
    else:
        name = name_version
        version = "latest"

    logger.info(f"Loading WandB artifact: {entity}/{project}/{artifact_type}/{name}:{version}")

    # Get the artifact
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{name}:{version}")

    # Download the artifact
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact.download(root=temp_dir)

        # Look for the model file
        model_path = os.path.join(temp_dir, "model.pt")
        if not os.path.exists(model_path):
            # Check if there are other .pt files
            pt_files = [f for f in os.listdir(temp_dir) if f.endswith(".pt")]
            if pt_files:
                model_path = os.path.join(temp_dir, pt_files[0])
            else:
                raise ValueError(f"No model file found in WandB artifact {wandb_uri}")

        return _load_policy_from_local(model_path, device)


def _load_policy_from_s3(s3_uri: str, device: torch.device) -> Any:
    """Load a policy from an S3 URI."""
    import tempfile
    from urllib.parse import urlparse

    import boto3

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3_client = boto3.client("s3")

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
        s3_client.download_file(bucket, key, tmp_file.name)
        return _load_policy_from_local(tmp_file.name, device)


def _load_policy_from_local(file_path: str, device: torch.device) -> Any:
    """Load a policy from a local file."""
    logger.info(f"Loading policy from local file: {file_path}")

    # Try multiple loading strategies for PyTorch 2.6+ compatibility
    checkpoint = None

    # Strategy 1: Try with weights_only=False (allows custom classes)
    try:
        checkpoint = torch.load(file_path, map_location=device, weights_only=False)
        logger.info("Successfully loaded with weights_only=False")
    except Exception as e:
        logger.warning(f"Failed to load with weights_only=False: {e}")

        # Strategy 2: Try with safe globals for PolicyRecord
        try:
            # Add PolicyRecord to safe globals if available
            try:
                from metta.agent.policy_record import PolicyRecord

                torch.serialization.add_safe_globals([PolicyRecord])
                logger.info("Added PolicyRecord to safe globals")
            except ImportError:
                logger.warning("PolicyRecord not available, skipping safe globals")

            checkpoint = torch.load(file_path, map_location=device)
            logger.info("Successfully loaded with safe globals")
        except Exception as e2:
            logger.warning(f"Failed to load with safe globals: {e2}")

            # Strategy 3: Try with weights_only=True (default in PyTorch 2.6+)
            try:
                checkpoint = torch.load(file_path, map_location=device, weights_only=True)
                logger.info("Successfully loaded with weights_only=True")
            except Exception as e3:
                logger.error(f"All loading strategies failed: {e3}")
                raise

    if checkpoint is None:
        raise ValueError("Failed to load checkpoint with any strategy")

    # Extract the policy from the checkpoint
    if isinstance(checkpoint, dict):
        # Check for common checkpoint structures
        if "policy" in checkpoint:
            policy = checkpoint["policy"]
        elif "model" in checkpoint:
            policy = checkpoint["model"]
        elif "state_dict" in checkpoint:
            # This might be a state dict that needs to be loaded into a model
            # For now, we'll assume it's the policy itself
            policy = checkpoint["state_dict"]
        else:
            # Assume the checkpoint is the policy itself
            policy = checkpoint
    else:
        # Assume the checkpoint is the policy itself
        policy = checkpoint

    # Ensure the policy is on the correct device
    if hasattr(policy, "to"):
        policy = policy.to(device)

    logger.info(f"Successfully loaded policy from {file_path}")
    return policy
