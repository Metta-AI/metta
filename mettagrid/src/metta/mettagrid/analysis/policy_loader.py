"""
Policy loading utilities for mettagrid analysis.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import wandb


class PolicyLoader:
    """
    Loads trained policies from wandb checkpoint URIs.

    This class handles downloading and loading policy checkpoints
    from wandb artifacts for analysis.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the policy loader.

        Args:
            cache_dir: Directory to cache downloaded checkpoints.
                      If None, uses system temp directory.
        """
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.cache_path = Path(self.cache_dir) / "mettagrid_policy_cache"
        self.cache_path.mkdir(exist_ok=True)

    def load_policy_from_wandb(self, wandb_uri: str) -> nn.Module:
        """
        Load a policy from a wandb checkpoint URI.

        Args:
            wandb_uri: Wandb artifact URI (e.g., "entity/project/run_id")

        Returns:
            Loaded policy model

        Raises:
            ValueError: If URI format is invalid
            RuntimeError: If policy loading fails
        """
        # Parse wandb URI
        try:
            entity, project, run_id = self._parse_wandb_uri(wandb_uri)
        except ValueError as e:
            raise ValueError(f"Invalid wandb URI format: {wandb_uri}. Expected: entity/project/run_id") from e

        # Download checkpoint
        checkpoint_path = self._download_checkpoint(entity, project, run_id)

        # Load policy
        try:
            policy = self._load_checkpoint(checkpoint_path)
            return policy
        except Exception as e:
            raise RuntimeError(f"Failed to load policy from {checkpoint_path}") from e

    def _parse_wandb_uri(self, uri: str) -> tuple[str, str, str]:
        """Parse wandb URI into components."""
        parts = uri.split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid URI format: {uri}")
        return parts[0], parts[1], parts[2]

    def _download_checkpoint(self, entity: str, project: str, run_id: str) -> Path:
        """Download checkpoint from wandb."""
        api = wandb.Api()

        # Find the run
        run = api.run(f"{entity}/{project}/{run_id}")

        # Find checkpoint artifact
        checkpoint_artifact = None
        for artifact in run.logged_artifacts():
            if "checkpoint" in artifact.name.lower() or "policy" in artifact.name.lower():
                checkpoint_artifact = artifact
                break

        if checkpoint_artifact is None:
            raise RuntimeError(f"No checkpoint artifact found in run {run_id}")

        # Download to cache
        download_path = self.cache_path / f"{entity}_{project}_{run_id}"
        checkpoint_artifact.download(root=str(download_path))

        # Find the actual checkpoint file
        checkpoint_file = None
        for file_path in download_path.rglob("*.pt"):
            checkpoint_file = file_path
            break

        if checkpoint_file is None:
            raise RuntimeError(f"No .pt checkpoint file found in {download_path}")

        return checkpoint_file

    def _load_checkpoint(self, checkpoint_path: Path) -> nn.Module:
        """Load policy from checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract policy from checkpoint
        # This will need to be customized based on the actual checkpoint format
        if "policy" in checkpoint:
            policy = checkpoint["policy"]
        elif "model" in checkpoint:
            policy = checkpoint["model"]
        elif "state_dict" in checkpoint:
            # Need to reconstruct model architecture
            # This is a placeholder - actual implementation depends on model structure
            raise NotImplementedError("Checkpoint contains state_dict - need model architecture")
        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint.keys()}")

        return policy

    def get_policy_info(self, wandb_uri: str) -> Dict[str, Any]:
        """Get metadata about a policy from wandb."""
        entity, project, run_id = self._parse_wandb_uri(wandb_uri)
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")

        return {
            "entity": entity,
            "project": project,
            "run_id": run_id,
            "config": run.config,
            "summary": run.summary,
            "tags": run.tags,
        }
