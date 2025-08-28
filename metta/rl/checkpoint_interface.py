"""Minimal checkpoint interface for evaluation integration - exactly what's needed, nothing more."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Simple checkpoint data container - groups policy, trainer state, and metadata.

    This is NOT backwards compatible with PolicyRecord - it's a clean, simple replacement.
    Just contains the essential data needed for evaluation: the run name, file location,
    basic metadata, and the loaded policy.
    """

    run_name: str
    uri: str
    metadata: Dict[str, Any]
    _cached_policy: Any = None

    def key_and_version(self) -> tuple[str, int]:
        """Extract (key, version) tuple for database normalization.

        For database integration, we use run_name as key and extract epoch from metadata.
        """
        epoch = self.metadata.get("epoch", 0)
        return self.run_name, epoch

    def extract_wandb_run_info(self) -> tuple[str, str, str, str | None]:
        """Extract wandb info from URI - kept for evaluation system compatibility."""
        if self.uri is None or not self.uri.startswith("wandb://"):
            raise ValueError("Cannot get wandb info without a valid URI.")
        try:
            entity, project, name = self.uri[len("wandb://") :].split("/")
            version: str | None = None
            if ":" in name:
                name, version = name.split(":")
            return entity, project, name, version
        except ValueError as e:
            raise ValueError(
                f"Failed to parse wandb URI: {self.uri}. Expected format: wandb://<entity>/<project>/<name>"
            ) from e


def get_checkpoint_from_dir(checkpoint_dir: str) -> Optional[Checkpoint]:
    """Get a checkpoint from a directory containing agent_epoch_*.pt files.

    Loads the latest checkpoint by epoch number, or None if no checkpoints found.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    # Find latest checkpoint
    agent_files = list(checkpoint_path.glob("agent_epoch_*.pt"))
    if not agent_files:
        logger.warning(f"No checkpoint files found in {checkpoint_dir}")
        return None

    # Get latest by epoch number
    try:
        latest_file = max(agent_files, key=lambda f: int(f.stem.split("_")[-1]))
    except (ValueError, IndexError) as e:
        logger.error(f"Failed to parse epoch numbers from checkpoint files: {e}")
        return None

    # Load the policy using weights_only=False
    try:
        agent = torch.load(latest_file, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint {latest_file}: {e}")
        return None

    # Extract run name from directory structure
    run_name = checkpoint_path.parent.name if checkpoint_path.parent else "unknown"

    return Checkpoint(run_name=run_name, uri=f"file://{latest_file}", metadata={}, _cached_policy=agent)


def get_checkpoint_tuples_for_stats_integration(checkpoint_dirs: list[str]) -> list[tuple[str, str, str | None]]:
    """Get checkpoint tuples for get_or_create_policy_ids function.

    Converts checkpoint directories into tuples suitable for stats server integration.
    """
    checkpoint_tuples = []

    for checkpoint_dir in checkpoint_dirs:
        checkpoint = get_checkpoint_from_dir(checkpoint_dir)
        if checkpoint:
            checkpoint_tuples.append(
                (
                    checkpoint.run_name,
                    checkpoint.uri,
                    None,  # No description needed
                )
            )

    return checkpoint_tuples
