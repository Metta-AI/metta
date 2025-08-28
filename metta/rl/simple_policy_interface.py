"""Minimal interface for PolicyEvaluator integration - exactly what's needed, nothing more."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class SimplePolicyRecord:
    """Minimal PolicyRecord replacement for evaluation system integration.

    Provides exactly the interface that PolicyEvaluator expects.
    """

    run_name: str
    uri: str
    metadata: Dict[str, Any]
    _cached_policy: Any = None

    def extract_wandb_run_info(self) -> tuple[str, str, str, str | None]:
        """Extract wandb info from URI - kept for compatibility."""
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


def get_policy_record_from_checkpoint_dir(checkpoint_dir: str) -> Optional[SimplePolicyRecord]:
    """Get a policy record from a checkpoint directory.

    Args:
        checkpoint_dir: Path to directory containing agent_epoch_*.pt files

    Returns:
        SimplePolicyRecord for the latest checkpoint, or None if no checkpoints found
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

    # Load the policy
    try:
        agent = torch.load(latest_file, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint {latest_file}: {e}")
        return None

    # Extract run name from directory structure
    run_name = checkpoint_path.parent.name if checkpoint_path.parent else "unknown"

    return SimplePolicyRecord(run_name=run_name, uri=f"file://{latest_file}", metadata={}, _cached_policy=agent)


def get_policy_tuples_for_stats_integration(checkpoint_dirs: list[str]) -> list[tuple[str, str, str | None]]:
    """Get policy tuples for get_or_create_policy_ids function.

    Args:
        checkpoint_dirs: List of checkpoint directory paths

    Returns:
        List of (policy_name, policy_uri, description) tuples
    """
    policy_tuples = []

    for checkpoint_dir in checkpoint_dirs:
        policy_record = get_policy_record_from_checkpoint_dir(checkpoint_dir)
        if policy_record:
            policy_tuples.append(
                (
                    policy_record.run_name,
                    policy_record.uri,
                    None,  # No description needed
                )
            )

    return policy_tuples
