"""Simple replacement for PolicyRecord using dataclasses for CheckpointManager integration."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Simple dataclass containing checkpoint information - replacement for PolicyRecord."""

    uri: str
    run_name: str
    epoch: int
    agent_step: int = 0
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize metadata if None."""
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def from_checkpoint_manager(
        cls, checkpoint_manager, epoch: Optional[int] = None, uri_override: Optional[str] = None
    ) -> Optional["CheckpointInfo"]:
        """Create CheckpointInfo from CheckpointManager."""
        if epoch is None:
            epoch = checkpoint_manager.get_latest_epoch()

        if epoch is None:
            return None

        metadata = checkpoint_manager.load_metadata(epoch)
        if metadata is None:
            logger.warning(f"No metadata found for epoch {epoch}")
            metadata = {}

        # Generate URI from checkpoint directory if not provided
        if uri_override:
            uri = uri_override
        else:
            checkpoint_dir = checkpoint_manager.checkpoint_dir
            uri = f"file://{checkpoint_dir / f'agent_epoch_{epoch}.pt'}"

        return cls(
            uri=uri,
            run_name=metadata.get("run", checkpoint_manager.run_name),
            epoch=epoch,
            agent_step=metadata.get("agent_step", 0),
            score=metadata.get("score", 0.0),
            metadata=metadata,
        )

    @classmethod
    def from_file_path(cls, file_path: str) -> Optional["CheckpointInfo"]:
        """Create CheckpointInfo from a checkpoint file path."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Checkpoint file not found: {file_path}")
            return None

        # Extract epoch from filename
        try:
            # Expected format: agent_epoch_123.pt
            if not path.name.endswith(".pt"):
                raise ValueError(f"Expected .pt file, got: {path.name}")

            name_without_ext = path.name[:-3]
            parts = name_without_ext.split("_")
            if len(parts) < 3 or parts[1] != "epoch":
                raise ValueError(f"Expected format 'agent_epoch_<number>.pt', got: {path.name}")

            epoch = int(parts[-1])

            # Try to find run name from directory structure
            run_name = "unknown"
            if path.parent.name == "checkpoints" and path.parent.parent.name:
                run_name = path.parent.parent.name

            # Try to load metadata from YAML file
            yaml_path = path.with_suffix(".yaml")
            metadata = {}
            if yaml_path.exists():
                import yaml

                try:
                    with open(yaml_path) as f:
                        metadata = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {yaml_path}: {e}")

            return cls(
                uri=f"file://{file_path}",
                run_name=metadata.get("run", run_name),
                epoch=epoch,
                agent_step=metadata.get("agent_step", 0),
                score=metadata.get("score", 0.0),
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to create CheckpointInfo from {file_path}: {e}")
            return None
