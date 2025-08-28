"""Minimal checkpoint interface for evaluation integration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from metta.rl.checkpoint_manager import parse_checkpoint_filename


@dataclass
class Checkpoint:
    """Simple checkpoint data container."""

    run_name: str
    uri: str
    metadata: Dict[str, Any]
    _cached_policy: Any = None

    def key_and_version(self) -> tuple[str, int]:
        return self.run_name, self.metadata.get("epoch", 0)

    def extract_wandb_run_info(self) -> tuple[str, str, str, str | None]:
        """Extract wandb info from URI."""
        if not self.uri or not self.uri.startswith("wandb://"):
            raise ValueError("Invalid wandb URI")

        parts = self.uri[8:].split("/")  # Remove "wandb://"
        if len(parts) < 3:
            raise ValueError(f"Invalid wandb URI format: {self.uri}")

        entity, project, name = parts[0], parts[1], parts[2]
        version = name.split(":")[1] if ":" in name else None
        if ":" in name:
            name = name.split(":")[0]
        return entity, project, name, version


def get_checkpoint_from_dir(checkpoint_dir: str) -> Optional[Checkpoint]:
    """Get latest checkpoint from directory, supporting both new dot-separated and old formats."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    run_name = checkpoint_path.parent.name if checkpoint_path.parent else "unknown"

    # Try new dot-separated format first
    new_format_files = list(checkpoint_path.glob(f"{run_name}.e*.s*.t*.pt"))
    if new_format_files:
        latest_file = max(new_format_files, key=lambda p: parse_checkpoint_filename(p.name)["epoch"])
        metadata = parse_checkpoint_filename(latest_file.name) or {}
        agent = torch.load(latest_file, weights_only=False)
        return Checkpoint(run_name=run_name, uri=f"file://{latest_file}", metadata=metadata, _cached_policy=agent)

    # Fallback to old format
    old_format_files = list(checkpoint_path.glob("agent_epoch_*.pt"))
    if old_format_files:
        latest_file = max(
            old_format_files, key=lambda f: int(f.stem.split("_")[-1]) if f.stem.split("_")[-1].isdigit() else 0
        )
        agent = torch.load(latest_file, weights_only=False)
        # Extract epoch from filename
        epoch_str = latest_file.stem.split("_")[-1]
        epoch = int(epoch_str) if epoch_str.isdigit() else 0
        metadata = {"epoch": epoch}
        return Checkpoint(run_name=run_name, uri=f"file://{latest_file}", metadata=metadata, _cached_policy=agent)

    return None


def get_checkpoint_tuples_for_stats_integration(checkpoint_dirs: list[str]) -> list[tuple[str, str, str | None]]:
    """Convert checkpoint directories to tuples for stats integration."""
    return [
        (checkpoint.run_name, checkpoint.uri, None)
        for checkpoint_dir in checkpoint_dirs
        if (checkpoint := get_checkpoint_from_dir(checkpoint_dir)) is not None
    ]
