"""Policy discovery utilities to restore critical functionality identified in Phase 11 audit."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from metta.rl.checkpoint_info import CheckpointInfo
from metta.rl.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def discover_checkpoints_in_directory(
    directory: Union[str, Path], pattern: str = "**/checkpoints/agent_epoch_*.pt"
) -> List[CheckpointInfo]:
    """Discover all checkpoints in a directory tree.

    Args:
        directory: Root directory to search
        pattern: Glob pattern for checkpoint files

    Returns:
        List of CheckpointInfo objects for discovered checkpoints
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    checkpoints = []
    for checkpoint_file in directory.glob(pattern):
        try:
            checkpoint_info = CheckpointInfo.from_file_path(str(checkpoint_file))
            if checkpoint_info:
                checkpoints.append(checkpoint_info)
        except Exception as e:
            logger.warning(f"Failed to create CheckpointInfo from {checkpoint_file}: {e}")

    logger.info(f"Discovered {len(checkpoints)} checkpoints in {directory}")
    return checkpoints


def filter_checkpoints_by_score(
    checkpoints: List[CheckpointInfo], min_score: Optional[float] = None, max_score: Optional[float] = None
) -> List[CheckpointInfo]:
    """Filter checkpoints by score range.

    Args:
        checkpoints: List of checkpoint infos to filter
        min_score: Minimum score threshold (inclusive)
        max_score: Maximum score threshold (inclusive)

    Returns:
        Filtered list of checkpoints
    """
    filtered = []
    for checkpoint in checkpoints:
        score = checkpoint.score

        if min_score is not None and score < min_score:
            continue
        if max_score is not None and score > max_score:
            continue

        filtered.append(checkpoint)

    return filtered


def select_top_checkpoints(checkpoints: List[CheckpointInfo], n: int, metric: str = "score") -> List[CheckpointInfo]:
    """Select top N checkpoints by a given metric.

    Args:
        checkpoints: List of checkpoint infos to select from
        n: Number of checkpoints to select
        metric: Metric to sort by ('score', 'epoch', 'agent_step')

    Returns:
        Top N checkpoints sorted by metric (highest first)
    """
    if not checkpoints:
        return []

    def get_metric_value(checkpoint: CheckpointInfo) -> float:
        if metric == "score":
            return checkpoint.score
        elif metric == "epoch":
            return float(checkpoint.epoch)
        elif metric == "agent_step":
            return float(checkpoint.agent_step)
        elif checkpoint.metadata and metric in checkpoint.metadata:
            return float(checkpoint.metadata[metric])
        else:
            logger.warning(f"Unknown metric '{metric}', using score")
            return checkpoint.score

    try:
        sorted_checkpoints = sorted(checkpoints, key=get_metric_value, reverse=True)
        return sorted_checkpoints[:n]
    except Exception as e:
        logger.error(f"Failed to sort checkpoints by {metric}: {e}")
        return checkpoints[:n]


def select_latest_checkpoints(checkpoints: List[CheckpointInfo], n: int = 1) -> List[CheckpointInfo]:
    """Select the most recent checkpoints by epoch.

    Args:
        checkpoints: List of checkpoint infos to select from
        n: Number of checkpoints to select

    Returns:
        Latest N checkpoints sorted by epoch (newest first)
    """
    return select_top_checkpoints(checkpoints, n, metric="epoch")


def group_checkpoints_by_run(checkpoints: List[CheckpointInfo]) -> Dict[str, List[CheckpointInfo]]:
    """Group checkpoints by run name.

    Args:
        checkpoints: List of checkpoint infos to group

    Returns:
        Dictionary mapping run names to lists of checkpoints
    """
    groups: Dict[str, List[CheckpointInfo]] = {}
    for checkpoint in checkpoints:
        run_name = checkpoint.run_name
        if run_name not in groups:
            groups[run_name] = []
        groups[run_name].append(checkpoint)

    # Sort checkpoints within each group by epoch
    for run_name, run_checkpoints in groups.items():
        run_checkpoints.sort(key=lambda cp: cp.epoch, reverse=True)

    return groups


def find_best_checkpoint_in_runs(
    train_dir: Union[str, Path], run_names: Optional[List[str]] = None, metric: str = "score", n_best: int = 1
) -> List[CheckpointInfo]:
    """Find the best checkpoint(s) across multiple training runs.

    Args:
        train_dir: Training directory containing run folders
        run_names: Specific run names to search (None for all runs)
        metric: Metric to optimize for
        n_best: Number of best checkpoints to return

    Returns:
        List of best CheckpointInfo objects
    """
    train_dir = Path(train_dir)
    if not train_dir.exists():
        logger.error(f"Training directory does not exist: {train_dir}")
        return []

    all_checkpoints = []

    # Determine which runs to search
    if run_names:
        run_dirs = [train_dir / name for name in run_names if (train_dir / name).exists()]
    else:
        run_dirs = [d for d in train_dir.iterdir() if d.is_dir()]

    for run_dir in run_dirs:
        try:
            run_name = run_dir.name
            checkpoint_manager = CheckpointManager(run_name=run_name, run_dir=str(train_dir))

            if not checkpoint_manager.exists():
                continue

            # Find best checkpoint for this run
            best_checkpoint_path = checkpoint_manager.find_best_checkpoint(metric)
            if best_checkpoint_path:
                checkpoint_info = CheckpointInfo.from_file_path(str(best_checkpoint_path))
                if checkpoint_info:
                    all_checkpoints.append(checkpoint_info)

        except Exception as e:
            logger.warning(f"Failed to process run {run_dir}: {e}")

    # Return top N across all runs
    return select_top_checkpoints(all_checkpoints, n_best, metric)


def create_checkpoint_summary(checkpoints: List[CheckpointInfo]) -> str:
    """Create a text summary of checkpoints for display.

    Args:
        checkpoints: List of checkpoint infos to summarize

    Returns:
        Formatted text summary
    """
    if not checkpoints:
        return "No checkpoints found."

    lines = [f"Found {len(checkpoints)} checkpoints:"]
    lines.append("-" * 60)

    for i, cp in enumerate(checkpoints, 1):
        lines.append(f"{i:2d}. {cp.run_name} (epoch {cp.epoch}) - score: {cp.score:.3f}, steps: {cp.agent_step}")

    return "\n".join(lines)
