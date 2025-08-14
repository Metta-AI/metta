"""Distributed statistics aggregation utilities."""

import logging
import numbers
from typing import Any, Dict

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def aggregate_dual_policy_stats(stats: Dict[str, Any], device: torch.device) -> None:
    """
    Aggregate dual_policy/* statistics across all distributed nodes.

    This function modifies stats in-place by aggregating values for all keys
    starting with "dual_policy/". To ensure consistent collective calls across
    ranks and that master observes keys produced on non-master ranks, we first
    build a union of keys across all ranks and then aggregate each key.

    Args:
        stats: Dictionary of statistics to aggregate
        device: Device to use for tensor operations
    """
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    # Collect local dual_policy keys and build a global, deterministic union
    local_keys = {k for k in stats.keys() if k.startswith("dual_policy/")}
    gathered_key_sets = [None] * world_size  # type: ignore[var-annotated]
    dist.all_gather_object(gathered_key_sets, local_keys)
    union_keys = sorted(set().union(*gathered_key_sets))

    if not union_keys:
        logger.debug("No dual_policy stats found on any node to aggregate")
        return

    logger.debug(f"Aggregating {len(union_keys)} dual_policy stats across {world_size} nodes: {union_keys}")

    for key in union_keys:
        values = stats.get(key, [])

        # Normalize to list
        if not isinstance(values, list):
            values = [values]

        # Keep only numeric values (support Python and NumPy scalars)
        numeric_vals = [v for v in values if isinstance(v, numbers.Number)]
        local_sum = float(sum(numeric_vals)) if numeric_vals else 0.0
        local_count = float(len(numeric_vals))

        tensor = torch.tensor([local_sum, local_count], dtype=torch.float32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        global_sum = tensor[0].item()
        global_count = tensor[1].item()
        stats[key] = (global_sum / global_count) if global_count > 0 else 0.0

        logger.debug(
            f"Aggregated {key}: global_sum={global_sum:.2f}, global_count={global_count:.0f}, "
            f"global_avg={stats[key]:.4f}"
        )
