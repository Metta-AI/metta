"""Distributed statistics aggregation utilities."""

import logging
from typing import Any, Dict

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def aggregate_dual_policy_stats(stats: Dict[str, Any], device: torch.device) -> None:
    """
    Aggregate dual_policy/* statistics across all distributed nodes.

    This function modifies stats in-place, aggregating values for all keys
    starting with "dual_policy/" across all distributed processes.

    Args:
        stats: Dictionary of statistics to aggregate
        device: Device to use for tensor operations
    """
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    # Find all dual_policy keys
    dual_policy_keys = [k for k in stats.keys() if k.startswith("dual_policy/")]

    if not dual_policy_keys:
        return

    logger.debug(f"Aggregating {len(dual_policy_keys)} dual_policy stats across {world_size} nodes")

    for key in dual_policy_keys:
        values = stats[key]

        # Convert to list if not already
        if not isinstance(values, list):
            values = [values]

        # Calculate local sum and count
        if values:
            # Handle numeric values
            try:
                local_sum = sum(v for v in values if v is not None)
                local_count = len([v for v in values if v is not None])
            except (TypeError, ValueError):
                # Skip non-numeric values
                continue

            # Create tensors for distributed operations
            local_tensor = torch.tensor([local_sum, local_count], dtype=torch.float32, device=device)
            global_tensor = torch.zeros_like(local_tensor)

            # All-reduce to sum across all nodes
            dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)

            # Calculate global average
            global_sum = local_tensor[0].item()
            global_count = local_tensor[1].item()

            if global_count > 0:
                # Replace with aggregated average
                stats[key] = global_sum / global_count
            else:
                stats[key] = 0.0

            logger.debug(
                f"Aggregated {key}: local_sum={local_sum:.2f}, local_count={local_count}, "
                f"global_sum={global_sum:.2f}, global_count={global_count}, "
                f"global_avg={stats[key]:.2f}"
            )
