"""
Distributed training utilities for aggregating values across processes.
"""

import torch


def dist_sum(value: float | int, device: torch.device | str) -> float:
    """Sum a value across all distributed processes."""
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()


def dist_mean(value: float | int, device: torch.device | str) -> float:
    """Average a value across all distributed processes."""
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()
