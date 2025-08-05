from __future__ import annotations

import os
from datetime import timedelta
from typing import Callable, TypeVar

import torch.distributed as dist

T = TypeVar("T")


def _init_process_group() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("NUM_NODES", "1")))
    if world_size <= 1:
        return False
    if dist.is_initialized():
        return False

    rank = int(os.environ.get("RANK", os.environ.get("NODE_INDEX", "0")))
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=30),
        init_method=os.environ.get("DIST_URL", "env://"),
        world_size=world_size,
        rank=rank,
    )
    return True


def run_once(fn: Callable[[], T]) -> T:
    """Run ``fn`` only on rank 0 and broadcast the result.

    If ``torch.distributed`` is not initialized, this function will attempt to
    initialize it using environment variables typically provided when running
    multi-node jobs (``WORLD_SIZE``/``NUM_NODES`` and ``RANK``/``NODE_INDEX``).
    The NCCL backend is used.
    """
    group_initialized = _init_process_group()
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    result: T | None = fn() if rank == 0 else None

    if dist.is_initialized():
        result_list = [result]
        dist.broadcast_object_list(result_list, src=0)
        result = result_list[0]

    if group_initialized:
        dist.destroy_process_group()

    return result
