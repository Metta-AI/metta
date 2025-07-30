from __future__ import annotations

import os
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
    # Auto-detect backend: use nccl for CUDA, gloo for CPU
    try:
        import torch

        backend = "nccl" if torch.cuda.is_available() else "gloo"
    except ImportError:
        backend = "gloo"  # fallback to gloo if torch not available

    dist.init_process_group(
        backend=backend,
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

    Args:
        fn: Function to run only on rank 0
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

    assert result is not None  # This should always be true after broadcast
    return result
