from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def broadcast_bytes(data: bytes | None, device: torch.device | str) -> bytes:
    """Broadcast bytes from rank 0 to all processes using NCCL.

    Parameters
    ----------
    data: bytes | None
        Data to broadcast on rank 0. Ignored on other ranks.
    device: torch.device | str
        Device for temporary tensors used in the broadcast.

    Returns
    -------
    bytes
        The broadcasted data on all ranks.
    """
    if not dist.is_initialized():
        assert data is not None
        return data

    device = torch.device(device)
    rank = dist.get_rank()

    if rank == 0:
        assert data is not None
        buffer = torch.as_tensor(np.frombuffer(data, dtype=np.uint8), device=device)
        size = torch.tensor([buffer.numel()], dtype=torch.long, device=device)
    else:
        buffer = torch.tensor([], dtype=torch.uint8, device=device)
        size = torch.tensor([0], dtype=torch.long, device=device)

    dist.broadcast(size, src=0)
    if rank != 0:
        buffer = torch.empty(size.item(), dtype=torch.uint8, device=device)
    dist.broadcast(buffer, src=0)

    if rank != 0:
        data = buffer.cpu().numpy().tobytes()
    return data


def distributed_torch_load(
    path: str, map_location: torch.device | str = "cpu", *, weights_only: bool | None = None
) -> Any:
    """Load a checkpoint on rank 0 and broadcast to other ranks."""
    if dist.is_initialized():
        broadcast_device = (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )
        if dist.get_rank() == 0:
            with open(path, "rb") as f:
                data = f.read()
        else:
            data = None
        data = broadcast_bytes(data, broadcast_device)
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location=map_location, weights_only=weights_only)

    return torch.load(path, map_location=map_location, weights_only=weights_only)
