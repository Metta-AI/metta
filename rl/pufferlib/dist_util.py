import io
import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger("dist_util")

def send_object(obj, dst_rank=0):
    """
    Send a Python object to the destination rank using torch.distributed.

    Args:
        obj: The Python object to send
        dst_rank: The destination rank (default: 0)
    """
    if not dist.is_initialized():
        logger.warning("Distributed training not initialized, skipping send")
        return

    # Serialize the object to a buffer
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)

    # Get the size of the serialized object
    size = torch.tensor(buffer.getbuffer().nbytes, dtype=torch.long)

    # Send the size
    dist.send(size, dst=dst_rank)

    # Send the serialized data
    data = torch.ByteTensor(list(buffer.getbuffer()))
    dist.send(data, dst=dst_rank)

    logger.debug(f"Sent object of size {size.item()} bytes to rank {dst_rank}")

def recv_object(src_rank=0):
    """
    Receive a Python object from the source rank using torch.distributed.

    Args:
        src_rank: The source rank (default: 0)

    Returns:
        The received Python object
    """
    if not dist.is_initialized():
        logger.warning("Distributed training not initialized, skipping receive")
        return None

    # Receive the size
    size = torch.tensor(0, dtype=torch.long)
    dist.recv(size, src=src_rank)

    # Receive the serialized data
    data = torch.ByteTensor(size.item())
    dist.recv(data, src=src_rank)

    # Deserialize the object
    buffer = io.BytesIO(data.numpy().tobytes())
    obj = torch.load(buffer)

    logger.debug(f"Received object of size {size.item()} bytes from rank {src_rank}")
    return obj

def _move_to_device(obj, device):
    """
    Recursively move an object and all its attributes that are torch modules or tensors to the specified device.

    Args:
        obj: The object to move
        device: The target device

    Returns:
        The object with all torch modules and tensors moved to the device
    """
    if isinstance(obj, nn.Module):
        return obj.to(device)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif hasattr(obj, '__dict__'):
        for key, val in obj.__dict__.items():
            if isinstance(val, (nn.Module, torch.Tensor)):
                obj.__dict__[key] = val.to(device)
            elif hasattr(val, '__dict__'):
                obj.__dict__[key] = _move_to_device(val, device)
    return obj

def broadcast_object(obj, src_rank=0, target_device=None):
    """
    Broadcast a Python object from the source rank to all processes.

    Note: This function assumes CUDA is available when distributed training is initialized.
    If target_device is specified, any torch modules in the object will be moved to that device.

    Args:
        obj: The Python object to broadcast (only needed on src_rank)
        src_rank: The source rank (default: 0)
        target_device: The device to move torch modules to after deserialization (default: None)

    Returns:
        The broadcast Python object
    """
    if not dist.is_initialized():
        logger.warning("Distributed training not initialized, skipping broadcast")
        return obj

    rank = dist.get_rank()

    if rank == src_rank:
        # Serialize the object to a buffer
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer.seek(0)

        # Get the size of the serialized object
        size = torch.tensor(buffer.getbuffer().nbytes, dtype=torch.long, device=target_device)

        # Convert buffer to tensor
        data = torch.ByteTensor(list(buffer.getbuffer())).to(target_device)
    else:
        # Create empty tensors to receive data
        size = torch.tensor(0, dtype=torch.long, device=target_device)
        data = None  # Will be initialized after receiving size

    # Broadcast the size
    dist.broadcast(size, src=src_rank)

    # Initialize data tensor on non-source ranks
    if rank != src_rank:
        data = torch.ByteTensor(size.item(), device=target_device)

    # Broadcast the data
    dist.broadcast(data, src=src_rank)

    # Deserialize on non-source ranks
    if rank != src_rank:
        # Move data to CPU for deserialization
        cpu_data = data.cpu()
        buffer = io.BytesIO(cpu_data.numpy().tobytes())
        obj = torch.load(buffer)

    # Move any torch modules to the target device if specified
    if target_device is not None:
        logger.debug(f"Moving object to device: {target_device}")
        obj = _move_to_device(obj, target_device)

    logger.debug(f"Broadcast object of size {size.item()} bytes from rank {src_rank}")
    return obj
