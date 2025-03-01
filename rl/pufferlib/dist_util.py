import io
import logging
import os
import sys
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
    obj = torch.load(buffer, weights_only=False)

    logger.debug(f"Received object of size {size.item()} bytes from rank {src_rank}")
    return obj

def _move_to_device(obj, device, visited=None, depth=0, max_depth=100):
    """
    Recursively move an object and all its attributes that are torch modules or tensors to the specified device.

    Args:
        obj: The object to move
        device: The target device
        visited: Set of object ids already visited (to prevent circular references)
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent stack overflow

    Returns:
        The object with all torch modules and tensors moved to the device
    """
    # Initialize visited set if this is the first call
    if visited is None:
        visited = set()

    # Check for circular references or excessive recursion
    obj_id = id(obj)
    if obj_id in visited or depth > max_depth:
        return obj

    # Add this object to visited set
    visited.add(obj_id)

    # Handle torch modules and tensors
    if isinstance(obj, nn.Module):
        return obj.to(device)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif hasattr(obj, '__dict__') and not isinstance(obj.__dict__, type({}).__dict__.__class__):
        # Only try to modify __dict__ if it's not a mappingproxy (which is immutable)
        try:
            for key, val in list(obj.__dict__.items()):
                if isinstance(val, (nn.Module, torch.Tensor)):
                    obj.__dict__[key] = val.to(device)
                elif hasattr(val, '__dict__'):
                    # Try to modify the nested object, but catch any errors from immutable objects
                    try:
                        obj.__dict__[key] = _move_to_device(val, device, visited, depth + 1, max_depth)
                    except (TypeError, AttributeError):
                        # If we can't modify it, just leave it as is
                        pass
        except (TypeError, AttributeError):
            # If we can't modify the object at all, just return it as is
            pass

    # Handle lists and tuples
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (nn.Module, torch.Tensor)):
                obj[i] = item.to(device)
            elif hasattr(item, '__dict__'):
                try:
                    obj[i] = _move_to_device(item, device, visited, depth + 1, max_depth)
                except (TypeError, AttributeError):
                    pass
    elif isinstance(obj, tuple):
        # Tuples are immutable, so we need to create a new one
        new_items = []
        for item in obj:
            if isinstance(item, (nn.Module, torch.Tensor)):
                new_items.append(item.to(device))
            elif hasattr(item, '__dict__'):
                try:
                    new_items.append(_move_to_device(item, device, visited, depth + 1, max_depth))
                except (TypeError, AttributeError):
                    new_items.append(item)
            else:
                new_items.append(item)
        if len(new_items) != len(obj):
            return obj  # Something went wrong, return original
        return type(obj)(new_items)  # Create a new tuple of the same type

    # Handle dictionaries
    elif isinstance(obj, dict):
        for key, val in list(obj.items()):
            if isinstance(val, (nn.Module, torch.Tensor)):
                obj[key] = val.to(device)
            elif hasattr(val, '__dict__'):
                try:
                    obj[key] = _move_to_device(val, device, visited, depth + 1, max_depth)
                except (TypeError, AttributeError):
                    pass

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
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f"cuda:{local_rank}")

    logger.debug(f"Rank {rank} using device {device} for broadcast")

    if rank == src_rank:
        # Serialize the object to a buffer
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer.seek(0)

        # Get the size of the serialized object
        size = torch.tensor(buffer.getbuffer().nbytes, dtype=torch.long).to(device)

        # Convert buffer to tensor - create on CPU first, then move to device
        data = torch.ByteTensor(list(buffer.getbuffer())).to(device)
    else:
        # Create empty tensors to receive data
        size = torch.tensor(0, dtype=torch.long).to(device)
        data = None  # Will be initialized after receiving size

    # Broadcast the size
    dist.broadcast(size, src_rank)

    # Initialize data tensor on non-source ranks
    if rank != src_rank:
        # Create on CPU first, then move to device
        data = torch.ByteTensor(size.item()).to(device)

    # Broadcast the data
    dist.broadcast(data, src_rank)

    # Deserialize on non-source ranks
    if rank != src_rank:
        # Move data to CPU for deserialization
        cpu_data = data.cpu()
        buffer = io.BytesIO(cpu_data.numpy().tobytes())
        obj = torch.load(buffer, weights_only=False)

    # Move any torch modules to the target device if specified
    if target_device is not None:
        try:
            logger.debug(f"Moving object to device: {target_device}")
            # Use the improved _move_to_device function with cycle detection
            obj = _move_to_device(obj, target_device)
        except Exception as e:
            logger.warning(f"Error moving object to device {target_device}: {e}")
            logger.warning("Continuing with object on its current device")

    logger.debug(f"Broadcast object of size {size.item()} bytes from rank {src_rank}")
    return obj
