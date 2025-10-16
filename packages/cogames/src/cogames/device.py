"""Utility functions for CoGames CLI."""

import torch
from rich.console import Console


def resolve_training_device(console: Console, requested: str) -> torch.device:
    normalized = requested.strip().lower()

    def cuda_usable() -> bool:
        cuda_backend = getattr(torch.backends, "cuda", None)
        if cuda_backend is None or not cuda_backend.is_built():
            return False
        if not hasattr(torch._C, "_cuda_getDeviceCount"):
            return False
        return torch.cuda.is_available()

    if normalized == "auto":
        if cuda_usable():
            return torch.device("cuda")
        console.print("[yellow]CUDA not available; falling back to CPU for training.[/yellow]")
        return torch.device("cpu")

    try:
        candidate = torch.device(requested)
    except (RuntimeError, ValueError):
        console.print(f"[yellow]Warning: Unknown device '{requested}'. Falling back to CPU.[/yellow]")
        return torch.device("cpu")

    if candidate.type == "cuda" and not cuda_usable():
        console.print("[yellow]CUDA requested but unavailable. Training will run on CPU instead.[/yellow]")
        return torch.device("cpu")

    return candidate
