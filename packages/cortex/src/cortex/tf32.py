from __future__ import annotations

import torch


def set_tf32_precision(mode: bool | str) -> None:
    if not torch.cuda.is_available():
        return

    enabled = mode if isinstance(mode, bool) else mode.lower() == "tf32"

    # Always use the legacy API to avoid conflicts with PyTorch's internal code
    # (e.g., torch._inductor) which still reads from allow_tf32.
    # Setting both APIs causes "mix of legacy and new APIs" errors.
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = enabled
