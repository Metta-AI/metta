from __future__ import annotations

import torch


def set_tf32_precision(mode: bool | str) -> None:
    if not torch.cuda.is_available():
        return

    enabled = mode if isinstance(mode, bool) else mode.lower() == "tf32"

    # For now, always use the legacy allow_tf32 API as torch._inductor appears to have an issue with the newer tf32 API
    #  and we get a "mix of legacy and new APIs" with certain Cortex cells.
    # We can try uncommenting the code below later if the issue is fixed.
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = enabled

    # Also set new API if available - keeps both in sync, avoids "mixing" errors
    # when PyTorch eventually updates inductor to use the new API
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32" if enabled else "ieee"

    # enabled = mode if isinstance(mode, bool) else mode.lower() == "tf32"
    # matmul_has_fp32 = hasattr(torch.backends.cuda.matmul, "fp32_precision")
    # cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
    # cudnn_has_fp32 = cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision")
    #
    # if matmul_has_fp32:
    #     torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
    # if cudnn_has_fp32:
    #     cudnn_conv.fp32_precision = "tf32" if enabled else "ieee"
    #
    # if not matmul_has_fp32 and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    #     torch.backends.cuda.matmul.allow_tf32 = enabled
    # if not cudnn_has_fp32 and hasattr(torch.backends.cudnn, "allow_tf32"):
    #     torch.backends.cudnn.allow_tf32 = enabled
