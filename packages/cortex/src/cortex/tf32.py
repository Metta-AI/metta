from __future__ import annotations

import torch


def set_tf32_precision(mode: bool | str) -> None:
    if not torch.cuda.is_available():
        return

    enabled = mode if isinstance(mode, bool) else mode.lower() == "tf32"

    # For now, we ALWAYS use the legacy allow_tf32 API as torch._inductor appears to have an issue with the newer
    #  fp32 tf32 API: https://github.com/pytorch/pytorch/issues/166387
    # Trying to only use the new API led to a "mix of legacy and new APIs" RuntimeError() on compilation of
    #  certain Cortex cells.
    # When PyTorch removes support for legacy API, we can move to the new API, and hopefully, the bug is fixed by then.
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = enabled

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
