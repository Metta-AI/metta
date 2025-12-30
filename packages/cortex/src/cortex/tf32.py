from __future__ import annotations

import torch


def set_tf32_precision(mode: bool | str) -> None:
    if not torch.cuda.is_available():
        return

    enabled = mode if isinstance(mode, bool) else mode.lower() == "tf32"
    # Keep float32 matmul precision consistent with TF32 backend flags.
    #
    # TorchDynamo (and other internals) may still query the legacy cuBLAS TF32
    # getter; if `torch.backends.cuda.matmul.fp32_precision` enables TF32 while
    # the global matmul precision remains at the default ("highest"), PyTorch
    # raises an error about mixing legacy and new TF32 APIs.
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enabled else "highest")
    matmul_has_fp32 = hasattr(torch.backends.cuda.matmul, "fp32_precision")
    cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
    cudnn_has_fp32 = cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision")

    if matmul_has_fp32:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
    if cudnn_has_fp32:
        cudnn_conv.fp32_precision = "tf32" if enabled else "ieee"

    if not matmul_has_fp32 and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = enabled
    if not cudnn_has_fp32 and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = enabled
