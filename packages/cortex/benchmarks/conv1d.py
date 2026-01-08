from __future__ import annotations

from typing import Dict, Tuple, cast

import torch
import torch.nn as nn
from cortex.kernels.pytorch.conv1d import causal_conv1d_pytorch
from cortex.kernels.triton.conv1d import causal_conv1d_triton

from .common import (
    BenchmarkCase,
    BenchmarkDefinition,
    BenchmarkSettings,
    measure_callable,
    register,
)

CONFIGS: Tuple[Tuple[int, int, int, int], ...] = (
    (2, 128, 256, 4),
    (2, 256, 256, 4),
    (2, 512, 256, 4),
    (4, 128, 512, 4),
    (4, 256, 512, 4),
    (4, 512, 512, 4),
    (8, 256, 512, 4),
    (8, 512, 512, 4),
    (8, 1024, 512, 4),
    (4, 512, 512, 8),
    (8, 512, 512, 8),
)


def _format_config(config: Tuple[int, int, int, int]) -> str:
    b, t, features, kernel = config
    return f"({b}, {t}, {features}, {kernel})"


def _run_case(case: BenchmarkCase, settings: BenchmarkSettings) -> Dict[str, object]:
    batch_size, seq_len, features, kernel_size = case.values
    device = torch.device(settings.device)
    dtype = settings.dtype

    x = torch.randn(batch_size, seq_len, features, device=device, dtype=dtype)
    conv_state = torch.randn(batch_size, kernel_size, features, device=device, dtype=dtype)
    weight = torch.randn(features, features, kernel_size, device=device, dtype=dtype)
    bias = torch.randn(features, device=device, dtype=dtype)
    resets = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

    groups = 1
    pad = kernel_size - 1

    conv = nn.Conv1d(
        in_channels=features,
        out_channels=features,
        kernel_size=kernel_size,
        padding=pad,
        groups=groups,
        bias=True,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        conv_weight = cast(torch.Tensor, conv.weight)
        conv_bias = cast(torch.Tensor, conv.bias)
        conv_weight.copy_(weight)
        conv_bias.copy_(bias)

    synchronize = device.type == "cuda"

    def run_pytorch():
        return causal_conv1d_pytorch(
            conv_state=conv_state.clone(),
            x=x,
            weight=weight,
            bias=bias,
            groups=groups,
            pad=pad,
            conv=conv,
            resets=None,
        )

    (y_pt, state_pt), pytorch_time = measure_callable(
        run_pytorch,
        warmup=settings.warmup,
        iterations=settings.iterations,
        synchronize=synchronize,
    )

    results: Dict[str, object] = {
        "pytorch_ms": pytorch_time * 1000.0,
        "triton_ms": None,
        "speedup": None,
        "max_diff": None,
    }

    if device.type != "cuda":
        return results

    def run_triton():
        return causal_conv1d_triton(
            conv_state=conv_state.clone(),
            x=x,
            weight=weight,
            bias=bias,
            groups=groups,
            resets=resets,
        )

    (y_tr, state_tr), triton_time = measure_callable(
        run_triton,
        warmup=settings.warmup,
        iterations=settings.iterations,
        synchronize=True,
    )
    results["triton_ms"] = triton_time * 1000.0
    if triton_time > 0:
        results["speedup"] = pytorch_time / triton_time
    results["max_diff"] = torch.max(torch.abs(y_pt - y_tr)).item()

    return results


register(
    BenchmarkDefinition(
        key="conv1d",
        title="Conv1D Triton vs PyTorch Benchmark",
        description="Benchmark causal Conv1D Triton kernels against the PyTorch fast path.",
        configs=CONFIGS,
        format_config=_format_config,
        run_case=_run_case,
        notes="Triton kernel expects channel-mixing mode; resets are passed but zero by default.",
    )
)
