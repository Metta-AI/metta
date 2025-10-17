from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from cortex.kernels.pytorch.lstm import lstm_sequence_pytorch
from cortex.kernels.triton.lstm import lstm_sequence_triton

from packages.cortex.benchmarks.common import (
    BenchmarkCase,
    BenchmarkDefinition,
    BenchmarkSettings,
    measure_callable,
    register,
)

CONFIGS: Tuple[Tuple[int, int, int], ...] = (
    (16, 128, 64),
    (16, 256, 64),
    (16, 512, 64),
    (32, 128, 64),
    (32, 256, 64),
    (32, 512, 64),
    (16, 256, 128),
    (32, 256, 128),
    (16, 512, 128),
    (32, 512, 128),
)


def _format_config(config: Tuple[int, int, int]) -> str:
    batch_size, seq_len, hidden_size = config
    return f"({batch_size}, {seq_len}, {hidden_size})"


def _run_case(case: BenchmarkCase, settings: BenchmarkSettings) -> Dict[str, object]:
    batch_size, seq_len, hidden_size = case.values
    device = torch.device(settings.device)
    dtype = settings.dtype

    lstm = nn.LSTM(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=True,
    ).to(device=device, dtype=dtype)

    x_seq = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    h0_bf = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)
    c0_bf = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)

    synchronize = device.type == "cuda"

    def run_pytorch():
        return lstm_sequence_pytorch(
            lstm=lstm,
            x_seq=x_seq,
            h0_bf=h0_bf,
            c0_bf=c0_bf,
            resets=None,
        )

    (y_pt, hn_pt, cn_pt), pytorch_time = measure_callable(
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
        return lstm_sequence_triton(
            lstm=lstm,
            x_seq=x_seq,
            h0_bf=h0_bf,
            c0_bf=c0_bf,
            resets=None,
        )

    try:
        (y_tr, hn_tr, cn_tr), triton_time = measure_callable(
            run_triton,
            warmup=settings.warmup,
            iterations=settings.iterations,
            synchronize=True,
        )
        results["triton_ms"] = triton_time * 1000.0
        if triton_time > 0:
            results["speedup"] = pytorch_time / triton_time
        max_diff = max(
            torch.max(torch.abs(y_pt - y_tr)).item(),
            torch.max(torch.abs(hn_pt - hn_tr)).item(),
            torch.max(torch.abs(cn_pt - cn_tr)).item(),
        )
        results["max_diff"] = max_diff
    except Exception as exc:  # pragma: no cover - defensive
        results["error"] = str(exc)

    return results


register(
    BenchmarkDefinition(
        key="lstm",
        title="LSTM Triton vs PyTorch Benchmark",
        description="Compare Triton-accelerated LSTM kernels against PyTorch implementations.",
        configs=CONFIGS,
        format_config=_format_config,
        run_case=_run_case,
        notes="hidden_size must be a power of 2 for the Triton kernel and batch size >= 16.",
    )
)
