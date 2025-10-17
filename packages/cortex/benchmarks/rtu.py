from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from cortex.cells.rtu import RTUCell
from cortex.config import RTUCellConfig

from packages.cortex.benchmarks.common import (
    BenchmarkCase,
    BenchmarkDefinition,
    BenchmarkSettings,
    measure_callable,
    register,
)


def _run_cell(cell: RTUCell, x: torch.Tensor, resets: Optional[torch.Tensor], which: str) -> torch.Tensor:
    import cortex.cells.rtu as cell_mod

    original = cell_mod.select_backend

    def chooser(*, triton_fn, pytorch_fn, tensor, allow_triton=True):  # type: ignore[override]
        return triton_fn if which == "triton" else pytorch_fn

    try:
        cell_mod.select_backend = chooser  # type: ignore[assignment]
        y, _ = cell(x, state=None, resets=resets)
        return y
    finally:
        cell_mod.select_backend = original  # type: ignore[assignment]


CONFIGS: Tuple[Tuple[int, int, int, int, bool, float], ...] = (
    (4, 128, 64, 8, False, 0.0),
    (4, 256, 64, 8, False, 0.0),
    (8, 256, 64, 16, False, 0.0),
    (8, 512, 64, 16, False, 0.0),
    (8, 512, 128, 16, False, 0.0),
    (8, 512, 64, 16, True, 0.1),
    (8, 1024, 64, 16, True, 0.1),
)


def _format_config(config: Tuple[int, int, int, int, bool, float]) -> str:
    b, t, h, r, use_resets, prob = config
    return f"({b}, {t}, {h}, {r}, {use_resets}, {prob})"


def _run_case(case: BenchmarkCase, settings: BenchmarkSettings) -> Dict[str, object]:
    batch_size, seq_len, hidden_size, rank, with_resets, reset_prob = case.values
    device = torch.device(settings.device)
    dtype = settings.dtype

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    resets = None
    if with_resets:
        resets = (torch.rand(batch_size, seq_len, device=device) < reset_prob).to(device=device)

    cell = RTUCell(
        RTUCellConfig(hidden_size=hidden_size, rank=rank, activation="SiLU"),
    ).to(device=device, dtype=dtype)

    synchronize = device.type == "cuda"

    def run_pytorch():
        return _run_cell(cell, x, resets, which="pytorch")

    output_pt, pytorch_time = measure_callable(
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
        return _run_cell(cell, x, resets, which="triton")

    try:
        output_tr, triton_time = measure_callable(
            run_triton,
            warmup=settings.warmup,
            iterations=settings.iterations,
            synchronize=True,
        )
        results["triton_ms"] = triton_time * 1000.0
        if triton_time > 0:
            results["speedup"] = pytorch_time / triton_time
        results["max_diff"] = torch.max(torch.abs(output_pt - output_tr)).item()
    except Exception as exc:  # pragma: no cover - defensive
        results["error"] = str(exc)

    return results


register(
    BenchmarkDefinition(
        key="rtu",
        title="RTU (low-rank) Triton vs PyTorch Benchmark",
        description="Benchmark the RTUCell with forced backend selection to compare Triton and PyTorch paths.",
        configs=CONFIGS,
        format_config=_format_config,
        run_case=_run_case,
        notes="Includes segmented-scan path by enabling resets in selected configurations.",
    )
)
