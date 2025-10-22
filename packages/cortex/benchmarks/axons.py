from __future__ import annotations

from typing import Dict, Optional, Tuple

import cortex.cells.core.axon_cell as cell_mod
import cortex.utils as utils_mod
import torch
from cortex.cells.core import AxonCell
from cortex.config import AxonConfig

from packages.cortex.benchmarks.common import (
    BenchmarkCase,
    BenchmarkDefinition,
    BenchmarkSettings,
    ColumnSpec,
    measure_callable,
    register,
)


def _run_cell(
    cell: AxonCell,
    x: torch.Tensor,
    resets: Optional[torch.Tensor],
    which: str,
    *,
    backend_sink: Optional[list] = None,
) -> torch.Tensor:
    """Run AxonCell with a forced backend choice.

    Note: AxonCell delegates selection to ``cortex.utils.select_backend``.
    Monkey‑patch that selector so the requested backend is authoritative.
    """
    original_utils = utils_mod.select_backend
    original_cell = getattr(cell_mod, "select_backend", None)

    def chooser(
        *,
        triton_fn=None,
        pytorch_fn=None,
        tensor=None,
        allow_triton=True,
        cuda_fn=None,
        allow_cuda=False,
    ):  # type: ignore[override]
        # Wrap chosen fn to record which backend actually executed.
        def _wrap(name: str, fn):
            if fn is None:
                return None
            printed = {"v": False}

            def _inner(*args, **kwargs):  # type: ignore[misc]
                if backend_sink is not None and not printed["v"]:
                    backend_sink[:] = [name]
                    printed["v"] = True
                return fn(*args, **kwargs)

            return _inner

        if which == "pytorch":
            return _wrap("pytorch", pytorch_fn)
        if which == "triton":
            return _wrap("triton", triton_fn)
        if which == "cuda":
            return _wrap("cuda", cuda_fn)
        return original_utils(
            triton_fn=triton_fn,
            pytorch_fn=pytorch_fn,
            tensor=tensor,
            allow_triton=allow_triton,
            cuda_fn=cuda_fn,
            allow_cuda=allow_cuda,
        )

    try:
        # Patch both the utils selector and the alias bound in the cell module.
        utils_mod.select_backend = chooser  # type: ignore[assignment]
        if original_cell is not None:
            cell_mod.select_backend = chooser  # type: ignore[assignment]
        y, _ = cell(x, state=None, resets=resets)
        return y
    finally:
        utils_mod.select_backend = original_utils  # type: ignore[assignment]
        if original_cell is not None:
            cell_mod.select_backend = original_cell  # type: ignore[assignment]


CONFIGS: Tuple[Tuple[int, int, int, bool, float], ...] = (
    (4, 128, 64, False, 0.0),
    (4, 256, 64, False, 0.0),
    (8, 256, 64, False, 0.0),
    (8, 512, 64, False, 0.0),
    (8, 512, 128, False, 0.0),
    (8, 512, 64, True, 0.1),
    (8, 1024, 64, True, 0.1),
)


def _format_config(config: Tuple[int, int, int, bool, float]) -> str:
    b, t, h, use_resets, prob = config
    return f"({b}, {t}, {h}, {use_resets}, {prob})"


def _run_case(case: BenchmarkCase, settings: BenchmarkSettings) -> Dict[str, object]:
    batch_size, seq_len, hidden_size, with_resets, reset_prob = case.values
    device = torch.device(settings.device)
    dtype = settings.dtype

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    resets = None
    if with_resets:
        resets = (torch.rand(batch_size, seq_len, device=device) < reset_prob).to(device=device)

    cell = AxonCell(AxonConfig(hidden_size=hidden_size, activation="SiLU")).to(device=device, dtype=dtype)

    synchronize = device.type == "cuda"

    pt_sink: list = []

    def run_pytorch():
        return _run_cell(cell, x, resets, which="pytorch", backend_sink=pt_sink)

    output_pt, pytorch_time = measure_callable(
        run_pytorch,
        warmup=settings.warmup,
        iterations=settings.iterations,
        synchronize=synchronize,
    )

    results: Dict[str, object] = {
        "pytorch_ms": pytorch_time * 1000.0,
        "triton_ms": None,
        "cuda_ms": None,
        "speedup": None,
        "speedup_cuda": None,
        "max_diff": None,
        "max_diff_cuda": None,
        "pt_backend": pt_sink[0] if pt_sink else None,
        "triton_backend": None,
        "cuda_backend": None,
    }

    if device.type != "cuda":
        return results

    tr_sink: list = []

    def run_triton():
        return _run_cell(cell, x, resets, which="triton", backend_sink=tr_sink)

    output_tr, triton_time = measure_callable(
        run_triton,
        warmup=settings.warmup,
        iterations=settings.iterations,
        synchronize=True,
    )
    results["triton_ms"] = triton_time * 1000.0
    results["triton_backend"] = tr_sink[0] if tr_sink else None
    if triton_time > 0:
        results["speedup"] = pytorch_time / triton_time
    results["max_diff"] = torch.max(torch.abs(output_pt - output_tr)).item()

    # Try CUDA backend explicitly
    cu_sink: list = []

    def run_cuda():
        return _run_cell(cell, x, resets, which="cuda", backend_sink=cu_sink)

    output_cu, cuda_time = measure_callable(
        run_cuda,
        warmup=settings.warmup,
        iterations=settings.iterations,
        synchronize=True,
    )
    results["cuda_ms"] = cuda_time * 1000.0
    results["cuda_backend"] = cu_sink[0] if cu_sink else None
    if cuda_time > 0:
        results["speedup_cuda"] = pytorch_time / cuda_time
    results["max_diff_cuda"] = torch.max(torch.abs(output_pt - output_cu)).item()

    return results


register(
    BenchmarkDefinition(
        key="axons",
        title="Axons (streaming RTU) PyTorch vs Triton vs CUDA",
        description=("Benchmark AxonCell with forced backend selection, reporting PyTorch, Triton, and CUDA timings."),
        configs=CONFIGS,
        format_config=_format_config,
        run_case=_run_case,
        notes=(
            "Uses a selector override to force backend choice. Triton/CUDA require a CUDA device; "
            "columns report both speedups against PyTorch and max abs diff vs PyTorch outputs."
        ),
        columns=(
            ColumnSpec("pytorch_ms", "PyTorch (ms)", lambda v: f"{float(v):.3f}"),
            ColumnSpec("triton_ms", "Triton (ms)", lambda v: f"{float(v):.3f}"),
            ColumnSpec("cuda_ms", "CUDA (ms)", lambda v: f"{float(v):.3f}"),
            ColumnSpec("speedup", "TRT/PT", lambda v: f"{float(v):.2f}x"),
            ColumnSpec("speedup_cuda", "CUDA/PT", lambda v: f"{float(v):.2f}x"),
            ColumnSpec("max_diff", "Max Diff TRT", lambda v: f"{float(v):.2e}"),
            ColumnSpec("max_diff_cuda", "Max Diff CUDA", lambda v: f"{float(v):.2e}"),
            ColumnSpec("pt_backend", "PT backend", lambda v: str(v) if v is not None else "-"),
            ColumnSpec("triton_backend", "TRT backend", lambda v: str(v) if v is not None else "-"),
            ColumnSpec("cuda_backend", "CUDA backend", lambda v: str(v) if v is not None else "-"),
        ),
    )
)
