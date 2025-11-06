import typing

import cortex.kernels.pytorch.mlstm
import cortex.kernels.triton.mlstm
import torch

import packages.cortex.benchmarks.common

CONFIGS: typing.Tuple[typing.Tuple[int, int, int, int, int], ...] = (
    (2, 4, 128, 64, 64),
    (2, 4, 256, 64, 64),
    (2, 4, 512, 64, 64),
    (4, 8, 128, 64, 64),
    (4, 8, 256, 64, 64),
    (4, 8, 512, 64, 64),
    (8, 8, 256, 64, 64),
    (8, 8, 512, 64, 64),
    (8, 8, 1024, 64, 64),
)


def _format_config(config: typing.Tuple[int, int, int, int, int]) -> str:
    b, h, t, d, chunk = config
    return f"({b}, {h}, {t}, {d}, {chunk})"


def _run_case(
    case: packages.cortex.benchmarks.common.BenchmarkCase, settings: packages.cortex.benchmarks.common.BenchmarkSettings
) -> typing.Dict[str, object]:
    batch_size, num_heads, seq_len, head_dim, chunk_size = case.values
    device = torch.device(settings.device)
    dtype = settings.dtype

    queries = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    keys = torch.randn_like(queries)
    values = torch.randn_like(queries)
    igate_preact = torch.randn(batch_size, num_heads, seq_len, device=device, dtype=dtype)
    fgate_preact = torch.randn(batch_size, num_heads, seq_len, device=device, dtype=dtype)

    synchronize = device.type == "cuda"

    def run_pytorch():
        return cortex.kernels.pytorch.mlstm.mlstm_chunkwise_simple(
            queries=queries,
            keys=keys,
            values=values,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            chunk_size=chunk_size,
        )

    output_pt, pytorch_time = packages.cortex.benchmarks.common.measure_callable(
        run_pytorch,
        warmup=settings.warmup,
        iterations=settings.iterations,
        synchronize=synchronize,
    )

    results: typing.Dict[str, object] = {
        "pytorch_ms": pytorch_time * 1000.0,
        "triton_ms": None,
        "speedup": None,
        "max_diff": None,
    }

    if device.type != "cuda":
        return results

    def run_triton():
        return cortex.kernels.triton.mlstm.mlstm_chunkwise_triton(
            queries=queries,
            keys=keys,
            values=values,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            chunk_size=chunk_size,
        )

    output_tr, triton_time = packages.cortex.benchmarks.common.measure_callable(
        run_triton,
        warmup=settings.warmup,
        iterations=settings.iterations,
        synchronize=True,
    )
    results["triton_ms"] = triton_time * 1000.0
    if triton_time > 0:
        results["speedup"] = pytorch_time / triton_time
    results["max_diff"] = torch.max(torch.abs(output_pt - output_tr)).item()

    return results


packages.cortex.benchmarks.common.register(
    packages.cortex.benchmarks.common.BenchmarkDefinition(
        key="mlstm",
        title="mLSTM Triton vs PyTorch Benchmark",
        description="Compare Triton-accelerated mLSTM chunkwise kernels against PyTorch implementations.",
        configs=CONFIGS,
        format_config=_format_config,
        run_case=_run_case,
        notes="Chunk size can be tuned; defaults mimic the previous benchmark suite.",
    )
)
