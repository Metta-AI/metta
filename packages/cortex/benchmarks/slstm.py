import typing

import cortex.kernels.pytorch.slstm
import cortex.kernels.triton.slstm
import torch

import packages.cortex.benchmarks.common

CONFIGS: typing.Tuple[typing.Tuple[int, int, int, int], ...] = (
    (2, 4, 128, 64),
    (2, 4, 256, 64),
    (2, 4, 512, 64),
    (4, 8, 128, 64),
    (4, 8, 256, 64),
    (4, 8, 512, 64),
    (8, 8, 256, 64),
    (8, 8, 512, 64),
    (8, 8, 1024, 64),
    (4, 4, 512, 128),
    (8, 4, 512, 128),
)


def _format_config(config: typing.Tuple[int, int, int, int]) -> str:
    b, heads, t, d = config
    return f"({b}, {heads}, {t}, {d})"


def _run_case(
    case: packages.cortex.benchmarks.common.BenchmarkCase, settings: packages.cortex.benchmarks.common.BenchmarkSettings
) -> typing.Dict[str, object]:
    batch_size, num_heads, seq_len, head_dim = case.values
    device = torch.device(settings.device)
    dtype = settings.dtype

    Wx = torch.randn(batch_size, seq_len, 4, num_heads, head_dim, device=device, dtype=dtype)
    R = torch.randn(4, num_heads, head_dim, head_dim, device=device, dtype=dtype)
    b = torch.randn(4, num_heads, head_dim, device=device, dtype=dtype)
    initial_states = torch.randn(4, batch_size, num_heads, head_dim, device=device, dtype=dtype)

    synchronize = device.type == "cuda"

    def run_pytorch():
        return cortex.kernels.pytorch.slstm.slstm_sequence_pytorch(Wx=Wx, R=R, b=b, initial_states=initial_states)

    (all_states_pt, last_state_pt), pytorch_time = packages.cortex.benchmarks.common.measure_callable(
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
        return cortex.kernels.triton.slstm.slstm_sequence_triton(Wx=Wx, R=R, b=b, initial_states=initial_states)

    (all_states_tr, last_state_tr), triton_time = packages.cortex.benchmarks.common.measure_callable(
        run_triton,
        warmup=settings.warmup,
        iterations=settings.iterations,
        synchronize=True,
    )
    results["triton_ms"] = triton_time * 1000.0
    if triton_time > 0:
        results["speedup"] = pytorch_time / triton_time
    diff_states = torch.max(torch.abs(all_states_pt - all_states_tr)).item()
    diff_last = torch.max(torch.abs(last_state_pt - last_state_tr)).item()
    results["max_diff"] = max(diff_states, diff_last)

    return results


packages.cortex.benchmarks.common.register(
    packages.cortex.benchmarks.common.BenchmarkDefinition(
        key="slstm",
        title="sLSTM Triton vs PyTorch Benchmark",
        description="Compare Triton-accelerated sLSTM kernels against PyTorch implementations.",
        configs=CONFIGS,
        format_config=_format_config,
        run_case=_run_case,
        notes="Head dimension must be a power of 2 for the Triton kernel.",
    )
)
