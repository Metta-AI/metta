from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    header: str
    formatter: Callable[[Any], str]
    fallback: str = "N/A"


DEFAULT_COLUMNS: Tuple[ColumnSpec, ...] = (
    ColumnSpec("pytorch_ms", "PyTorch (ms)", lambda v: f"{float(v):.3f}"),
    ColumnSpec("triton_ms", "Triton (ms)", lambda v: f"{float(v):.3f}"),
    ColumnSpec("speedup", "Speedup", lambda v: f"{float(v):.2f}x"),
    ColumnSpec("max_diff", "Max Diff", lambda v: f"{float(v):.2e}"),
)


@dataclass(frozen=True)
class BenchmarkDefinition:
    key: str
    title: str
    description: str
    configs: Sequence[Any]
    format_config: Callable[[Any], str]
    run_case: Callable[["BenchmarkCase", "BenchmarkSettings"], Dict[str, Any]]
    notes: Optional[str] = None
    columns: Sequence[ColumnSpec] = DEFAULT_COLUMNS
    default_warmup: int = 5
    default_iterations: int = 20


@dataclass(frozen=True)
class BenchmarkCase:
    values: Any
    index: int


@dataclass
class BenchmarkSettings:
    device: str
    warmup: int
    iterations: int
    dtype: torch.dtype = torch.float32


_REGISTRY: Dict[str, BenchmarkDefinition] = {}


def register(definition: BenchmarkDefinition) -> BenchmarkDefinition:
    if definition.key in _REGISTRY:
        raise ValueError(f"Benchmark key '{definition.key}' already registered")
    _REGISTRY[definition.key] = definition
    return definition


def get_registry() -> Mapping[str, BenchmarkDefinition]:
    return _REGISTRY


def measure_callable(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    synchronize: bool,
) -> Tuple[Any, float]:
    last: Any = None
    for _ in range(max(0, warmup)):
        last = fn()
    if synchronize:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max(1, iterations)):
        last = fn()
    if synchronize:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return last, elapsed / max(1, iterations)


def ensure_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
