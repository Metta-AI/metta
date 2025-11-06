
import dataclasses
import time
import typing

import torch


@dataclasses.dataclass(frozen=True)
class ColumnSpec:
    key: str
    header: str
    formatter: typing.Callable[[typing.Any], str]
    fallback: str = "N/A"


DEFAULT_COLUMNS: typing.Tuple[ColumnSpec, ...] = (
    ColumnSpec("pytorch_ms", "PyTorch (ms)", lambda v: f"{float(v):.3f}"),
    ColumnSpec("triton_ms", "Triton (ms)", lambda v: f"{float(v):.3f}"),
    ColumnSpec("speedup", "Speedup", lambda v: f"{float(v):.2f}x"),
    ColumnSpec("max_diff", "Max Diff", lambda v: f"{float(v):.2e}"),
)


@dataclasses.dataclass(frozen=True)
class BenchmarkDefinition:
    key: str
    title: str
    description: str
    configs: typing.Sequence[typing.Any]
    format_config: typing.Callable[[typing.Any], str]
    run_case: typing.Callable[["BenchmarkCase", "BenchmarkSettings"], typing.Dict[str, typing.Any]]
    notes: typing.Optional[str] = None
    columns: typing.Sequence[ColumnSpec] = DEFAULT_COLUMNS
    default_warmup: int = 5
    default_iterations: int = 20


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    values: typing.Any
    index: int


@dataclasses.dataclass
class BenchmarkSettings:
    device: str
    warmup: int
    iterations: int
    dtype: torch.dtype = torch.float32


_REGISTRY: typing.Dict[str, BenchmarkDefinition] = {}


def register(definition: BenchmarkDefinition) -> BenchmarkDefinition:
    if definition.key in _REGISTRY:
        raise ValueError(f"Benchmark key '{definition.key}' already registered")
    _REGISTRY[definition.key] = definition
    return definition


def get_registry() -> typing.Mapping[str, BenchmarkDefinition]:
    return _REGISTRY


def measure_callable(
    fn: typing.Callable[[], typing.Any],
    *,
    warmup: int,
    iterations: int,
    synchronize: bool,
) -> typing.Tuple[typing.Any, float]:
    last: typing.Any = None
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


def ensure_device(device: typing.Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
