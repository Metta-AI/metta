from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import List, Sequence

import torch

if __package__ in (None, ""):
    _FILE = Path(__file__).resolve()
    _pkg_root = _FILE.parent  # packages/cortex/benchmarks
    _parent = _pkg_root.parent  # packages/cortex
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    _src = _parent / "src"
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    __package__ = _pkg_root.name

from .common import (
    BenchmarkCase,
    BenchmarkDefinition,
    BenchmarkSettings,
    ensure_device,
    get_registry,
)


def _load_benchmarks() -> None:
    """Import all benchmark modules in this directory to populate the registry."""
    here = os.path.dirname(__file__)
    pkg = __package__ or ""
    for entry in os.listdir(here):
        if not entry.endswith(".py"):
            continue
        mod_name = entry[:-3]
        if mod_name in {"__init__", "common", "run"}:
            continue
        fullname = f"{pkg}.{mod_name}" if pkg else mod_name
        importlib.import_module(fullname)


def _format_available(registry: dict[str, BenchmarkDefinition]) -> str:
    lines = ["Available benchmarks:"]
    for key in sorted(registry):
        bench = registry[key]
        lines.append(f"  {key:<10} {bench.title}")
    return "\n".join(lines)


def _resolve_indices(requested: Sequence[int] | None, total: int) -> List[int]:
    if not requested:
        return list(range(total))
    unique = sorted(set(requested))
    for idx in unique:
        if idx < 0 or idx >= total:
            raise ValueError(f"Config index {idx} out of range (0..{total - 1})")
    return unique


def _print_device_info(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Device: CPU (Triton benchmarks will be skipped)")


def _print_header(bench: BenchmarkDefinition, device: str, warmup: int, iterations: int) -> None:
    print("=" * 90)
    print(bench.title)
    print("=" * 90)
    print()
    _print_device_info(device)
    print(f"Warmup iterations: {warmup}, Benchmark iterations: {iterations}")
    if bench.notes:
        print()
        print(bench.notes)
    print()


def _prepare_table(bench: BenchmarkDefinition, config_width: int) -> None:
    headers = ["Config"] + [col.header for col in bench.columns]
    widths = [config_width] + [max(len(col.header), 15) for col in bench.columns]
    row = " ".join(f"{header:<{width}}" for header, width in zip(headers, widths, strict=False))
    print(row)
    underline = " ".join("-" * width for width in widths)
    print(underline)


def _format_row(
    bench: BenchmarkDefinition,
    config_text: str,
    results: dict[str, object],
    widths: List[int],
) -> str:
    columns = [config_text]
    for col in bench.columns:
        value = results.get(col.key)
        if value is None:
            display = col.fallback
        else:
            display = col.formatter(value)
        columns.append(display)
    return " ".join(f"{col:<{width}}" for col, width in zip(columns, widths, strict=False))


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Cortex kernel benchmarks from a unified interface.")
    parser.add_argument("benchmark", nargs="?", help="Benchmark key to run (use --list to view all).")
    parser.add_argument("--list", action="store_true", help="List available benchmarks and exit.")
    parser.add_argument("--device", help="Force device (default: auto-detect).")
    parser.add_argument("--warmup", type=int, help="Override warmup iteration count.")
    parser.add_argument("--iterations", type=int, help="Override benchmark iteration count.")
    parser.add_argument(
        "--config",
        type=int,
        action="append",
        help="Run only the specified config index (can be provided multiple times).",
    )

    args = parser.parse_args(argv)

    # Ensure all benchmark modules are imported so their registrations run
    _load_benchmarks()
    registry = dict(get_registry())
    if not registry:
        print("No benchmarks registered.", file=sys.stderr)
        return 1

    if args.list:
        print(_format_available(registry))
        return 0

    if not args.benchmark:
        parser.print_help()
        print()
        print(_format_available(registry))
        return 1

    benchmark_key = args.benchmark.lower()
    bench = registry.get(benchmark_key)
    if bench is None:
        print(f"Unknown benchmark '{args.benchmark}'.\n", file=sys.stderr)
        print(_format_available(registry))
        return 2

    device = ensure_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is not available; falling back to CPU.", file=sys.stderr)
        device = "cpu"

    warmup = args.warmup if args.warmup is not None else bench.default_warmup
    iterations = args.iterations if args.iterations is not None else bench.default_iterations

    indices = _resolve_indices(args.config, len(bench.configs))
    cases = [BenchmarkCase(values=bench.configs[i], index=i) for i in indices]

    settings = BenchmarkSettings(device=device, warmup=warmup, iterations=iterations)

    _print_header(bench, device, warmup, iterations)

    config_texts = [f"[{case.index}] {bench.format_config(case.values)}" for case in cases]
    config_width = max(len("Config"), *(len(text) for text in config_texts)) + 2
    column_widths = [config_width] + [max(len(col.header), 15) for col in bench.columns]

    _prepare_table(bench, config_width)

    for case, config_text in zip(cases, config_texts, strict=False):
        results = bench.run_case(case, settings)
        row = _format_row(bench, config_text, results, column_widths)
        print(row)
        if "error" in results:
            print(f"{'':<{config_width}} Error: {results['error']}")

    print()
    print("=" * 90)
    print("Benchmark complete!")
    print("=" * 90)

    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
