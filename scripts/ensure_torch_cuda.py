#!/usr/bin/env python3
"""Reinstall PyTorch with CUDA kernels when GPUs are present."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib import import_module, metadata

from packaging.version import Version

CUDA_TAGS = {
    "11.8": ["cu118"],
    "12.1": ["cu121"],
    "12.2": ["cu122", "cu121"],
    "12.3": ["cu122"],
    "12.4": ["cu124", "cu122"],
    "12.5": ["cu124"],
    "12.6": ["cu126", "cu124"],
    "12.7": ["cu126"],
    "12.8": ["cu126", "cu124"],
}
FALLBACK_TAGS = ["cu128", "cu126", "cu124", "cu122", "cu121", "cu118"]
SKIP_ENV = "METTA_SKIP_TORCH_CUDA_FIX"
TAG_ENV = "METTA_TORCH_CUDA_TAG"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="reinstall even if torch looks usable")
    parser.add_argument("--quiet", action="store_true", help="suppress informational logs")
    return parser.parse_args()


def log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[torch-cuda] {message}")


def detect_gpus(torch_mod: object) -> set[str]:
    caps: set[str] = set()
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is not None:
        try:
            for idx in range(cuda.device_count()):
                major, minor = cuda.get_device_capability(idx)
                caps.add(f"sm_{major}{minor}")
        except Exception:
            caps.clear()
    if caps:
        return caps
    if shutil.which("nvidia-smi") is None:
        return set()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return set()
    for line in result.stdout.splitlines():
        parts = [p for p in line.strip().split(".") if p]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            caps.add(f"sm_{parts[0]}{parts[1]}")
    return caps


def wheel_ok(torch_mod: object, gpu_caps: set[str]) -> tuple[bool, str]:
    cuda_version = getattr(torch_mod.version, "cuda", None)
    cuda_available = getattr(torch_mod.cuda, "is_available", lambda: False)()
    compiled = {arch.lower() for arch in getattr(torch_mod.cuda, "get_arch_list", lambda: [])()}
    if not gpu_caps:
        return True, "no GPUs detected"
    if cuda_version is None:
        return False, "CPU-only PyTorch wheel"
    if not cuda_available:
        return False, "torch.cuda.is_available() is False"
    missing = {cap for cap in gpu_caps if cap not in compiled}
    if missing:
        return False, f"wheel missing {', '.join(sorted(missing))}"
    return True, "current torch wheel already targets GPUs"


def pick_tags(cuda_version: str | None) -> list[str]:
    forced = os.getenv(TAG_ENV)
    if forced:
        return [forced]
    tags = CUDA_TAGS.get(cuda_version or "", []) + FALLBACK_TAGS
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def install(tag: str, version: Version, *, quiet: bool) -> bool:
    target = f"{version.base_version}+{tag}"
    cmd = [
        "uv",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "--index-url",
        f"https://download.pytorch.org/whl/{tag}",
        "--extra-index-url",
        "https://pypi.org/simple",
        f"torch=={target}",
    ]
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=quiet)
        return True
    except subprocess.CalledProcessError as exc:
        log(f"Failed to install torch=={target} ({exc}).", quiet=quiet)
        return False


def verify(gpu_caps: set[str], *, quiet: bool) -> bool:
    script = f"""
import torch
caps = {sorted(gpu_caps)!r}
cuda_version = getattr(torch.version, 'cuda', None)
available = getattr(torch.cuda, 'is_available', lambda: False)()
archs = {{a.lower() for a in getattr(torch.cuda, 'get_arch_list', lambda: [])()}}
missing = [cap for cap in caps if cap not in archs]
raise SystemExit(0 if caps and cuda_version and available and not missing else 1)
"""
    try:
        subprocess.run([sys.executable, "-c", script], check=True, text=True, capture_output=quiet)
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    args = parse_args()

    if os.getenv(SKIP_ENV, "").lower() in {"1", "true", "yes"}:
        log(f"Skipping because {SKIP_ENV} is set.", quiet=args.quiet)
        return 0

    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        log("PyTorch not installed; skipping.", quiet=args.quiet)
        return 0

    try:
        torch_mod = import_module("torch")
    except ImportError as exc:
        log(f"Cannot import torch ({exc}); skipping.", quiet=args.quiet)
        return 0

    gpu_caps = detect_gpus(torch_mod)
    if not gpu_caps:
        log("No NVIDIA GPUs detected; leaving torch as-is.", quiet=args.quiet)
        return 0

    ok, reason = wheel_ok(torch_mod, gpu_caps)
    if ok and not args.force:
        log(reason, quiet=args.quiet)
        return 0

    log(f"Reinstalling torch {version} for CUDA ({reason}).")
    for tag in pick_tags(getattr(torch_mod.version, "cuda", None)):
        if install(tag, Version(version), quiet=args.quiet) and verify(gpu_caps, quiet=args.quiet):
            log(f"Installed torch=={Version(version).base_version}+{tag} with CUDA support.")
            return 0

    log("Unable to install a matching CUDA wheel automatically. Set METTA_TORCH_CUDA_TAG or install manually.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
