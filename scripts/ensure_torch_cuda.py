#!/usr/bin/env python3
"""Ensure the installed PyTorch wheel contains CUDA kernels for local GPUs.

The default PyPI wheels that `uv sync` installs are CPU-only. On Linux hosts
with NVIDIA GPUs (e.g. RTX 4090 / sm_89) these wheels emit warnings and force
`system.device=cpu`. This helper detects that situation and reinstalls the
matching CUDA wheel from the official PyTorch index so that training runs can
use the GPU without manual intervention.

The script is intentionally idempotent and safe to call on every `metta
install`. It only modifies the environment when a GPU is present and the
current wheel is missing CUDA support (e.g. CPU wheel or missing architectures).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence

from importlib import metadata
from packaging.version import Version


CUDA_TAG_PREFERENCES: dict[str, list[str]] = {
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

DEFAULT_TAG_PRIORITY = ["cu128", "cu126", "cu124", "cu122", "cu121", "cu118"]
SKIP_ENV = "METTA_SKIP_TORCH_CUDA_FIX"
TAG_OVERRIDE_ENV = "METTA_TORCH_CUDA_TAG"


@dataclass
class TorchState:
    version: Version
    cuda_version: str | None
    compiled_arches: set[str]
    gpu_arches: set[str]
    cuda_available: bool

    def missing_arches(self) -> set[str]:
        return {cap for cap in self.gpu_arches if cap not in self.compiled_arches}

    def needs_reinstall(self) -> tuple[bool, str]:
        if not self.gpu_arches:
            return False, "no NVIDIA GPU detected"
        if self.cuda_version is None:
            return True, "PyTorch CPU-only wheel detected"
        if not self.cuda_available:
            return True, "torch.cuda.is_available() is False"
        missing = self.missing_arches()
        if missing:
            return True, f"CUDA wheel missing {', '.join(sorted(missing))}"
        return False, "current wheel already supports detected GPUs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="suppress no-op logs")
    parser.add_argument("--force", action="store_true", help="force reinstall even if wheel looks OK")
    return parser.parse_args()


def info(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(f"[torch-cuda] {message}")


def warn(message: str) -> None:
    print(f"[torch-cuda] {message}")


def run(cmd: Sequence[str], *, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=quiet, check=True)


def detect_cuda_version(torch_cuda_version: str | None) -> str | None:
    if torch_cuda_version:
        return torch_cuda_version
    smi_output = query_nvidia_smi(["cuda_version"])
    if smi_output:
        return smi_output[0]
    return None


def query_nvidia_smi(fields: list[str]) -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []

    cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError:
        return []

    values: list[str] = []
    for line in result.stdout.strip().splitlines():
        val = line.strip()
        if not val or val == "N/A":
            continue
        values.append(val)
    return values


def detect_gpu_arches(torch_module: object) -> set[str]:
    arches = set(_gpu_arches_from_smi())
    return arches or set(_gpu_arches_from_torch(torch_module))


def _gpu_arches_from_smi() -> list[str]:
    caps = query_nvidia_smi(["compute_cap"])
    arches: list[str] = []
    for cap in caps:
        parts = cap.split(".")
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            continue
        arches.append(f"sm_{major}{minor}")
    return arches


def _gpu_arches_from_torch(torch_module: object) -> list[str]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return []
    try:
        count = cuda.device_count()
    except Exception:
        return []

    arches: list[str] = []
    for idx in range(count):
        try:
            major, minor = cuda.get_device_capability(idx)
        except Exception:
            continue
        arches.append(f"sm_{major}{minor}")
    return arches


def compiled_arches(torch_module: object) -> set[str]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return set()
    get_arch_list = getattr(cuda, "get_arch_list", None)
    if get_arch_list is None:
        return set()
    try:
        return {arch.lower() for arch in get_arch_list()}
    except Exception:
        return set()


def current_torch_state() -> TorchState | None:
    try:
        version_str = metadata.version("torch")
    except metadata.PackageNotFoundError:
        warn("PyTorch is not installed; skipping CUDA wheel detection.")
        return None

    import importlib

    try:
        torch_module = importlib.import_module("torch")
    except ImportError as exc:
        warn(f"Unable to import torch ({exc}); skipping CUDA wheel detection.")
        return None
    version = Version(version_str)
    cuda_version = detect_cuda_version(getattr(torch_module.version, "cuda", None))
    arches = detect_gpu_arches(torch_module)
    compiled = compiled_arches(torch_module)
    cuda_available = False
    cuda_attr = getattr(torch_module, "cuda", None)
    if cuda_attr is not None:
        try:
            cuda_available = bool(cuda_attr.is_available())
        except Exception:
            cuda_available = False

    return TorchState(
        version=version,
        cuda_version=cuda_version,
        compiled_arches=compiled,
        gpu_arches=arches,
        cuda_available=cuda_available,
    )


def pick_tags(cuda_version: str | None) -> list[str]:
    forced_tag = os.getenv(TAG_OVERRIDE_ENV)
    if forced_tag:
        return [forced_tag]

    tags: list[str] = []
    if cuda_version and cuda_version in CUDA_TAG_PREFERENCES:
        tags.extend(CUDA_TAG_PREFERENCES[cuda_version])
    tags.extend(DEFAULT_TAG_PRIORITY)

    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def install_torch(tag: str, base_version: Version, *, quiet: bool) -> bool:
    version_pin = f"{base_version.base_version}+{tag}"
    index_url = f"https://download.pytorch.org/whl/{tag}"
    cmd: list[str] = [
        "uv",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "--index-url",
        index_url,
        "--extra-index-url",
        "https://pypi.org/simple",
        f"torch=={version_pin}",
    ]
    try:
        run(cmd, quiet=quiet)
        return True
    except subprocess.CalledProcessError as exc:  # pragma: no cover - network failure
        warn(f"Failed to install torch=={version_pin} ({exc}).")
        return False


def verify_install(required_arches: set[str], *, quiet: bool) -> bool:
    script = """
import sys
import torch

required = {required_arches!r}
cuda_version = getattr(torch.version, "cuda", None)
cuda_available = getattr(torch.cuda, "is_available", lambda: False)()
arch_fn = getattr(torch.cuda, "get_arch_list", lambda: [])
compiled = {arch.lower() for arch in arch_fn()}
missing = [cap for cap in required if cap not in compiled]
if not required:
    raise SystemExit(0)
if cuda_version is None or not cuda_available or missing:
    raise SystemExit(1)
"""
    try:
        run([sys.executable, "-c", script], quiet=quiet)
        return True
    except subprocess.CalledProcessError:
        return False


def should_skip() -> bool:
    return os.getenv(SKIP_ENV, "").lower() in {"1", "true", "yes"}


def main() -> int:
    args = parse_args()

    if should_skip():
        info(f"Skipping CUDA torch fix because {SKIP_ENV} is set.", quiet=args.quiet)
        return 0

    state = current_torch_state()
    if state is None:
        return 0

    if not state.gpu_arches:
        info("No NVIDIA GPUs detected; leaving torch as-is.", quiet=args.quiet)
        return 0

    needs_reinstall, reason = state.needs_reinstall()
    if not needs_reinstall and not args.force:
        info(reason, quiet=args.quiet)
        return 0

    info(
        f"Reinstalling torch {state.version} for CUDA (reason: {reason}).",
        quiet=False,
    )

    tags = pick_tags(state.cuda_version)
    if not tags:
        warn("Could not determine a suitable CUDA wheel tag; install torch manually.")
        return 1

    for tag in tags:
        if install_torch(tag, state.version, quiet=args.quiet):
            if verify_install(state.gpu_arches, quiet=args.quiet):
                info(
                    f"Installed torch=={state.version.base_version}+{tag} with CUDA support.",
                    quiet=False,
                )
                return 0
            warn(f"Installed torch with {tag} tag but verification failed; trying next tag.")

    warn("Unable to install a CUDA-enabled PyTorch wheel automatically. "
         "Set METTA_TORCH_CUDA_TAG=cuXXX and rerun or reinstall manually.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
