#!/usr/bin/env python3
"""Install optional CUDA extras for the DRAMA state-space recipes.

This script best-effort installs `flash-attn` and `causal-conv1d` for the
currently detected CUDA toolkit. It is idempotent and exits with status 0 even
if the installation fails, printing actionable guidance instead.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Iterable

# Mapping from CUDA version reported by PyTorch to preferred wheel tags in the
# PyTorch extra index. We include a couple of fallbacks for neighbouring
# versions to maximise the chance of finding a matching wheel.
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


def run(cmd: Iterable[str]) -> bool:
    """Run a command returning True on success."""

    try:
        subprocess.run(list(cmd), check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_cuda_home() -> None:
    """Populate CUDA_HOME from torch if it is currently unset."""

    if os.getenv("CUDA_HOME"):
        return

    try:
        from torch.utils.cpp_extension import CUDA_HOME as torch_cuda_home
    except Exception:  # pragma: no cover - torch missing or stale
        return

    if torch_cuda_home:
        os.environ["CUDA_HOME"] = torch_cuda_home


def install_with_index(packages: list[str], tag: str) -> bool:
    index = f"https://download.pytorch.org/whl/{tag}"
    cmd = ["uv", "pip", "install", "--extra-index-url", index, *packages]
    return run(cmd)


def install_default(packages: list[str]) -> bool:
    cmd = ["uv", "pip", "install", *packages]
    return run(cmd)


def install_no_isolation(packages: list[str]) -> bool:
    cmd = ["pip", "install", "--no-build-isolation", *packages]
    return run(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="suppress success output")
    args = parser.parse_args()

    if shutil.which("nvidia-smi") is None:
        if not args.quiet:
            print("[drama-gpu] No NVIDIA GPU detected; skipping CUDA extras.")
        return 0

    try:
        import torch
    except ImportError:
        print("[drama-gpu] PyTorch not installed yet; skipping CUDA extras.")
        return 0

    cuda_version = torch.version.cuda
    if not cuda_version:
        print("[drama-gpu] PyTorch build lacks CUDA support; skipping CUDA extras.")
        return 0

    tags = CUDA_TAG_PREFERENCES.get(cuda_version, [])
    if not tags:
        print(
            f"[drama-gpu] No wheel tag mapping for CUDA {cuda_version}; "
            "install flash-attn / causal-conv1d manually if needed.",
        )
        return 0

    ensure_cuda_home()

    # Ensure build requirements that flash-attn often assumes are already present.
    install_default(["packaging"])  # idempotent
    install_default(["torch"])

    flash_installed = False
    causal_installed = False

    # Try tag-specific wheels first, then fall back to source installs.
    for tag in tags:
        if not flash_installed:
            flash_installed = install_with_index(["flash-attn"], tag)
        if not causal_installed:
            for pkg in (f"causal-conv1d-{tag}", "causal-conv1d"):
                if install_with_index([pkg], tag):
                    causal_installed = True
                    break
        if flash_installed and causal_installed:
            break

    if not flash_installed:
        flash_installed = install_default(["flash-attn"])
    if not causal_installed:
        for pkg in ("causal-conv1d",):
            if install_default([pkg]):
                causal_installed = True
                break

    if not flash_installed:
        flash_installed = install_no_isolation(["flash-attn"])
    if not causal_installed:
        causal_installed = install_no_isolation(["causal-conv1d"])

    if flash_installed and causal_installed:
        if not args.quiet:
            print("[drama-gpu] Installed flash-attn and causal-conv1d successfully.")
        return 0

    missing = []
    if not flash_installed:
        missing.append("flash-attn")
    if not causal_installed:
        missing.append("causal-conv1d")

    print(
        "[drama-gpu] Unable to install %s automatically. "
        "Please install the matching CUDA wheels manually using:\n"
        "  pip install --extra-index-url https://download.pytorch.org/whl/<tag> %s"
        % (", ".join(missing), " ".join(missing)),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
