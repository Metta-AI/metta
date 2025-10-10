#!/usr/bin/env python3
"""Print the commands needed to install the optional CUDA extras (flash-attn / causal-conv1d)."""

from __future__ import annotations

from importlib import import_module

CUDA_TAG = {
    "11.8": "cu118",
    "12.1": "cu121",
    "12.2": "cu122",
    "12.3": "cu122",
    "12.4": "cu124",
    "12.5": "cu124",
    "12.6": "cu126",
    "12.7": "cu126",
    "12.8": "cu126",
}


def main() -> None:  # pragma: no cover - utility script
    try:
        torch = import_module("torch")
    except ImportError:
        print("PyTorch is not installed; install flash-attn/causal-conv1d manually if needed.")
        return

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        print("PyTorch build does not report CUDA support; no GPU extras needed.")
        return

    tag = CUDA_TAG.get(cuda_version)
    if tag is None:
        print(
            f"Unsupported CUDA version {cuda_version}. Install manually using:\n"
            "  pip install --extra-index-url https://download.pytorch.org/whl/<tag> flash-attn causal-conv1d"
        )
        return

    print("Run the following commands to enable the DRAMA CUDA fast path:")
    print(f"  pip install --extra-index-url https://download.pytorch.org/whl/{tag} flash-attn")
    print(f"  pip install --extra-index-url https://download.pytorch.org/whl/{tag} causal-conv1d")


if __name__ == "__main__":
    main()
