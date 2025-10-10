#!/usr/bin/env python3
"""Print instructions for installing the optional CUDA extras used by the DRAMA model."""

from __future__ import annotations

import torch

# Map CUDA version -> preferred PyTorch wheel tag
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
    cuda = getattr(torch.version, "cuda", None)
    if not cuda:
        print("PyTorch build does not expose CUDA; no GPU extras needed.")
        return

    tag = CUDA_TAG.get(cuda)
    if not tag:
        print(
            "Unsupported CUDA version {cuda}. Install flash-attn / causal-conv1d manually, e.g.\n"
            "  pip install --extra-index-url https://download.pytorch.org/whl/<tag> flash-attn causal-conv1d"
        )
        return

    print("Run these commands to enable the DRAMA CUDA fast-path:")
    print(f"  pip install --extra-index-url https://download.pytorch.org/whl/{tag} flash-attn")
    print(f"  pip install --extra-index-url https://download.pytorch.org/whl/{tag} causal-conv1d")


if __name__ == "__main__":
    main()
