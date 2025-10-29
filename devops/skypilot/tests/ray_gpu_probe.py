#!/usr/bin/env python

"""Minimal Ray GPU visibility probe.

Run this from within the SkyPilot job (ideally on the head node) to confirm
whether Ray assigns GPU IDs to remote tasks when connecting via Ray Client.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import ray


def _connect(address: str) -> None:
    print(f"[probe] Connecting to Ray at {address!r}")
    ray.init(address=address, namespace="gpu_probe", ignore_reinit_error=True)
    cluster_resources = ray.cluster_resources()
    print(f"[probe] Cluster resources: {json.dumps(cluster_resources, indent=2)}")


@ray.remote(num_gpus=1)
def _gpu_task() -> Dict[str, Any]:
    import subprocess
    from ray import get_gpu_ids

    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostics only
        torch_info: Dict[str, Any] | str = f"import failed: {exc}"
        torch_cuda_available = False
        torch_device_count = 0
    else:
        torch_info = {
            "torch_version": getattr(torch, "__version__", "unknown"),
            "torch_cuda_version": getattr(torch.version, "cuda", "unknown"),
        }
        torch_cuda_available = torch.cuda.is_available()
        torch_device_count = torch.cuda.device_count()

    try:
        ray_gpu_ids = get_gpu_ids()
    except Exception as exc:  # pragma: no cover - diagnostics only
        ray_gpu_ids = f"error: {exc}"

    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as exc:  # pragma: no cover - diagnostics only
        nvidia_smi = f"error: {exc}"

    return {
        "ray_gpu_ids": ray_gpu_ids,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ray_gpu_ids_env": os.environ.get("RAY_GPU_IDS"),
        "num_gpus_env": os.environ.get("NUM_GPUS"),
        "torch_cuda_available": torch_cuda_available,
        "torch_device_count": torch_device_count,
        "torch_info": torch_info,
        "nvidia_smi": nvidia_smi,
    }


def main() -> int:
    address = os.environ.get("RAY_ADDRESS")
    if len(sys.argv) > 1:
        address = sys.argv[1]

    if not address:
        print("Usage: set RAY_ADDRESS or pass ray://host:port as an argument", file=sys.stderr)
        return 1

    _connect(address)

    try:
        result = ray.get(_gpu_task.remote())
    finally:
        ray.shutdown()

    print("[probe] Remote GPU task diagnostics:")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
