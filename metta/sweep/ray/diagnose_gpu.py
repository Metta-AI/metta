"""Diagnostic script to check GPU allocation in Ray and subprocess."""

import os
import subprocess
import sys

import ray
from ray import tune


def print_gpu_env(prefix=""):
    """Print all GPU-related environment variables."""
    print(f"\n{prefix} GPU Environment:")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"  RAY_CUDA_VISIBLE_DEVICES: {os.environ.get('RAY_CUDA_VISIBLE_DEVICES', 'NOT SET')}")

    # Check if torch can see GPUs
    try:
        import torch
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
    except ImportError:
        print("  torch: not imported")


def test_subprocess_gpu():
    """Check if subprocess sees GPUs."""
    print("\nSubprocess GPU check:")
    result = subprocess.run(
        [sys.executable, "-c", "import os; print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET'))"],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    print(f"  {result.stdout.strip()}")


def trial_function(config):
    """Ray Tune trial function."""
    print("\n" + "=" * 60)
    print("INSIDE TRIAL FUNCTION")
    print("=" * 60)

    print_gpu_env("Trial Function")
    test_subprocess_gpu()

    # Try to get Ray's GPU IDs
    try:
        gpu_ids = ray.get_gpu_ids()
        print(f"\nray.get_gpu_ids(): {gpu_ids}")
    except Exception as e:
        print(f"\nray.get_gpu_ids() failed: {e}")


if __name__ == "__main__":
    print("Starting Ray GPU Diagnostic")
    print("=" * 60)

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    print_gpu_env("Before Trial")

    # Run a trial with GPU allocation
    print("\nRunning trial with 1 GPU allocated...")
    trainable = tune.with_resources(trial_function, {"cpu": 2, "gpu": 1})

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(num_samples=1),
        param_space={},
    )

    results = tuner.fit()
    print("\nTrial completed!")
