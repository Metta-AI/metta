"""
Run training smoke tests with benchmarking.

This script runs a training session and measures performance.
"""

import os
import sys
import time
from typing import Tuple

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.benchmark import run_with_benchmark, write_github_output


def run_training_with_benchmark(timeout: int = 600) -> Tuple[bool, str, float, float]:
    """
    Run training with benchmarking.

    Args:
        timeout: Maximum time to wait for training (seconds)

    Returns:
        Tuple of (success, full_output, duration, peak_memory_mb)
    """
    cmd = [
        "python3",
        "./tools/train.py",
        "+hardware=github",
        "wandb=off",
    ]

    print("\nRunning training...")

    # Run with benchmarking
    result = run_with_benchmark(cmd=cmd, name="training", timeout=timeout)

    if not result["success"]:
        print(f"Training failed with exit code: {result['exit_code']}")
        if result["timeout"]:
            print("Training timed out")
        elif result["stderr"]:
            print("STDERR output:")
            print(result["stderr"][:1000])
            if len(result["stderr"]) > 1000:
                print("... (truncated)")

    full_output = result["stdout"] + "\n" + result["stderr"]
    return (result["success"], full_output, result["duration"], result["memory_peak_mb"])


def main():
    """Main training smoke test runner with benchmarking."""
    # Get configuration from environment
    timeout = int(os.environ.get("TRAINING_TIMEOUT", "600"))

    print("=" * 60)
    print("Training Smoke Test")
    print("=" * 60)
    print(f"Timeout: {timeout}s")
    print("=" * 60)

    # Run training
    start_time = time.time()
    success, output, duration, memory = run_training_with_benchmark(timeout=timeout)
    total_duration = time.time() - start_time

    # Summary
    print(f"\n{'=' * 60}")
    print("Benchmark Summary")
    print(f"{'=' * 60}")
    print(f"Total duration: {total_duration:.1f}s")
    print(f"Training duration: {duration:.1f}s")
    print(f"Peak memory usage: {memory:.1f} MB")
    print(f"Exit status: {'SUCCESS' if success else 'FAILED'}")

    # Write GitHub Actions outputs
    outputs = {
        "duration": f"{total_duration:.1f}",
        "memory_peak_mb": f"{memory:.1f}",
        "exit_code": "0" if success else "1",
    }

    write_github_output(outputs)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
