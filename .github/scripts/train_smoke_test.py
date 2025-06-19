"""
Run training smoke tests with benchmarking.

This script runs a training session, measures performance,
and verifies that the training completes successfully and produces expected outputs.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.benchmark import run_with_benchmark, write_github_output


def run_training_with_benchmark(checkpoint_path: str, timeout: int = 600) -> Tuple[bool, Dict, str, float, float]:
    """
    Run training with benchmarking.

    Args:
        checkpoint_path: Path where checkpoints should be saved
        timeout: Maximum time to wait for training (seconds)

    Returns:
        Tuple of (success, metrics_dict, full_output, duration, peak_memory_mb)
    """
    cmd = [
        "python3",
        "./tools/train.py",
        "+hardware=github",
        "wandb=off",
        f"checkpoint_dir={checkpoint_path}",
    ]

    print("\nRunning training...")
    print(f"Checkpoint path: {checkpoint_path}")

    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Run with benchmarking
    result = run_with_benchmark(cmd=cmd, name="training", timeout=timeout, cwd=".")

    if not result["success"]:
        print(f"Training failed with exit code: {result['exit_code']}")
        if result["timeout"]:
            print("Training timed out")
        elif result["stderr"]:
            print("STDERR output:")
            print(result["stderr"][:1000])
            if len(result["stderr"]) > 1000:
                print("... (truncated)")

    # Extract metrics from output
    full_output = result["stdout"] + "\n" + result["stderr"]
    metrics = extract_training_metrics(full_output)

    return (result["success"], metrics, full_output, result["duration"], result["memory_peak_mb"])


def extract_training_metrics(output: str) -> Dict:
    """Extract training metrics from output."""
    metrics = {
        "steps_completed": None,
        "final_loss": None,
        "training_completed": False,
        "checkpoint_saved": False,
    }

    # Look for common training indicators in output
    lines = output.split("\n")
    for line in lines:
        # Look for step/iteration counts
        if "step" in line.lower() or "iteration" in line.lower():
            import re

            # Try to extract step number
            step_match = re.search(r"step[:\s]+(\d+)", line, re.IGNORECASE)
            if step_match:
                metrics["steps_completed"] = int(step_match.group(1))
            else:
                iter_match = re.search(r"iteration[:\s]+(\d+)", line, re.IGNORECASE)
                if iter_match:
                    metrics["steps_completed"] = int(iter_match.group(1))

        # Look for loss values
        if "loss" in line.lower():
            import re

            loss_match = re.search(r"loss[:\s]+([\d.]+)", line, re.IGNORECASE)
            if loss_match:
                try:
                    metrics["final_loss"] = float(loss_match.group(1))
                except ValueError:
                    pass

        # Look for completion indicators
        if any(indicator in line.lower() for indicator in ["training complete", "finished training", "done"]):
            metrics["training_completed"] = True

        # Look for checkpoint saving indicators
        if any(indicator in line.lower() for indicator in ["checkpoint saved", "saving checkpoint", "saved model"]):
            metrics["checkpoint_saved"] = True

    return metrics


def verify_checkpoint_files(checkpoint_path: str) -> Dict[str, bool]:
    """Verify that expected checkpoint files were created."""
    checkpoint_dir = Path(checkpoint_path)

    verification = {
        "directory_exists": checkpoint_dir.exists(),
        "has_files": False,
        "has_model_files": False,
        "file_count": 0,
        "total_size_mb": 0,
    }

    if checkpoint_dir.exists():
        files = list(checkpoint_dir.rglob("*"))
        verification["file_count"] = len([f for f in files if f.is_file()])
        verification["has_files"] = verification["file_count"] > 0

        # Check for common model file extensions
        model_extensions = {".pt", ".pth", ".ckpt", ".pkl", ".bin", ".safetensors"}
        model_files = [f for f in files if f.suffix.lower() in model_extensions]
        verification["has_model_files"] = len(model_files) > 0

        # Calculate total size
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        verification["total_size_mb"] = total_size / (1024 * 1024)

    return verification


def main():
    """Main training smoke test runner with benchmarking."""
    # Get configuration from environment
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "train_dir/checkpoints")
    min_steps = int(os.environ.get("MIN_TRAINING_STEPS", "10"))
    max_loss = float(os.environ.get("MAX_FINAL_LOSS", "10.0"))
    timeout = int(os.environ.get("TRAINING_TIMEOUT", "600"))

    print("=" * 60)
    print("Training Smoke Test Configuration")
    print("=" * 60)
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Minimum training steps: {min_steps}")
    print(f"Maximum final loss: {max_loss}")
    print(f"Timeout: {timeout}s")
    print("=" * 60)

    # Track overall metrics
    start_time = time.time()

    # Run training
    success, metrics, output, duration, memory = run_training_with_benchmark(
        checkpoint_path=checkpoint_path, timeout=timeout
    )

    # Verify checkpoint files
    print(f"\n{'=' * 60}")
    print("Verifying checkpoint files...")
    print(f"{'=' * 60}")

    checkpoint_verification = verify_checkpoint_files(checkpoint_path)

    for key, value in checkpoint_verification.items():
        print(f"{key}: {value}")

    # Check success criteria
    all_checks_passed = True

    print(f"\n{'=' * 60}")
    print("Success Criteria Check")
    print(f"{'=' * 60}")

    # Check 1: Training process completed successfully
    if success:
        print("✓ Training process completed successfully")
    else:
        print("✗ Training process failed")
        all_checks_passed = False

    # Check 2: Minimum steps completed
    steps = metrics.get("steps_completed")
    if steps and steps >= min_steps:
        print(f"✓ Completed {steps} steps (>= {min_steps})")
    else:
        print(f"✗ Steps completed: {steps} (expected >= {min_steps})")
        all_checks_passed = False

    # Check 3: Loss is reasonable
    loss = metrics.get("final_loss")
    if loss is not None and loss <= max_loss:
        print(f"✓ Final loss {loss:.4f} (<= {max_loss})")
    elif loss is not None:
        print(f"✗ Final loss {loss:.4f} (> {max_loss})")
        all_checks_passed = False
    else:
        print("⚠ Could not extract loss value from output")

    # Check 4: Checkpoint files created
    if checkpoint_verification["has_model_files"]:
        print(
            f"✓ Model files created ({checkpoint_verification['file_count']} files, "
            f"{checkpoint_verification['total_size_mb']:.1f} MB)"
        )
    else:
        print("✗ No model checkpoint files found")
        all_checks_passed = False

    # Summary
    total_duration = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("Benchmark Summary")
    print(f"{'=' * 60}")
    print(f"Total duration: {total_duration:.1f}s")
    print(f"Training duration: {duration:.1f}s")
    print(f"Peak memory usage: {memory:.1f} MB")
    print(f"Overall result: {'SUCCESS' if all_checks_passed else 'FAILED'}")

    # Write GitHub Actions outputs
    outputs = {
        "duration": f"{total_duration:.1f}",
        "memory_peak_mb": f"{memory:.1f}",
        "exit_code": "0" if all_checks_passed else "1",
        "steps_completed": str(steps) if steps else "0",
        "checkpoint_files": str(checkpoint_verification["file_count"]),
        "checkpoint_size_mb": f"{checkpoint_verification['total_size_mb']:.1f}",
    }

    write_github_output(outputs)

    # Also set the exit code for the step
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
