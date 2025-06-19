"""
Run replay smoke tests with benchmarking.

This script runs a replay session, measures performance,
and verifies that replay functionality works correctly.
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.benchmark import run_with_benchmark, write_github_output


def run_replay_with_benchmark(
    replay_dir: str = "replay_output", timeout: int = 300, additional_args: List[str] = None
) -> Tuple[bool, Dict, str, float, float]:
    """
    Run replay with benchmarking.

    Args:
        replay_dir: Directory for replay outputs
        timeout: Maximum time to wait for replay (seconds)
        additional_args: Additional command line arguments

    Returns:
        Tuple of (success, metrics_dict, full_output, duration, peak_memory_mb)
    """
    cmd = [
        "python3",
        "./tools/replay.py",
        "+hardware=github",
        "wandb=off",
        f"output_dir={replay_dir}",
    ]

    if additional_args:
        cmd.extend(additional_args)

    print("\nRunning replay...")
    print(f"Output directory: {replay_dir}")
    print(f"Command: {' '.join(cmd)}")

    # Create output directory if it doesn't exist
    Path(replay_dir).mkdir(parents=True, exist_ok=True)

    # Run with benchmarking
    result = run_with_benchmark(cmd=cmd, name="replay", timeout=timeout, cwd=".")

    if not result["success"]:
        print(f"Replay failed with exit code: {result['exit_code']}")
        if result["timeout"]:
            print("Replay timed out")
        elif result["stderr"]:
            print("STDERR output:")
            print(result["stderr"][:1000])
            if len(result["stderr"]) > 1000:
                print("... (truncated)")

    # Extract metrics from output
    full_output = result["stdout"] + "\n" + result["stderr"]
    metrics = extract_replay_metrics(full_output)

    return (result["success"], metrics, full_output, result["duration"], result["memory_peak_mb"])


def extract_replay_metrics(output: str) -> Dict:
    """Extract replay metrics from output."""
    metrics = {
        "episodes_replayed": 0,
        "frames_processed": 0,
        "replay_completed": False,
        "errors_encountered": 0,
        "warnings_count": 0,
        "fps": None,
        "success_rate": None,
    }

    lines = output.split("\n")
    for line in lines:
        # Look for episode counts
        episode_match = re.search(r"episode[s]?[:\s]+(\d+)", line, re.IGNORECASE)
        if episode_match:
            metrics["episodes_replayed"] = max(metrics["episodes_replayed"], int(episode_match.group(1)))

        # Look for frame counts
        frame_match = re.search(r"frame[s]?[:\s]+(\d+)", line, re.IGNORECASE)
        if frame_match:
            metrics["frames_processed"] = max(metrics["frames_processed"], int(frame_match.group(1)))

        # Look for FPS metrics
        fps_match = re.search(r"fps[:\s]+([\d.]+)", line, re.IGNORECASE)
        if fps_match:
            try:
                metrics["fps"] = float(fps_match.group(1))
            except ValueError:
                pass

        # Look for success rate
        success_match = re.search(r"success[_\s]?rate[:\s]+([\d.]+)%?", line, re.IGNORECASE)
        if success_match:
            try:
                metrics["success_rate"] = float(success_match.group(1))
            except ValueError:
                pass

        # Look for completion indicators
        completion_terms = ["replay complete", "finished replay", "all episodes processed", "done"]
        if any(term in line.lower() for term in completion_terms):
            metrics["replay_completed"] = True

        # Count errors and warnings
        if "error" in line.lower() and "error: 0" not in line.lower():
            metrics["errors_encountered"] += 1
        if "warning" in line.lower():
            metrics["warnings_count"] += 1

    return metrics


def verify_replay_outputs(replay_dir: str) -> Dict[str, any]:
    """Verify that expected replay outputs were created."""
    output_dir = Path(replay_dir)

    verification = {
        "directory_exists": output_dir.exists(),
        "has_outputs": False,
        "video_files": 0,
        "log_files": 0,
        "data_files": 0,
        "total_files": 0,
        "total_size_mb": 0,
        "output_types": set(),
    }

    if output_dir.exists():
        files = list(output_dir.rglob("*"))
        all_files = [f for f in files if f.is_file()]
        verification["total_files"] = len(all_files)
        verification["has_outputs"] = verification["total_files"] > 0

        # Categorize files by type
        video_extensions = {".mp4", ".avi", ".mov", ".gif", ".webm"}
        log_extensions = {".log", ".txt"}
        data_extensions = {".json", ".pkl", ".npy", ".npz", ".csv", ".parquet"}

        for f in all_files:
            suffix = f.suffix.lower()
            if suffix in video_extensions:
                verification["video_files"] += 1
                verification["output_types"].add("video")
            elif suffix in log_extensions:
                verification["log_files"] += 1
                verification["output_types"].add("log")
            elif suffix in data_extensions:
                verification["data_files"] += 1
                verification["output_types"].add("data")
            else:
                verification["output_types"].add(f"other({suffix})")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in all_files)
        verification["total_size_mb"] = total_size / (1024 * 1024)

        # Convert set to list for JSON serialization
        verification["output_types"] = list(verification["output_types"])

    return verification


def check_replay_quality(metrics: Dict, verification: Dict) -> Dict[str, bool]:
    """Check various quality metrics for the replay."""
    quality_checks = {
        "has_episodes": metrics["episodes_replayed"] > 0,
        "has_frames": metrics["frames_processed"] > 0,
        "reasonable_fps": metrics["fps"] is not None and metrics["fps"] > 0,
        "no_critical_errors": metrics["errors_encountered"] == 0,
        "produced_outputs": verification["has_outputs"],
        "replay_indicated_complete": metrics["replay_completed"],
    }

    # Add success rate check if available
    if metrics["success_rate"] is not None:
        quality_checks["acceptable_success_rate"] = metrics["success_rate"] >= 50.0

    return quality_checks


def main():
    """Main replay smoke test runner with benchmarking."""
    # Get configuration from environment
    replay_dir = os.environ.get("REPLAY_OUTPUT_DIR", "replay_output")
    min_episodes = int(os.environ.get("MIN_EPISODES", "1"))
    min_frames = int(os.environ.get("MIN_FRAMES", "10"))
    timeout = int(os.environ.get("REPLAY_TIMEOUT", "300"))
    require_video = os.environ.get("REQUIRE_VIDEO_OUTPUT", "false").lower() == "true"

    # Parse additional arguments if provided
    additional_args = os.environ.get("REPLAY_ADDITIONAL_ARGS", "").split()
    additional_args = [arg for arg in additional_args if arg]  # Filter empty strings

    print("=" * 60)
    print("Replay Smoke Test Configuration")
    print("=" * 60)
    print(f"Output directory: {replay_dir}")
    print(f"Minimum episodes: {min_episodes}")
    print(f"Minimum frames: {min_frames}")
    print(f"Require video output: {require_video}")
    print(f"Timeout: {timeout}s")
    if additional_args:
        print(f"Additional args: {' '.join(additional_args)}")
    print("=" * 60)

    # Track overall metrics
    start_time = time.time()

    # Run replay
    success, metrics, output, duration, memory = run_replay_with_benchmark(
        replay_dir=replay_dir, timeout=timeout, additional_args=additional_args
    )

    # Verify outputs
    print(f"\n{'=' * 60}")
    print("Verifying replay outputs...")
    print(f"{'=' * 60}")

    verification = verify_replay_outputs(replay_dir)

    print(f"Directory exists: {verification['directory_exists']}")
    print(f"Total files: {verification['total_files']}")
    print(f"Video files: {verification['video_files']}")
    print(f"Log files: {verification['log_files']}")
    print(f"Data files: {verification['data_files']}")
    print(f"Total size: {verification['total_size_mb']:.1f} MB")
    print(f"Output types: {', '.join(verification['output_types']) if verification['output_types'] else 'None'}")

    # Check replay quality
    quality_checks = check_replay_quality(metrics, verification)

    # Check success criteria
    all_checks_passed = True

    print(f"\n{'=' * 60}")
    print("Success Criteria Check")
    print(f"{'=' * 60}")

    # Check 1: Replay process completed successfully
    if success:
        print("✓ Replay process completed successfully")
    else:
        print("✗ Replay process failed")
        all_checks_passed = False

    # Check 2: Minimum episodes replayed
    if metrics["episodes_replayed"] >= min_episodes:
        print(f"✓ Replayed {metrics['episodes_replayed']} episodes (>= {min_episodes})")
    else:
        print(f"✗ Episodes replayed: {metrics['episodes_replayed']} (expected >= {min_episodes})")
        all_checks_passed = False

    # Check 3: Minimum frames processed
    if metrics["frames_processed"] >= min_frames:
        print(f"✓ Processed {metrics['frames_processed']} frames (>= {min_frames})")
    else:
        print(f"✗ Frames processed: {metrics['frames_processed']} (expected >= {min_frames})")
        all_checks_passed = False

    # Check 4: No critical errors
    if quality_checks["no_critical_errors"]:
        print("✓ No critical errors encountered")
    else:
        print(f"✗ Encountered {metrics['errors_encountered']} errors")
        all_checks_passed = False

    # Check 5: Required outputs created
    if verification["has_outputs"]:
        print(f"✓ Outputs created ({verification['total_files']} files)")
    else:
        print("✗ No output files created")
        all_checks_passed = False

    # Check 6: Video output (if required)
    if require_video:
        if verification["video_files"] > 0:
            print(f"✓ Video outputs created ({verification['video_files']} files)")
        else:
            print("✗ No video outputs created (required)")
            all_checks_passed = False

    # Additional quality indicators
    print(f"\n{'=' * 60}")
    print("Quality Indicators")
    print(f"{'=' * 60}")

    if metrics["fps"] is not None:
        print(f"FPS: {metrics['fps']:.1f}")

    if metrics["success_rate"] is not None:
        print(f"Success rate: {metrics['success_rate']:.1f}%")

    print(f"Warnings encountered: {metrics['warnings_count']}")

    # Summary
    total_duration = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("Benchmark Summary")
    print(f"{'=' * 60}")
    print(f"Total duration: {total_duration:.1f}s")
    print(f"Replay duration: {duration:.1f}s")
    print(f"Peak memory usage: {memory:.1f} MB")
    print(f"Overall result: {'SUCCESS' if all_checks_passed else 'FAILED'}")

    # Write GitHub Actions outputs
    outputs = {
        "duration": f"{total_duration:.1f}",
        "memory_peak_mb": f"{memory:.1f}",
        "exit_code": "0" if all_checks_passed else "1",
        "episodes_replayed": str(metrics["episodes_replayed"]),
        "frames_processed": str(metrics["frames_processed"]),
        "output_files": str(verification["total_files"]),
        "output_size_mb": f"{verification['total_size_mb']:.1f}",
        "video_files": str(verification["video_files"]),
        "fps": f"{metrics['fps']:.1f}" if metrics["fps"] else "0",
    }

    write_github_output(outputs)

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
