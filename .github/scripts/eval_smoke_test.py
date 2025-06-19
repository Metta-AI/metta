"""
Run evaluation smoke tests for a W&B policy with benchmarking.

This script runs multiple evaluation attempts, measures performance,
and checks if the achieved reward meets the minimum threshold.
"""

import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.benchmark import run_with_benchmark, write_github_output


def run_evaluation_with_benchmark(
    attempt: int, policy: str, timeout: int = 300
) -> Tuple[bool, Dict, str, float, float]:
    """
    Run a single evaluation attempt with benchmarking.

    Args:
        attempt: Attempt number (used for run naming)
        policy: W&B policy identifier
        timeout: Maximum time to wait for evaluation (seconds)

    Returns:
        Tuple of (success, metrics_dict, full_output, duration, peak_memory_mb)
    """
    cmd = [
        "python3",
        "-m",
        "tools.sim",
        "sim=navigation",
        f"run=navigation_smoke_{attempt}",
        f"policy_uri=wandb://run/{policy}",
        "+eval_db_uri=wandb://artifacts/navigation_db",
        "seed=31415",
        "torch_deterministic=True",
        "device=cpu",
    ]

    print(f"\nRunning evaluation attempt {attempt}...")

    # Run with benchmarking
    result = run_with_benchmark(cmd=cmd, name=f"eval_attempt_{attempt}", timeout=timeout)

    if not result["success"]:
        print(f"Evaluation failed with exit code: {result['exit_code']}")
        if result["timeout"]:
            print("Evaluation timed out")
        elif result["stderr"]:
            print("STDERR output:")
            print(result["stderr"][:1000])
            if len(result["stderr"]) > 1000:
                print("... (truncated)")

    # Extract metrics from output
    full_output = result["stdout"] + "\n" + result["stderr"]
    metrics = extract_metrics_from_output(full_output)

    return (result["success"], metrics, full_output, result["duration"], result["memory_peak_mb"])


def extract_metrics_from_output(output: str) -> Dict:
    """Extract metrics JSON from output."""
    # Look for JSON between markers
    json_start = output.find("===JSON_OUTPUT_START===")
    json_end = output.find("===JSON_OUTPUT_END===")

    if json_start != -1 and json_end != -1:
        json_str = output[json_start + len("===JSON_OUTPUT_START===") : json_end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON between markers: {e}")
            print(f"JSON string: {json_str[:200]}...")

    # Try to find JSON-like content in the output
    import re

    json_pattern = r'\{.*"policies".*\}'
    matches = re.findall(json_pattern, output, re.DOTALL)
    if matches:
        # Try parsing matches from shortest to longest (more likely to be valid)
        for match in sorted(matches, key=len):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    print("No valid JSON found in output")
    print("Output preview (first 500 chars):")
    print(output[:500])
    return {}


def extract_reward(metrics: Dict) -> Optional[float]:
    """Extract reward value from metrics dictionary."""
    try:
        reward = metrics["policies"][0]["checkpoints"][0]["metrics"]["reward_avg"]
        return float(reward)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        print(f"Failed to extract reward: {e}")
        if metrics:
            print(f"Metrics structure: {json.dumps(metrics, indent=2)[:500]}...")
        return None


def main():
    """Main smoke test runner with benchmarking."""
    # Get configuration from environment
    policy = os.environ.get("POLICY", "b.rwalters.0605.nav.pr811.00")
    min_reward = float(os.environ.get("MIN_REWARD", "0.15"))
    max_attempts = int(os.environ.get("MAX_ATTEMPTS", "4"))

    print("=" * 60)
    print("Evaluation Smoke Test Configuration")
    print("=" * 60)
    print(f"Policy: {policy}")
    print(f"Minimum reward: {min_reward}")
    print(f"Max attempts: {max_attempts}")
    print("=" * 60)

    # Track overall metrics
    total_start_time = time.time()
    all_durations = []
    all_memories = []
    successful_attempt = None

    # Run evaluation attempts
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'=' * 60}")
        print(f"Attempt {attempt}/{max_attempts}")
        print(f"{'=' * 60}")

        success, metrics, output, duration, memory = run_evaluation_with_benchmark(attempt, policy)

        all_durations.append(duration)
        all_memories.append(memory)

        if not success:
            print("Evaluation failed!")
            continue

        # Extract and check reward
        reward = extract_reward(metrics)

        if reward is None:
            print("Failed to extract reward from metrics")
            continue

        print(f"Achieved reward: {reward}")

        if reward >= min_reward:
            print(f"✓ SUCCESS: Reward {reward} >= {min_reward}")
            successful_attempt = attempt
            break
        else:
            print(f"✗ FAILED: Reward {reward} < {min_reward}")

    # Calculate summary statistics
    total_duration = time.time() - total_start_time
    avg_duration = sum(all_durations) / len(all_durations) if all_durations else 0
    max_memory = max(all_memories) if all_memories else 0

    print(f"\n{'=' * 60}")
    print("Benchmark Summary")
    print(f"{'=' * 60}")
    print(f"Total duration: {total_duration:.1f}s")
    print(f"Average attempt duration: {avg_duration:.1f}s")
    print(f"Peak memory usage: {max_memory:.1f} MB")

    # Write GitHub Actions outputs using the utility
    benchmark_result = {
        "duration": total_duration,
        "memory_peak_mb": max_memory,
        "exit_code": 0 if successful_attempt else 1,
    }
    write_github_output(benchmark_result)

    if successful_attempt:
        return 0
    else:
        print(f"\n✗ All {max_attempts} attempts failed to meet minimum reward {min_reward}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
