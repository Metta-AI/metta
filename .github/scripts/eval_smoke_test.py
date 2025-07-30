#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psutil>=6.0.0",
# ]
# ///
"""
Run evaluation smoke tests with reward checking.
"""

import json
import os
import sys
import time
from typing import Tuple

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.benchmark import run_with_benchmark
from utils.smoke_test import SmokeTest


class EvaluationSmokeTest(SmokeTest):
    """Evaluation smoke test with reward checking."""

    def __init__(self):
        self.policy = os.environ["POLICY"]
        self.min_reward = float(os.environ["MIN_REWARD"])
        self.max_attempts = int(os.environ["MAX_ATTEMPTS"])
        super().__init__()

    def get_command(self) -> list[str]:
        # This is overridden in get_command_for_attempt
        return []

    def get_command_for_attempt(self, attempt: int) -> list[str]:
        """Get command for a specific attempt."""
        return [
            "uv",
            "run",
            "./tools/sim.py",
            "sim=navigation",
            f"run=navigation_smoke_{attempt}",
            f"policy_uri=wandb://run/{self.policy}",
            "+eval_db_uri=wandb://artifacts/navigation_db",
            "seed=31415",
            "torch_deterministic=True",
            "device=cpu",
        ]

    def get_timeout(self) -> int:
        return int(os.environ.get("EVAL_TIMEOUT", "300"))

    def header_config_lines(self) -> list[str]:
        """Print evaluation-specific configuration."""
        return [f"Policy: {self.policy}", f"Minimum reward: {self.min_reward}", f"Max attempts: {self.max_attempts}"]

    def extract_metrics_from_output(self, output: str) -> dict:
        """Extract metrics JSON from output using the delimiters."""
        json_start = output.find("===JSON_OUTPUT_START===")
        json_end = output.find("===JSON_OUTPUT_END===")

        if json_start != -1 and json_end != -1:
            json_str = output[json_start + len("===JSON_OUTPUT_START===") : json_end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON between markers: {e}")
                print(f"JSON string: {json_str[:200]}...")
        else:
            print("JSON markers not found in output")
            print("Output preview (first 500 chars):")
            print(output[:500])

        return {}

    def extract_reward(self, metrics: dict) -> float | None:
        """Extract reward value from metrics dictionary."""
        try:
            reward = metrics["policies"][0]["checkpoints"][0]["metrics"]["reward_avg"]
            return float(reward)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"Failed to extract reward: {e}")
            if metrics:
                print(f"Metrics structure: {json.dumps(metrics, indent=2)[:500]}...")
            return None

    def run_evaluation_attempt(self, attempt: int) -> Tuple[bool, float | None, float, float]:
        """
        Run a single evaluation attempt.

        Returns:
            Tuple of (success, reward, duration, memory)
        """
        print(f"\n{'=' * 60}")
        print(f"Attempt {attempt}/{self.max_attempts}")
        print(f"{'=' * 60}")

        cmd = self.get_command_for_attempt(attempt)
        print(f"\nRunning evaluation attempt {attempt}...")

        result = run_with_benchmark(cmd=cmd, name=f"eval_attempt_{attempt}", timeout=self.timeout)

        success, output = self.process_result(result)

        if not success:
            print("Evaluation process failed!")
            # Continue to next attempt
            return False, None, result["duration"], result["memory_peak_mb"]

        # Extract metrics and reward
        metrics = self.extract_metrics_from_output(result["stdout"])

        if not metrics:
            print("No metrics found in output")
            return False, None, result["duration"], result["memory_peak_mb"]

        reward = self.extract_reward(metrics)

        if reward is None:
            print("Failed to extract reward from metrics")
            return False, None, result["duration"], result["memory_peak_mb"]

        print(f"Achieved reward: {reward}")

        if reward >= self.min_reward:
            print(f"✓ SUCCESS: Reward {reward} >= {self.min_reward}")
            return True, reward, result["duration"], result["memory_peak_mb"]
        else:
            print(f"✗ FAILED: Reward {reward} < {self.min_reward}")
            return False, reward, result["duration"], result["memory_peak_mb"]

    def run(self) -> int:
        """Run evaluation with multiple attempts and reward checking."""
        self.print_header()

        total_start_time = time.time()
        all_durations = []
        all_memories = []
        successful_attempt = None

        # Run evaluation attempts
        for attempt in range(1, self.max_attempts + 1):
            success, reward, duration, memory = self.run_evaluation_attempt(attempt)

            all_durations.append(duration)
            all_memories.append(memory)

            # Check if this attempt was successful with reward meeting threshold
            if success and reward is not None and reward >= self.min_reward:
                successful_attempt = attempt
                break

        # Calculate summary statistics
        total_duration = time.time() - total_start_time
        avg_duration = sum(all_durations) / len(all_durations) if all_durations else 0
        max_memory = max(all_memories) if all_memories else 0

        # Print summary
        print(f"\n{'=' * 60}")
        print("Benchmark Summary")
        print(f"{'=' * 60}")
        print(f"Total duration: {total_duration:.1f}s")
        print(f"Average attempt duration: {avg_duration:.1f}s")
        print(f"Peak memory usage: {max_memory:.1f} MB")

        if successful_attempt:
            print(f"\n✓ Smoke test passed on attempt {successful_attempt}")
        else:
            print(f"\n✗ All {self.max_attempts} attempts failed to meet minimum reward {self.min_reward}")

        # Write outputs
        self.write_outputs(total_duration, max_memory, successful_attempt is not None)

        return 0 if successful_attempt else 1


if __name__ == "__main__":
    test = EvaluationSmokeTest()
    sys.exit(test.run())
