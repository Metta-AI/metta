#!/usr/bin/env python3
"""Performance debugging tool for curriculum learning.

This script helps diagnose performance degradation in Learning Progress curriculum
by comparing different configurations and collecting detailed timing data.

Usage:
    # Run quick comparison (discrete vs LP)
    uv run ./tools/debug_performance.py --quick

    # Run with detailed profiling
    uv run ./tools/debug_performance.py --profile

    # Test specific configuration
    uv run ./tools/debug_performance.py --config performance_mode

    # Compare with main branch
    uv run ./tools/debug_performance.py --compare-branch main
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Test configurations
CONFIGS = {
    "baseline_discrete": {
        "use_lp": False,
        "description": "Discrete random curriculum (baseline)",
        "expected": "Fast, constant SPS",
    },
    "lp_default": {
        "use_lp": True,
        "description": "Learning Progress with default settings",
        "expected": "Slow, degrading SPS",
    },
    "lp_performance_mode": {
        "use_lp": True,
        "performance_mode": True,
        "description": "Learning Progress with performance_mode=True",
        "expected": "Fast if stats are the issue",
    },
}


class PerformanceTest:
    """Run and analyze performance tests."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("outputs/perf_debug")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, Any] = {}

    def run_test(
        self,
        config_name: str,
        timeout: int = 300,
        recipe: str = "experiment.cogs_v_clips",
    ) -> dict[str, Any]:
        """Run a single performance test."""
        config = CONFIGS[config_name]
        print(f"\n{'=' * 80}")
        print(f"Running: {config_name}")
        print(f"Description: {config['description']}")
        print(f"Expected: {config['expected']}")
        print(f"{'=' * 80}\n")

        # Build command
        cmd = [
            "uv",
            "run",
            "./tools/run.py",
            recipe,
            f"run=perf_debug_{config_name}",
        ]

        # Add config parameters
        for key, value in config.items():
            if key not in ["description", "expected"]:
                cmd.append(f"{key}={value}")

        print(f"Command: {' '.join(cmd)}")
        print(f"Timeout: {timeout}s\n")

        # Run with timeout
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - start_time
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            elapsed = timeout
            success = True  # Timeout is expected for these tests
            result = None

        # Parse results
        test_result = {
            "config_name": config_name,
            "description": config["description"],
            "elapsed_time": elapsed,
            "success": success,
            "sps_samples": self._extract_sps_from_logs(config_name),
        }

        # Calculate degradation
        if test_result["sps_samples"]:
            samples = test_result["sps_samples"]
            test_result["initial_sps"] = samples[0] if samples else None
            test_result["final_sps"] = samples[-1] if samples else None
            test_result["degradation_pct"] = (
                ((samples[0] - samples[-1]) / samples[0] * 100) if len(samples) > 1 and samples[0] > 0 else 0.0
            )

        self.results[config_name] = test_result
        self._print_test_result(test_result)

        return test_result

    def _extract_sps_from_logs(self, config_name: str) -> list[float]:
        """Extract SPS values from training logs."""
        # Find most recent log file for this run
        log_pattern = f"perf_debug_{config_name}"
        outputs_dir = Path("outputs")

        sps_values = []
        for log_dir in outputs_dir.glob(f"*{log_pattern}*/logs"):
            log_file = log_dir / "train.log"
            if log_file.exists():
                with open(log_file) as f:
                    for line in f:
                        # Parse SPS from log lines
                        # Format typically: "SPS: 1234.5" or "steps_per_second: 1234.5"
                        if "SPS:" in line or "steps_per_second" in line:
                            try:
                                # Extract number after SPS: or steps_per_second:
                                parts = line.split(":")
                                if len(parts) >= 2:
                                    sps = float(parts[-1].strip().split()[0])
                                    sps_values.append(sps)
                            except (ValueError, IndexError):
                                continue

        return sps_values

    def _print_test_result(self, result: dict[str, Any]) -> None:
        """Pretty print test results."""
        print(f"\n{'─' * 80}")
        print(f"Results for: {result['config_name']}")
        print(f"{'─' * 80}")
        print(f"Description: {result['description']}")
        print(f"Elapsed time: {result['elapsed_time']:.1f}s")

        if result.get("sps_samples"):
            print(f"SPS samples collected: {len(result['sps_samples'])}")
            print(f"Initial SPS: {result.get('initial_sps', 0):.1f}")
            print(f"Final SPS: {result.get('final_sps', 0):.1f}")
            print(f"Degradation: {result.get('degradation_pct', 0):.1f}%")

            # Show trend
            if len(result["sps_samples"]) > 5:
                print("\nSPS trend (first 5 and last 5 epochs):")
                samples = result["sps_samples"]
                for i, sps in enumerate(samples[:5]):
                    print(f"  Epoch ~{i + 1:3d}: {sps:7.1f} SPS")
                if len(samples) > 10:
                    print("  ...")
                for i, sps in enumerate(samples[-5:]):
                    epoch_num = len(samples) - 5 + i + 1
                    print(f"  Epoch ~{epoch_num:3d}: {sps:7.1f} SPS")
        else:
            print("No SPS data found in logs")

        print()

    def run_comparison(self, configs: list[str] | None = None, timeout: int = 300) -> None:
        """Run comparison between multiple configs."""
        configs = configs or ["baseline_discrete", "lp_default", "lp_performance_mode"]

        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON TEST")
        print("=" * 80)
        print(f"Configs to test: {', '.join(configs)}")
        print(f"Timeout per test: {timeout}s")
        print(f"Total estimated time: ~{timeout * len(configs) / 60:.1f} minutes")
        print("=" * 80)

        for config_name in configs:
            if config_name not in CONFIGS:
                print(f"Warning: Unknown config '{config_name}', skipping")
                continue

            self.run_test(config_name, timeout=timeout)

            # Save intermediate results
            self._save_results()

        self._print_comparison_summary()

    def _print_comparison_summary(self) -> None:
        """Print summary comparison of all tests."""
        if not self.results:
            print("No results to compare")
            return

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        # Table header
        print(f"{'Config':<25} {'Initial SPS':>12} {'Final SPS':>12} {'Degradation':>12}")
        print("─" * 80)

        # Sort by degradation (worst first)
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: x.get("degradation_pct", 0),
            reverse=True,
        )

        for result in sorted_results:
            initial = result.get("initial_sps", 0)
            final = result.get("final_sps", 0)
            degradation = result.get("degradation_pct", 0)

            print(f"{result['config_name']:<25} {initial:>12.1f} {final:>12.1f} {degradation:>11.1f}%")

        print()

        # Analysis
        baseline = self.results.get("baseline_discrete")
        lp_default = self.results.get("lp_default")
        lp_perf = self.results.get("lp_performance_mode")

        if baseline and lp_default:
            print("\nANALYSIS:")
            print("─" * 80)

            baseline_deg = baseline.get("degradation_pct", 0)
            lp_deg = lp_default.get("degradation_pct", 0)

            if lp_deg > baseline_deg + 5:
                print("✗ Learning Progress shows significantly more degradation than baseline")
                print(f"  Degradation difference: {lp_deg - baseline_deg:.1f}%")

                if lp_perf:
                    lp_perf_deg = lp_perf.get("degradation_pct", 0)
                    if lp_perf_deg < lp_deg - 5:
                        print("\n✓ performance_mode=True significantly reduces degradation")
                        print(f"  Improvement: {lp_deg - lp_perf_deg:.1f}%")
                        print("\n  RECOMMENDATION: Enable performance_mode=True in production")
                    else:
                        print("\n✗ performance_mode=True does not fix the issue")
                        print("  RECOMMENDATION: Further investigation needed (see investigation plan)")
            else:
                print("✓ Learning Progress degradation is similar to baseline")
                print("  This may be a general training issue, not curriculum-specific")

        print()

    def _save_results(self) -> None:
        """Save results to JSON file."""
        output_file = self.output_dir / "performance_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Debug performance degradation in curriculum learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick comparison (5 min timeout per test)",
    )

    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        help="Run specific configuration only",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per test in seconds (default: 300)",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations",
    )

    args = parser.parse_args()

    if args.list_configs:
        print("\nAvailable configurations:")
        print("=" * 80)
        for name, config in CONFIGS.items():
            print(f"\n{name}:")
            print(f"  Description: {config['description']}")
            print(f"  Expected: {config['expected']}")
            print(f"  Parameters: {[k for k in config.keys() if k not in ['description', 'expected']]}")
        return 0

    tester = PerformanceTest()

    if args.config:
        # Run single config
        tester.run_test(args.config, timeout=args.timeout)
    elif args.quick:
        # Quick comparison with short timeout
        tester.run_comparison(timeout=180)  # 3 minutes each
    else:
        # Full comparison
        tester.run_comparison(timeout=args.timeout)

    return 0


if __name__ == "__main__":
    sys.exit(main())
