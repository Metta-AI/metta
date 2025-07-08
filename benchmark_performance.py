#!/usr/bin/env -S uv run
"""
Performance benchmarking script to measure training speedup.

This script runs both the original and optimized configurations
and compares their performance to quantify the speedup achieved.
"""

import logging
import os
import time
import subprocess
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_benchmark(config_name: str, script_path: str, max_epochs: int = 10) -> dict:
    """Run a training benchmark and measure performance."""
    logger.info(f"🚀 Running benchmark: {config_name}")

    # Create a temporary config to limit training time
    temp_config = f"""
# Temporary config for benchmarking
trainer:
  total_timesteps: 1000000  # Small number for quick benchmark
  checkpoint:
    checkpoint_interval: 0  # Disable checkpoints for speed
  simulation:
    evaluate_interval: 0  # Disable evaluation for speed
    replay_interval: 0  # Disable replay for speed
  grad_mean_variance_interval: 0  # Disable gradient stats for speed
"""

    # Write temporary config
    config_path = Path(f"temp_benchmark_{config_name}.yaml")
    with open(config_path, "w") as f:
        f.write(temp_config)

    start_time = time.time()

    try:
        # Run the training script with limited epochs
        env = os.environ.copy()
        env["BENCHMARK_MODE"] = "1"
        env["MAX_EPOCHS"] = str(max_epochs)

        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        # Parse output for performance metrics
        steps_per_sec = 0
        for line in result.stdout.split('\n'):
            if 'steps/sec' in line:
                try:
                    # Extract steps/sec from log line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.endswith('steps/sec'):
                            steps_per_sec = float(part.replace('steps/sec', ''))
                            break
                except:
                    pass

        return {
            "config_name": config_name,
            "duration": duration,
            "steps_per_sec": steps_per_sec,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"⏰ Benchmark timed out: {config_name}")
        return {
            "config_name": config_name,
            "duration": 300,
            "steps_per_sec": 0,
            "success": False,
            "stdout": "",
            "stderr": "Timeout"
        }
    finally:
        # Clean up temporary config
        if config_path.exists():
            config_path.unlink()

def compare_performance(results: list) -> None:
    """Compare performance between different configurations."""
    logger.info("\n📊 Performance Comparison Results:")
    logger.info("=" * 60)

    # Find baseline (original) and optimized results
    baseline = None
    optimized = None

    for result in results:
        if "original" in result["config_name"].lower():
            baseline = result
        elif "optimized" in result["config_name"].lower() or "ultra" in result["config_name"].lower():
            optimized = result

    if baseline and optimized:
        # Calculate speedup
        if baseline["steps_per_sec"] > 0 and optimized["steps_per_sec"] > 0:
            speedup = optimized["steps_per_sec"] / baseline["steps_per_sec"]

            logger.info(f"🏁 Baseline ({baseline['config_name']}):")
            logger.info(f"   Steps/sec: {baseline['steps_per_sec']:.0f}")
            logger.info(f"   Duration: {baseline['duration']:.1f}s")

            logger.info(f"\n⚡ Optimized ({optimized['config_name']}):")
            logger.info(f"   Steps/sec: {optimized['steps_per_sec']:.0f}")
            logger.info(f"   Duration: {optimized['duration']:.1f}s")

            logger.info(f"\n🚀 Speedup: {speedup:.1f}x")

            if speedup >= 10:
                logger.info("🎉 Target achieved! 10x+ speedup reached!")
            elif speedup >= 5:
                logger.info("👍 Good progress! 5x+ speedup achieved.")
            else:
                logger.info("📈 Some improvement, but more optimization needed.")
        else:
            logger.warning("⚠️ Could not calculate speedup - missing performance data")
    else:
        logger.error("❌ Missing baseline or optimized results for comparison")

def main():
    """Main benchmarking function."""
    logger.info("🔬 Starting performance benchmarking...")

    results = []

    # Benchmark original configuration
    original_result = run_benchmark("original", "run.py")
    results.append(original_result)

    # Benchmark optimized configuration
    optimized_result = run_benchmark("optimized", "run_optimized.py")
    results.append(optimized_result)

    # Compare results
    compare_performance(results)

    # Save detailed results
    with open("benchmark_results.txt", "w") as f:
        f.write("Performance Benchmark Results\n")
        f.write("=" * 40 + "\n\n")

        for result in results:
            f.write(f"Configuration: {result['config_name']}\n")
            f.write(f"Duration: {result['duration']:.1f}s\n")
            f.write(f"Steps/sec: {result['steps_per_sec']:.0f}\n")
            f.write(f"Success: {result['success']}\n")
            f.write(f"Stdout:\n{result['stdout']}\n")
            f.write(f"Stderr:\n{result['stderr']}\n")
            f.write("-" * 40 + "\n\n")

    logger.info("💾 Detailed results saved to benchmark_results.txt")

if __name__ == "__main__":
    main()
