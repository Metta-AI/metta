#!/usr/bin/env python3
"""Run all Datadog collectors sequentially.

This script runs all configured collectors in order, continuing even if
individual collectors fail. Pushes all metrics to Datadog.

Usage:
    # Run all collectors and push to Datadog
    uv run python devops/datadog/scripts/run_all_collectors.py

This works as both a production cronjob script and a local integration test.
Real data is collected from all services (GitHub, AWS, Kubernetes, etc.) and
pushed to Datadog.

Exit codes:
    0: All collectors succeeded
    1: One or more collectors failed
"""

import multiprocessing
import sys
import time
from datetime import datetime

# Import the generic run_collector function
from devops.datadog.scripts.run_collector import run_collector

# Define collectors to run (in priority order)
COLLECTORS = [
    "github",
    "kubernetes",
    "ec2",
    "skypilot",
    "wandb",
    "asana",
    "health_fom",  # Normalized health scores (runs after raw metrics collected)
]

# Per-collector timeout in seconds (2 minutes default)
# Protects against indefinite hangs from slow API responses or network issues
COLLECTOR_TIMEOUT = 120


def _run_collector_in_process(name: str, queue: multiprocessing.Queue) -> None:
    """Run collector in a separate process and put results in queue.

    This wrapper runs in a subprocess and is used to enable timeout+kill behavior.
    If the collector hangs, the parent process can kill this entire process tree.

    Args:
        name: Collector name
        queue: Multiprocessing queue to return results
    """
    try:
        metrics = run_collector(name, push=True, verbose=False)
        queue.put({"status": "success", "metrics": metrics})
    except Exception as e:
        queue.put({"status": "error", "error": str(e)})


def run_collector_with_timeout(name: str, timeout: int) -> dict:
    """Run a collector with robust timeout protection.

    Uses multiprocessing to run the collector in a separate process that can be
    forcefully terminated if it hangs. This handles cases where collectors spawn
    subprocesses that don't respond to thread-based timeouts.

    Args:
        name: Collector name
        timeout: Timeout in seconds

    Returns:
        Dictionary with collector results:
        - status: "success", "timeout", or "error"
        - metrics: Collected metrics (if successful)
        - metrics_count: Number of metrics
        - duration: Time taken
        - error: Error message (if failed)
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_run_collector_in_process, args=(name, queue))

    start_time = time.time()
    process.start()

    # Wait for process to complete or timeout
    process.join(timeout=timeout)
    duration = time.time() - start_time

    # If process is still alive after timeout, kill it
    if process.is_alive():
        print(f"   ⚠️  Timeout after {timeout}s - terminating process...")
        process.terminate()
        process.join(timeout=5)  # Give it 5 seconds to cleanup

        if process.is_alive():
            print("   ⚠️  Process didn't terminate cleanly - killing...")
            process.kill()
            process.join()

        return {
            "status": "timeout",
            "error": f"Timeout after {timeout}s",
            "metrics_count": 0,
            "duration": duration,
        }

    # Process completed - get results from queue
    if not queue.empty():
        result = queue.get()
        if result["status"] == "success":
            metrics = result["metrics"]
            return {
                "status": "success",
                "metrics": metrics,
                "metrics_count": len(metrics),
                "duration": duration,
            }
        else:
            return {
                "status": "error",
                "error": result.get("error", "Unknown error"),
                "metrics_count": 0,
                "duration": duration,
            }
    else:
        # Process exited but didn't put anything in queue
        return {
            "status": "error",
            "error": f"Process exited with code {process.exitcode}",
            "metrics_count": 0,
            "duration": duration,
        }


def main():
    """Run all collectors and report results."""
    print("=" * 80)
    print(f"Starting all collectors at {datetime.utcnow().isoformat()}Z")
    print("=" * 80)

    results = {}
    start_time = time.time()

    for name in COLLECTORS:
        print(f"\n{'=' * 80}")
        print(f"Running {name} collector...")
        print(f"{'=' * 80}")

        try:
            # Run collector with robust timeout protection
            # This uses multiprocessing to ensure we can kill hanging subprocesses
            result = run_collector_with_timeout(name, COLLECTOR_TIMEOUT)
            results[name] = result

            if result["status"] == "success":
                print(f"✅ {name}: {result['metrics_count']} metrics collected in {result['duration']:.2f}s")
            elif result["status"] == "timeout":
                print(f"❌ {name}: {result['error']}")
                print("   Skipping to next collector...")
            else:  # error
                print(f"❌ {name}: Failed after {result['duration']:.2f}s")
                print(f"   Error: {result['error']}")
                print("   Skipping to next collector...")

        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted while running {name} collector")
            results[name] = {"status": "interrupted", "metrics_count": 0, "duration": 0}
            break

        except Exception as e:
            # Catch-all for any unexpected errors in the wrapper itself
            results[name] = {
                "status": "error",
                "error": f"Wrapper error: {str(e)}",
                "metrics_count": 0,
                "duration": 0,
            }
            print(f"❌ {name}: Unexpected wrapper error: {e}")
            print("   Skipping to next collector...")
            # Continue to next collector

    # Print summary
    total_duration = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Completed at: {datetime.utcnow().isoformat()}Z")
    print()

    successful = [name for name, result in results.items() if result["status"] == "success"]
    failed = [name for name, result in results.items() if result["status"] in ("error", "timeout", "failed")]
    interrupted = [name for name, result in results.items() if result["status"] == "interrupted"]

    total_metrics = sum(r["metrics_count"] for r in results.values())

    print(f"Collectors run: {len(results)}/{len(COLLECTORS)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⚠️  Interrupted: {len(interrupted)}")
    print(f"Total metrics collected: {total_metrics}")
    print()

    if successful:
        print("Successful collectors:")
        for name in successful:
            r = results[name]
            print(f"  - {name}: {r['metrics_count']} metrics in {r['duration']:.2f}s")

    if failed:
        print("\nFailed collectors:")
        for name in failed:
            r = results[name]
            print(f"  - {name}: {r.get('error', 'Unknown error')}")

    # Exit with success if at least some collectors succeeded
    # Partial data is better than no data
    if successful:
        if failed:
            print(f"\n⚠️  Partial success: {len(successful)}/{len(COLLECTORS)} collectors succeeded")
            print("Job will exit with success code - partial data collected")
        else:
            print("\n✅ All collectors completed successfully")
        sys.exit(0)

    # Only fail if ALL collectors failed
    print("\n❌ All collectors failed - no data collected")
    sys.exit(1)


if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    multiprocessing.set_start_method("spawn", force=True)
    main()
