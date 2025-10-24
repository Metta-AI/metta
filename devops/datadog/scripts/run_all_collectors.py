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

        collector_start = time.time()
        try:
            metrics = run_collector(name, push=True, verbose=False, json_output=False)
            collector_duration = time.time() - collector_start

            results[name] = {
                "status": "success",
                "metrics_count": len(metrics),
                "duration": collector_duration,
            }

            print(f"✅ {name}: {len(metrics)} metrics collected in {collector_duration:.2f}s")

        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted while running {name} collector")
            results[name] = {"status": "interrupted", "metrics_count": 0, "duration": 0}
            break

        except Exception as e:
            collector_duration = time.time() - collector_start
            results[name] = {
                "status": "failed",
                "error": str(e),
                "metrics_count": 0,
                "duration": collector_duration,
            }

            print(f"❌ {name}: Failed after {collector_duration:.2f}s")
            print(f"   Error: {e}")
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
    failed = [name for name, result in results.items() if result["status"] == "failed"]
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
    main()
