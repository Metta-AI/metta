#!/usr/bin/env python3
"""Run all Datadog collectors sequentially.

This script runs all configured collectors in order, continuing even if
individual collectors fail. Used by the dashboard-cronjob CronJob to
collect all metrics on a single schedule.

Exit codes:
    0: All collectors succeeded
    1: One or more collectors failed
"""

import sys
import time
from datetime import datetime

# Import all collector functions
from devops.datadog.run_collector import (
    run_asana_collector,
    run_ec2_collector,
    run_github_collector,
    run_kubernetes_collector,
    run_skypilot_collector,
    run_wandb_collector,
)

# Define collectors to run (in priority order)
COLLECTORS = [
    ("github", run_github_collector),
    ("kubernetes", run_kubernetes_collector),
    ("ec2", run_ec2_collector),
    ("skypilot", run_skypilot_collector),
    ("wandb", run_wandb_collector),
    ("asana", run_asana_collector),
]


def main():
    """Run all collectors and report results."""
    print("=" * 80)
    print(f"Starting all collectors at {datetime.utcnow().isoformat()}Z")
    print("=" * 80)

    results = {}
    start_time = time.time()

    for name, collector_func in COLLECTORS:
        print(f"\n{'=' * 80}")
        print(f"Running {name} collector...")
        print(f"{'=' * 80}")

        collector_start = time.time()
        try:
            metrics = collector_func(push=True, verbose=False, json_output=False)
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

    # Exit with error if any collectors failed
    if failed or interrupted:
        sys.exit(1)

    print("\n✅ All collectors completed successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()
