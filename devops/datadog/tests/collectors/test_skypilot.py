#!/usr/bin/env python3
"""Simple test script for SkypilotCollector.

Run this to verify the collector can successfully collect metrics.
"""

import json

from devops.datadog.collectors.skypilot import SkypilotCollector


def main():
    print("Testing SkypilotCollector...")
    print("-" * 60)

    # Create collector instance
    collector = SkypilotCollector()
    print(f"✓ Created collector: {collector.name}")

    # Collect metrics safely
    print("\nCollecting metrics...")
    metrics = collector.collect_safe()

    # Display results
    print("\nCollected metrics:")
    print(json.dumps(metrics, indent=2))

    # Check for errors
    if not metrics:
        print("\n⚠ No metrics collected (may indicate Sky API is not available or jobs controller is down)")
    else:
        # Count successful vs failed metrics
        successful = sum(1 for v in metrics.values() if v is not None)
        failed = sum(1 for v in metrics.values() if v is None)

        print("\nMetrics summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        if failed > 0:
            print("\n⚠ Some metrics failed to collect (this is normal if jobs controller is down)")
        else:
            print("\n✓ All metrics collected successfully!")


if __name__ == "__main__":
    main()
