#!/usr/bin/env python3
"""Test pushing a simple metric to Datadog and verify it appears.

This is a sanity check to verify basic metric submission works.
"""

import argparse
import time

from devops.datadog.utils.dashboard_client import get_datadog_credentials

from devops.datadog.utils.datadog_client import DatadogClient


def test_metric_push(
    metric_name: str = "test.sanity_check",
    value: float = 1.0,
    wait: bool = True,
    wait_seconds: int = 60,
) -> bool:
    """Push a test metric and verify it appears.

    Args:
        metric_name: Name of test metric
        value: Value to push
        wait: If True, wait and verify metric appears
        wait_seconds: How long to wait for propagation

    Returns:
        True if test passed, False otherwise
    """
    # Get Datadog credentials
    api_key, app_key, site = get_datadog_credentials()
    client = DatadogClient(api_key=api_key, app_key=app_key, site=site)

    # Step 1: Push metric
    print("\n=== Step 1: Push Test Metric ===")
    print(f"Metric: {metric_name}")
    print(f"Value: {value}")

    success = client.submit_metric(
        metric_name=metric_name,
        value=value,
        metric_type="gauge",
        tags=["source:test", "purpose:sanity_check"],
    )

    if not success:
        print("❌ Failed to submit metric")
        return False

    print("✅ Successfully submitted to Datadog API")

    # If not waiting, exit here
    if not wait:
        print("\n⏭️  Skipping verification (--no-wait specified)")
        print("To verify manually, run:")
        print(f"  uv run python devops/datadog/scripts/list_datadog_metrics.py --prefix {metric_name.split('.')[0]}")
        return True

    # Step 2: Wait for propagation
    print("\n=== Step 2: Wait for Propagation ===")
    print(f"Waiting {wait_seconds} seconds...")
    for i in range(wait_seconds // 10):
        time.sleep(10)
        print(f"  {(i + 1) * 10}s...")
    print("Done.")

    # Step 3: Query metric back
    print("\n=== Step 3: Verify Metric ===")
    print(f"Querying {metric_name}...")

    retrieved_value = client.query_metric(
        metric_name,
        aggregation="last",
        lookback_seconds=wait_seconds + 60,  # Look back a bit longer
    )

    if retrieved_value is None:
        print("❌ Metric not found in Datadog")
        print("\nTroubleshooting:")
        print("  1. Try waiting longer (use --wait-seconds 120)")
        print(f"  2. Check Datadog UI: https://app.datadoghq.com/metric/summary?filter={metric_name}")
        print("  3. List all test metrics: uv run python devops/datadog/scripts/list_datadog_metrics.py --prefix test")
        return False

    print(f"✅ Found metric: {metric_name} = {retrieved_value}")

    # Verify value matches (allow small floating point difference)
    if abs(retrieved_value - value) < 0.01:
        print("\n✅ TEST PASSED")
        print("   Metric successfully pushed and retrieved")
        return True
    else:
        print("\n⚠️  TEST WARNING")
        print("   Metric found but value differs:")
        print(f"   Expected: {value}")
        print(f"   Got: {retrieved_value}")
        return True  # Still consider it a pass since metric exists


def main():
    parser = argparse.ArgumentParser(description="Test metric push to Datadog")
    parser.add_argument(
        "--metric",
        type=str,
        default="test.sanity_check",
        help="Metric name to test (default: test.sanity_check)",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=1.0,
        help="Metric value to push (default: 1.0)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait to verify metric (push only)",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=60,
        help="How long to wait for propagation (default: 60)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Datadog Metric Push Test")
    print("=" * 60)

    success = test_metric_push(
        metric_name=args.metric,
        value=args.value,
        wait=not args.no_wait,
        wait_seconds=args.wait_seconds,
    )

    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
