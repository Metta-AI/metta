#!/usr/bin/env python3
"""Programmatically check Datadog logs for SkyPilot jobs."""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.logs_api import LogsApi
from datadog_api_client.v2.model.logs_list_request import LogsListRequest
from datadog_api_client.v2.model.logs_query_filter import LogsQueryFilter
from softmax.aws.secrets_manager import get_secretsmanager_secret


def get_datadog_credentials() -> tuple[str, str]:
    """Get Datadog API key and app key from environment or AWS Secrets Manager.

    Uses the same pattern as datadog_agent.py - checks env var first, then AWS Secrets Manager.
    Falls back to default AWS credentials if profile-based access fails.
    """
    # Get API key - same pattern as datadog_agent.py
    api_key = os.environ.get("DD_API_KEY")
    if not api_key:
        try:
            api_key = get_secretsmanager_secret("datadog/api-key", require_exists=False)
        except Exception as e:
            # Try with default boto3 session (no profile) if profile-based fails
            try:
                import boto3
                from metta.common.util.constants import METTA_AWS_REGION
                # Temporarily unset AWS_PROFILE to use default credentials
                old_profile = os.environ.pop("AWS_PROFILE", None)
                old_default_profile = os.environ.pop("AWS_DEFAULT_PROFILE", None)
                try:
                    client = boto3.client("secretsmanager", region_name=METTA_AWS_REGION)
                    resp = client.get_secret_value(SecretId="datadog/api-key")
                    if "SecretString" in resp and resp["SecretString"]:
                        api_key = resp["SecretString"]
                finally:
                    # Restore original profile settings
                    if old_profile:
                        os.environ["AWS_PROFILE"] = old_profile
                    if old_default_profile:
                        os.environ["AWS_DEFAULT_PROFILE"] = old_default_profile
            except Exception:
                # AWS credentials not configured or secret doesn't exist
                api_key = None

    # Get app key - needed for API queries
    app_key = os.environ.get("DD_APP_KEY")
    if not app_key:
        try:
            app_key = get_secretsmanager_secret("datadog/app-key", require_exists=False)
        except Exception:
            # Try with default boto3 session (no profile) if profile-based fails
            try:
                import boto3
                from metta.common.util.constants import METTA_AWS_REGION
                # Temporarily unset AWS_PROFILE to use default credentials
                old_profile = os.environ.pop("AWS_PROFILE", None)
                old_default_profile = os.environ.pop("AWS_DEFAULT_PROFILE", None)
                try:
                    client = boto3.client("secretsmanager", region_name=METTA_AWS_REGION)
                    resp = client.get_secret_value(SecretId="datadog/app-key")
                    if "SecretString" in resp and resp["SecretString"]:
                        app_key = resp["SecretString"]
                finally:
                    # Restore original profile settings
                    if old_profile:
                        os.environ["AWS_PROFILE"] = old_profile
                    if old_default_profile:
                        os.environ["AWS_DEFAULT_PROFILE"] = old_default_profile
            except Exception:
                # AWS credentials not configured or secret doesn't exist
                app_key = None

    if not api_key:
        raise ValueError(
            "DD_API_KEY not found. Set DD_API_KEY environment variable or configure AWS credentials."
        )
    if not app_key:
        raise ValueError(
            "DD_APP_KEY not found. Set DD_APP_KEY environment variable or configure AWS credentials."
        )

    return api_key, app_key


def query_logs(
    run_id: str | None = None,
    service: str | None = None,
    hostname: str | None = None,
    task_id: str | None = None,
    hours_back: int = 1,
    limit: int = 100,
    query: str | None = None,
) -> dict[str, Any]:
    """Query Datadog logs with various filters."""
    api_key, app_key = get_datadog_credentials()

    # Configure API client
    configuration = Configuration()
    configuration.api_key["apiKeyAuth"] = api_key
    configuration.api_key["appKeyAuth"] = app_key
    configuration.server_variables["site"] = os.environ.get("DD_SITE", "datadoghq.com")

    # Build query string
    query_parts = []
    if query:
        query_parts.append(query)
    if run_id:
        query_parts.append(f"metta_run_id:{run_id}")
    if service:
        query_parts.append(f"service:{service}")
    if hostname:
        query_parts.append(f"host:{hostname}")
    if task_id:
        query_parts.append(f"skypilot_task_id:{task_id}")

    query_string = " ".join(query_parts) if query_parts else "*"

    # Set time range
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=hours_back)
    end_time = now

    # Create filter
    # Datadog API expects timestamps as strings (ISO format or Unix timestamp as string)
    filter_obj = LogsQueryFilter(
        query=query_string,
        _from=str(int(start_time.timestamp())),
        to=str(int(end_time.timestamp())),
    )

    # Create request
    request = LogsListRequest(
        filter=filter_obj,
    )

    # Query logs
    with ApiClient(configuration) as api_client:
        api_instance = LogsApi(api_client)
        response = api_instance.list_logs(body=request)

    # Get total count from metadata if available
    total_count = 0
    if hasattr(response, "meta") and response.meta:
        if hasattr(response.meta, "page") and response.meta.page:
            total_count = getattr(response.meta.page, "total_count", 0)
        elif hasattr(response.meta, "elapsed"):
            # If no page info, use length of data as estimate
            total_count = len(response.data) if hasattr(response, "data") else 0

    # Fallback to data length if no metadata
    if total_count == 0 and hasattr(response, "data"):
        total_count = len(response.data)

    return {
        "query": query_string,
        "time_range": {
            "from": start_time.isoformat(),
            "to": end_time.isoformat(),
        },
        "total_count": total_count,
        "logs": [
            {
                "timestamp": log.attributes.timestamp.isoformat() if log.attributes.timestamp else None,
                "service": log.attributes.service,
                "source": log.attributes.source,
                "host": log.attributes.host,
                "tags": log.attributes.tags if hasattr(log.attributes, "tags") else [],
                "message": log.attributes.message,
                "status": log.attributes.status if hasattr(log.attributes, "status") else None,
            }
            for log in (response.data if hasattr(response, "data") else [])
        ],
    }


def print_results(results: dict[str, Any], verbose: bool = False) -> None:
    """Print query results in a readable format."""
    print("=" * 80)
    print("DATADOG LOGS QUERY RESULTS")
    print("=" * 80)
    print(f"Query: {results['query']}")
    print(f"Time Range: {results['time_range']['from']} to {results['time_range']['to']}")
    print(f"Total Logs Found: {results['total_count']}")
    print("=" * 80)

    if results["total_count"] == 0:
        print("\n❌ No logs found matching the query.")
        print("\nPossible reasons:")
        print("  - Job hasn't started yet (wait a few minutes)")
        print("  - Time range is too narrow (try --hours-back 2)")
        print("  - Logs haven't been collected yet")
        print("  - Query filters are too restrictive")
        return

    print(f"\n📋 Showing {len(results['logs'])} logs:\n")

    for i, log in enumerate(results["logs"], 1):
        print(f"[{i}] {log['timestamp']}")
        print(f"    Service: {log['service']}")
        print(f"    Source: {log['source']}")
        print(f"    Host: {log['host']}")
        if log["tags"]:
            print(f"    Tags: {', '.join(log['tags'])}")
        if log["status"]:
            print(f"    Status: {log['status']}")
        if verbose:
            print(f"    Message: {log['message'][:200]}...")
        else:
            # Show first line of message
            first_line = log["message"].split("\n")[0][:150]
            print(f"    Message: {first_line}...")
        print()


def check_log_collection_health(run_id: str, hours_back: int = 1) -> dict[str, Any]:
    """Check if log collection is working properly for a run."""
    checks = {
        "total_logs": 0,
        "by_service": {},
        "by_source": {},
        "has_training_logs": False,
        "has_metta_run_id_tag": False,
        "has_skypilot_task_id_tag": False,
        "sample_messages": [],
    }

    # Query logs
    results = query_logs(run_id=run_id, hours_back=hours_back, limit=1000)

    checks["total_logs"] = results["total_count"]

    for log in results["logs"]:
        # Count by service
        service = log["service"] or "unknown"
        checks["by_service"][service] = checks["by_service"].get(service, 0) + 1

        # Count by source
        source = log["source"] or "unknown"
        checks["by_source"][source] = checks["by_source"].get(source, 0) + 1

        # Check for training logs
        if log["service"] == "skypilot-training" or log["source"] == "training":
            checks["has_training_logs"] = True

        # Check tags
        tags = log.get("tags", [])
        for tag in tags:
            if tag.startswith("metta_run_id:"):
                checks["has_metta_run_id_tag"] = True
            if tag.startswith("skypilot_task_id:"):
                checks["has_skypilot_task_id_tag"] = True

        # Collect sample messages
        if len(checks["sample_messages"]) < 5:
            checks["sample_messages"].append(log["message"][:200])

    return checks


def print_health_report(health: dict[str, Any], run_id: str) -> None:
    """Print a health report for log collection."""
    print("=" * 80)
    print("DATADOG LOG COLLECTION HEALTH REPORT")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Total Logs: {health['total_logs']}")
    print()

    # Service breakdown
    if health["by_service"]:
        print("Logs by Service:")
        for service, count in sorted(health["by_service"].items(), key=lambda x: -x[1]):
            print(f"  {service}: {count}")
        print()

    # Source breakdown
    if health["by_source"]:
        print("Logs by Source:")
        for source, count in sorted(health["by_source"].items(), key=lambda x: -x[1]):
            print(f"  {source}: {count}")
        print()

    # Health checks
    print("Health Checks:")
    print(f"  ✅ Training logs found: {health['has_training_logs']}")
    print(f"  ✅ metta_run_id tag present: {health['has_metta_run_id_tag']}")
    print(f"  ✅ skypilot_task_id tag present: {health['has_skypilot_task_id_tag']}")
    print()

    # Overall status
    if health["total_logs"] > 0 and health["has_training_logs"] and health["has_metta_run_id_tag"]:
        print("✅ LOG COLLECTION IS WORKING")
    elif health["total_logs"] > 0:
        print("⚠️  LOGS FOUND BUT MISSING EXPECTED TAGS OR TRAINING LOGS")
    else:
        print("❌ NO LOGS FOUND - CHECK JOB STATUS AND TIME RANGE")

    # Sample messages
    if health["sample_messages"]:
        print("\nSample Messages:")
        for i, msg in enumerate(health["sample_messages"], 1):
            print(f"  [{i}] {msg[:150]}...")
    print("=" * 80)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query Datadog logs for SkyPilot jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check logs for a specific run ID
  %(prog)s --run-id datadog_tags_test_20251118_231243

  # Check logs with health report
  %(prog)s --run-id datadog_tags_test_20251118_231243 --health

  # Query by service
  %(prog)s --service skypilot-training

  # Custom query
  %(prog)s --query "service:skypilot-training AND source:training"

  # Check logs from last 2 hours
  %(prog)s --run-id my_run --hours-back 2
        """,
    )
    parser.add_argument("--run-id", help="Metta run ID to filter by")
    parser.add_argument("--service", help="Service name to filter by (e.g., skypilot-training)")
    parser.add_argument("--hostname", help="Hostname to filter by")
    parser.add_argument("--task-id", help="SkyPilot task ID to filter by")
    parser.add_argument("--query", help="Custom Datadog query string")
    parser.add_argument("--hours-back", type=int, default=1, help="Hours to look back (default: 1)")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of logs to return (default: 100)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full log messages")
    parser.add_argument("--health", action="store_true", help="Show health report instead of raw logs")

    args = parser.parse_args()

    try:
        if args.health:
            if not args.run_id:
                print("Error: --health requires --run-id")
                return 1
            health = check_log_collection_health(args.run_id, hours_back=args.hours_back)
            print_health_report(health, args.run_id)
        else:
            results = query_logs(
                run_id=args.run_id,
                service=args.service,
                hostname=args.hostname,
                task_id=args.task_id,
                hours_back=args.hours_back,
                limit=args.limit,
                query=args.query,
            )
            print_results(results, verbose=args.verbose)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

