#!/usr/bin/env python3
"""Run a Datadog metrics collector.

This script runs in the project environment and has access to all dependencies.
Used by cli.py to execute collectors.

Usage:
    uv run python devops/datadog/scripts/run_collector.py github [--push] [--verbose]
    uv run python devops/datadog/scripts/run_collector.py skypilot [--push] [--verbose]
    uv run python devops/datadog/scripts/run_collector.py asana [--push] [--verbose]
    uv run python devops/datadog/scripts/run_collector.py ec2 [--push] [--verbose]
"""

import argparse
import json
import logging
import os
import sys

from devops.datadog.collectors import (
    AsanaCollector,
    EC2Collector,
    GitHubCollector,
    HealthFomCollector,
    KubernetesCollector,
    SkypilotCollector,
    WandBCollector,
)
from devops.datadog.utils.datadog_client import DatadogClient

# Collector registry with configuration
COLLECTORS = {
    "github": {
        "class": GitHubCollector,
        "source": "github-collector",
        "description": "GitHub metrics",
    },
    "skypilot": {
        "class": SkypilotCollector,
        "source": "skypilot-collector",
        "description": "Skypilot job metrics",
    },
    "asana": {
        "class": AsanaCollector,
        "source": "asana-collector",
        "description": "Asana project metrics",
    },
    "ec2": {
        "class": EC2Collector,
        "source": "ec2-collector",
        "description": "AWS EC2 metrics",
    },
    "wandb": {
        "class": WandBCollector,
        "source": "wandb-collector",
        "description": "WandB training metrics",
    },
    "kubernetes": {
        "class": KubernetesCollector,
        "source": "kubernetes-collector",
        "description": "Kubernetes efficiency metrics",
    },
    "health_fom": {
        "class": HealthFomCollector,
        "source": "health-fom-collector",
        "description": "Health FoM metrics",
    },
}


def setup_logging(verbose: bool) -> None:
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_credential(env_var: str, secret_key: str, required: bool = True) -> str | None:
    """Get credential from environment or AWS Secrets Manager.

    Args:
        env_var: Environment variable name
        secret_key: AWS Secrets Manager key
        required: If True, exit on failure

    Returns:
        Credential value or None
    """
    value = os.getenv(env_var)
    if value:
        return value

    # Try AWS Secrets Manager
    try:
        from devops.datadog.utils.secrets import get_secretsmanager_secret

        return get_secretsmanager_secret(secret_key)
    except Exception as e:
        if required:
            raise RuntimeError(f"{env_var} not found in environment or AWS Secrets Manager. {e}") from e
        return None


def get_collector_instance(collector_name: str):
    """Create collector instance with appropriate credentials.

    Args:
        collector_name: Name of collector to instantiate

    Returns:
        Collector instance
    """
    if collector_name == "github":
        token = get_credential("GITHUB_TOKEN", "github/dashboard-token")
        org = os.getenv("GITHUB_ORG", "PufferAI")
        repo = os.getenv("GITHUB_REPO", "metta")
        return GitHubCollector(organization=org, repository=repo, github_token=token)

    elif collector_name == "asana":
        token = get_credential("ASANA_ACCESS_TOKEN", "asana/access-token")
        workspace = get_credential("ASANA_WORKSPACE_GID", "asana/workspace-gid")
        bugs_project = get_credential("ASANA_BUGS_PROJECT_GID", "asana/bugs-project-gid", required=False)
        return AsanaCollector(access_token=token, workspace_gid=workspace, bugs_project_gid=bugs_project)

    elif collector_name == "ec2":
        region = os.getenv("AWS_REGION", "us-east-1")
        return EC2Collector(region=region)

    elif collector_name == "wandb":
        api_key = get_credential("WANDB_API_KEY", "wandb/api-key")
        entity = os.getenv("WANDB_ENTITY", "metta-research")
        project = os.getenv("WANDB_PROJECT", "metta")
        return WandBCollector(api_key=api_key, entity=entity, project=project)

    elif collector_name in ["skypilot", "kubernetes", "health_fom"]:
        # These collectors don't need credentials
        collector_class = COLLECTORS[collector_name]["class"]
        return collector_class()

    else:
        raise ValueError(f"Unknown collector: {collector_name}")


def get_extra_tags(collector_name: str) -> list[str]:
    """Get collector-specific tags.

    Args:
        collector_name: Name of collector

    Returns:
        List of additional tags
    """
    tags = []

    if collector_name == "ec2":
        region = os.getenv("AWS_REGION", "us-east-1")
        tags.append(f"region:{region}")

    elif collector_name == "kubernetes":
        cluster = os.getenv("K8S_CLUSTER_NAME", "main")
        tags.append(f"cluster:{cluster}")

    return tags


def push_metrics_to_datadog(metrics: dict, source_tag: str, extra_tags: list[str] | None = None) -> bool:
    """Push metrics to Datadog.

    Args:
        metrics: Dictionary of metric name to value
        source_tag: Source tag for metrics (e.g., "github-collector")
        extra_tags: Additional tags to include

    Returns:
        True if successful, False otherwise
    """
    # Get Datadog credentials
    api_key = get_credential("DD_API_KEY", "datadog/api-key")
    app_key = get_credential("DD_APP_KEY", "datadog/app-key")
    site = os.getenv("DD_SITE", "datadoghq.com")

    # Create client
    client = DatadogClient(api_key=api_key, app_key=app_key, site=site)

    # Format metrics
    base_tags = [f"source:{source_tag}", "env:production"]
    if extra_tags:
        base_tags.extend(extra_tags)

    metrics_to_submit = []
    for name, value in metrics.items():
        if value is None:
            continue

        # Handle new per-run metric format: list of (value, tags) tuples
        if isinstance(value, list) and value and isinstance(value[0], tuple):
            for metric_value, metric_tags in value:
                # Combine base tags with metric-specific tags
                all_tags = base_tags + metric_tags
                metrics_to_submit.append(
                    {
                        "metric": name,
                        "value": metric_value,
                        "type": "gauge",
                        "tags": all_tags,
                    }
                )
        # Handle traditional single-value format
        else:
            metrics_to_submit.append(
                {
                    "metric": name,
                    "value": value,
                    "type": "gauge",
                    "tags": base_tags,
                }
            )

    # Submit
    return client.submit_metrics_batch(metrics_to_submit)


def display_metrics(metrics: dict, json_output: bool = False, precision: int = 2) -> None:
    """Display collected metrics.

    Args:
        metrics: Dictionary of metric name to value or list of (value, tags) tuples
        json_output: If True, output as JSON
        precision: Decimal precision for float values
    """
    if json_output:
        # Convert tuple lists to serializable format for JSON
        serializable = {}
        for key, value in metrics.items():
            if isinstance(value, list) and value and isinstance(value[0], tuple):
                # Convert to list of dicts for JSON
                serializable[key] = [{"value": v, "tags": t} for v, t in value]
            else:
                serializable[key] = value
        print(json.dumps(serializable, indent=2, sort_keys=True, default=str))
    else:
        print("\nCollected metrics:")
        for key, value in sorted(metrics.items()):
            if value is None:
                continue
            # Handle per-run metrics (list of tuples)
            if isinstance(value, list) and value and isinstance(value[0], tuple):
                print(f"  {key}: {len(value)} data points")
                for i, (val, tags) in enumerate(value[:3], 1):  # Show first 3
                    tags_str = ", ".join(tags)
                    if isinstance(val, float):
                        print(f"    [{i}] {val:.{precision}f} ({tags_str})")
                    else:
                        print(f"    [{i}] {val} ({tags_str})")
                if len(value) > 3:
                    print(f"    ... and {len(value) - 3} more")
            # Handle single-value metrics
            else:
                if isinstance(value, float):
                    print(f"  {key}: {value:.{precision}f}")
                else:
                    print(f"  {key}: {value}")


def print_status(msg: str, json_output: bool = False) -> None:
    """Print status message (to stderr in JSON mode).

    Args:
        msg: Message to print
        json_output: If True, print to stderr
    """
    if json_output:
        print(msg, file=sys.stderr)
    else:
        print(msg)


def run_collector(collector_name: str, push: bool = False, verbose: bool = False, json_output: bool = False) -> dict:
    """Run a collector and optionally push metrics to Datadog.

    Args:
        collector_name: Name of collector to run
        push: If True, push metrics to Datadog
        verbose: If True, enable verbose logging
        json_output: If True, output metrics as JSON

    Returns:
        Dictionary of collected metrics
    """
    setup_logging(verbose)

    if collector_name not in COLLECTORS:
        available = ", ".join(COLLECTORS.keys())
        print(f"Error: Unknown collector '{collector_name}'", file=sys.stderr)
        print(f"Available collectors: {available}", file=sys.stderr)
        sys.exit(1)

    config = COLLECTORS[collector_name]
    description = config["description"]

    print_status(f"Collecting {description}...", json_output)

    # Create and run collector
    collector = get_collector_instance(collector_name)
    metrics = collector.collect_safe()

    if not metrics:
        print("Warning: No metrics collected", file=sys.stderr)
        sys.exit(1)

    print_status(f"Collected {len(metrics)} metrics", json_output)

    # Push if requested
    if push:
        print_status("Pushing metrics to Datadog...", json_output)
        extra_tags = get_extra_tags(collector_name)
        success = push_metrics_to_datadog(metrics, config["source"], extra_tags)

        if success:
            print_status(f"Successfully pushed {len(metrics)} metrics to Datadog", json_output)
        else:
            print("Error: Failed to push metrics to Datadog", file=sys.stderr)
            sys.exit(1)

    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run a Datadog metrics collector")
    parser.add_argument(
        "collector",
        choices=list(COLLECTORS.keys()),
        help="Collector name",
    )
    parser.add_argument("--push", action="store_true", help="Push metrics to Datadog")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")

    args = parser.parse_args()

    try:
        # Run collector
        metrics = run_collector(
            collector_name=args.collector,
            push=args.push,
            verbose=args.verbose,
            json_output=args.json,
        )

        # Display results (with special precision for health_fom)
        precision = 3 if args.collector == "health_fom" else 2
        display_metrics(metrics, json_output=args.json, precision=precision)

    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"Error: Failed to run {args.collector} collector: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
