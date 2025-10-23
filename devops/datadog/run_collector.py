#!/usr/bin/env python3
"""Run a Datadog metrics collector.

This script runs in the project environment and has access to all dependencies.
Used by cli.py to execute collectors.

Usage:
    uv run python devops/datadog/run_collector.py github [--push] [--verbose]
"""

import argparse
import json
import logging
import os
import sys


def run_github_collector(push: bool = False, verbose: bool = False, json_output: bool = False) -> dict:
    """Run the GitHub metrics collector."""
    from devops.datadog.collectors.github import GitHubCollector
    from devops.datadog.common.datadog_client import DatadogClient

    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Helper to print messages (stderr for json mode, stdout otherwise)
    def print_status(msg):
        if json_output:
            print(msg, file=sys.stderr)
        else:
            print(msg)

    # Get GitHub token
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        # Try AWS Secrets Manager
        try:
            from softmax.aws.secrets_manager import get_secretsmanager_secret

            github_token = get_secretsmanager_secret("github/dashboard-token")
        except Exception as e:
            print(f"Error: GitHub token not found. {e}", file=sys.stderr)
            sys.exit(1)

    # Get GitHub org and repo
    github_org = os.getenv("GITHUB_ORG", "PufferAI")
    github_repo = os.getenv("GITHUB_REPO", "metta")

    print_status(f"Collecting metrics from {github_org}/{github_repo}...")

    # Create and run collector
    collector = GitHubCollector(
        organization=github_org,
        repository=github_repo,
        github_token=github_token,
    )

    metrics = collector.collect_safe()

    if not metrics:
        print("Warning: No metrics collected", file=sys.stderr)
        sys.exit(1)

    print_status(f"Collected {len(metrics)} metrics")

    # Push to Datadog if requested
    if push:
        api_key = os.getenv("DD_API_KEY")
        app_key = os.getenv("DD_APP_KEY")
        site = os.getenv("DD_SITE", "datadoghq.com")

        if not api_key or not app_key:
            print("Error: DD_API_KEY and DD_APP_KEY must be set for --push", file=sys.stderr)
            sys.exit(1)

        print_status("Pushing metrics to Datadog...")

        # Create Datadog client
        datadog_client = DatadogClient(
            api_key=api_key,
            app_key=app_key,
            site=site,
        )

        # Format metrics for submission
        metrics_to_submit = []
        for name, value in metrics.items():
            if value is not None:
                metrics_to_submit.append(
                    {
                        "metric": name,
                        "value": value,
                        "type": "gauge",
                        "tags": ["source:github-collector", "env:production"],
                    }
                )

        # Submit metrics
        success = datadog_client.submit_metrics_batch(metrics_to_submit)

        if success:
            print_status(f"Successfully pushed {len(metrics_to_submit)} metrics to Datadog")
        else:
            print("Error: Failed to push metrics to Datadog", file=sys.stderr)
            sys.exit(1)

    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run a Datadog metrics collector")
    parser.add_argument("collector", help="Collector name (e.g., 'github')")
    parser.add_argument("--push", action="store_true", help="Push metrics to Datadog")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")

    args = parser.parse_args()

    if args.collector == "github":
        try:
            metrics = run_github_collector(push=args.push, verbose=args.verbose, json_output=args.json)

            if args.json:
                print(json.dumps(metrics, indent=2, sort_keys=True))
            else:
                print("\nCollected metrics:")
                for key, value in sorted(metrics.items()):
                    if value is not None:
                        if isinstance(value, float):
                            print(f"  {key}: {value:.2f}")
                        else:
                            print(f"  {key}: {value}")

        except KeyboardInterrupt:
            print("\nInterrupted", file=sys.stderr)
            sys.exit(130)

        except Exception as e:
            print(f"Error: Failed to run {args.collector} collector: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)

    else:
        print(f"Error: Unknown collector '{args.collector}'", file=sys.stderr)
        print("Available collectors: github", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
