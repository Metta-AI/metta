#!/usr/bin/env python3
"""Run a Datadog metrics collector.

This script runs in the project environment and has access to all dependencies.
Used by cli.py to execute collectors.

Usage:
    uv run python devops/datadog/run_collector.py github [--push] [--verbose]
    uv run python devops/datadog/run_collector.py skypilot [--push] [--verbose]
    uv run python devops/datadog/run_collector.py asana [--push] [--verbose]
    uv run python devops/datadog/run_collector.py ec2 [--push] [--verbose]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def load_env_file() -> None:
    """Load .env file from devops/datadog/ if it exists."""
    # Try to find .env file
    env_paths = [
        Path("devops/datadog/.env"),  # From repo root
        Path(".env"),  # From devops/datadog/
    ]

    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            # Only set if not already in environment
                            if key.strip() not in os.environ:
                                os.environ[key.strip()] = value.strip()
            return


def get_datadog_credentials() -> tuple[str, str, str]:
    """Get Datadog credentials from environment or AWS Secrets Manager.

    Returns:
        Tuple of (api_key, app_key, site)

    Raises:
        SystemExit if credentials cannot be found
    """
    api_key = os.getenv("DD_API_KEY")
    app_key = os.getenv("DD_APP_KEY")
    site = os.getenv("DD_SITE", "datadoghq.com")

    # Fetch from AWS Secrets Manager if not in environment
    if not api_key:
        try:
            from softmax.aws.secrets_manager import get_secretsmanager_secret

            api_key = get_secretsmanager_secret("datadog/api-key")
        except Exception as e:
            print(f"Error: DD_API_KEY not found in environment or AWS Secrets Manager. {e}", file=sys.stderr)
            sys.exit(1)

    if not app_key:
        try:
            from softmax.aws.secrets_manager import get_secretsmanager_secret

            app_key = get_secretsmanager_secret("datadog/app-key")
        except Exception as e:
            print(f"Error: DD_APP_KEY not found in environment or AWS Secrets Manager. {e}", file=sys.stderr)
            sys.exit(1)

    return api_key, app_key, site


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
        api_key, app_key, site = get_datadog_credentials()
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


def run_skypilot_collector(push: bool = False, verbose: bool = False, json_output: bool = False) -> dict:
    """Run the Skypilot metrics collector."""
    from devops.datadog.collectors.skypilot import SkypilotCollector
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

    print_status("Collecting metrics from Skypilot...")

    # Create and run collector
    collector = SkypilotCollector()
    metrics = collector.collect_safe()

    if not metrics:
        print("Warning: No metrics collected", file=sys.stderr)
        sys.exit(1)

    print_status(f"Collected {len(metrics)} metrics")

    # Push to Datadog if requested
    if push:
        api_key, app_key, site = get_datadog_credentials()
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
                        "tags": ["source:skypilot-collector", "env:production"],
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


def run_asana_collector(push: bool = False, verbose: bool = False, json_output: bool = False) -> dict:
    """Run the Asana metrics collector."""
    from devops.datadog.collectors.asana import AsanaCollector
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

    # Get Asana credentials
    asana_token = os.getenv("ASANA_ACCESS_TOKEN")
    if not asana_token:
        # Try AWS Secrets Manager
        try:
            from softmax.aws.secrets_manager import get_secretsmanager_secret

            asana_token = get_secretsmanager_secret("asana/access-token")
        except Exception as e:
            print(f"Error: Asana token not found. {e}", file=sys.stderr)
            sys.exit(1)

    # Get workspace and bugs project IDs
    # Priority: environment variables > AWS Secrets Manager
    workspace_gid = os.getenv("ASANA_WORKSPACE_GID")
    if not workspace_gid:
        # Try AWS Secrets Manager
        try:
            from softmax.aws.secrets_manager import get_secretsmanager_secret

            workspace_gid = get_secretsmanager_secret("asana/workspace-gid")
        except Exception as e:
            print(f"Error: ASANA_WORKSPACE_GID not found in environment or AWS Secrets Manager. {e}", file=sys.stderr)
            sys.exit(1)

    bugs_project_gid = os.getenv("ASANA_BUGS_PROJECT_GID")
    if not bugs_project_gid:
        # Try AWS Secrets Manager (optional - not required)
        try:
            from softmax.aws.secrets_manager import get_secretsmanager_secret

            bugs_project_gid = get_secretsmanager_secret("asana/bugs-project-gid")
        except Exception:
            # Bugs project is optional, so don't fail if not found
            pass

    print_status(f"Collecting metrics from Asana workspace {workspace_gid}...")

    # Create and run collector
    collector = AsanaCollector(
        access_token=asana_token,
        workspace_gid=workspace_gid,
        bugs_project_gid=bugs_project_gid,
    )

    metrics = collector.collect_safe()

    if not metrics:
        print("Warning: No metrics collected", file=sys.stderr)
        sys.exit(1)

    print_status(f"Collected {len(metrics)} metrics")

    # Push to Datadog if requested
    if push:
        api_key, app_key, site = get_datadog_credentials()
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
                        "tags": ["source:asana-collector", "env:production"],
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


def run_health_fom_collector(push: bool = False, verbose: bool = False, json_output: bool = False) -> dict:
    """Run the Health FoM collector."""
    from devops.datadog.collectors.health_fom import HealthFomCollector
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

    print_status("Collecting FoM metrics from Datadog...")

    # Create and run collector
    collector = HealthFomCollector()
    metrics = collector.collect_safe()

    if not metrics:
        print("Warning: No metrics collected", file=sys.stderr)
        sys.exit(1)

    print_status(f"Collected {len(metrics)} FoM metrics")

    # Push to Datadog if requested
    if push:
        api_key, app_key, site = get_datadog_credentials()
        print_status("Pushing FoM metrics to Datadog...")

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
                        "tags": ["source:health-fom-collector", "env:production"],
                    }
                )

        # Submit metrics
        success = datadog_client.submit_metrics_batch(metrics_to_submit)

        if success:
            print_status(f"Successfully pushed {len(metrics_to_submit)} FoM metrics to Datadog")
        else:
            print("Error: Failed to push FoM metrics to Datadog", file=sys.stderr)
            sys.exit(1)

    return metrics


def run_ec2_collector(push: bool = False, verbose: bool = False, json_output: bool = False) -> dict:
    """Run the EC2 metrics collector."""
    from devops.datadog.collectors.ec2 import EC2Collector
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

    # Get AWS region (optional, defaults to us-east-1)
    region = os.getenv("AWS_REGION", "us-east-1")

    print_status(f"Collecting metrics from EC2 in {region}...")

    # Create and run collector
    collector = EC2Collector(region=region)
    metrics = collector.collect_safe()

    if not metrics:
        print("Warning: No metrics collected", file=sys.stderr)
        sys.exit(1)

    print_status(f"Collected {len(metrics)} metrics")

    # Push to Datadog if requested
    if push:
        api_key, app_key, site = get_datadog_credentials()
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
                        "tags": ["source:ec2-collector", "env:production", f"region:{region}"],
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


def run_wandb_collector(push: bool = False, verbose: bool = False, json_output: bool = False) -> dict:
    """Run the WandB metrics collector."""
    from devops.datadog.collectors.wandb import WandBCollector
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

    # Get WandB credentials
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        # Try AWS Secrets Manager
        try:
            from softmax.aws.secrets_manager import get_secretsmanager_secret

            wandb_api_key = get_secretsmanager_secret("wandb/api-key")
        except Exception as e:
            print(f"Error: WANDB_API_KEY not found in environment or AWS Secrets Manager. {e}", file=sys.stderr)
            sys.exit(1)

    # Get WandB entity and project
    entity = os.getenv("WANDB_ENTITY", "pufferai")
    project = os.getenv("WANDB_PROJECT", "metta")

    print_status(f"Collecting metrics from WandB ({entity}/{project})...")

    # Create and run collector
    collector = WandBCollector(api_key=wandb_api_key, entity=entity, project=project)
    metrics = collector.collect_safe()

    if not metrics:
        print("Warning: No metrics collected", file=sys.stderr)
        sys.exit(1)

    print_status(f"Collected {len(metrics)} metrics\n")

    # Display metrics
    if not json_output:
        print_status("Collected metrics:")
        for key, value in sorted(metrics.items()):
            print_status(f"  {key}: {value}")
    else:
        # JSON output
        print(json.dumps(metrics, indent=2, default=str))

    # Push to Datadog if requested
    if push:
        print_status("\nPushing metrics to Datadog...")
        api_key, app_key, site = get_datadog_credentials()

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
                        "tags": ["source:wandb-collector", "env:production"],
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
    # Load .env file if it exists
    load_env_file()

    parser = argparse.ArgumentParser(description="Run a Datadog metrics collector")
    parser.add_argument(
        "collector", help="Collector name (e.g., 'github', 'skypilot', 'asana', 'health_fom', 'ec2', 'wandb')"
    )
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

    elif args.collector == "skypilot":
        try:
            metrics = run_skypilot_collector(push=args.push, verbose=args.verbose, json_output=args.json)

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

    elif args.collector == "asana":
        try:
            metrics = run_asana_collector(push=args.push, verbose=args.verbose, json_output=args.json)

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

    elif args.collector == "health_fom":
        try:
            metrics = run_health_fom_collector(push=args.push, verbose=args.verbose, json_output=args.json)

            if args.json:
                print(json.dumps(metrics, indent=2, sort_keys=True))
            else:
                print("\nCollected FoM metrics:")
                for key, value in sorted(metrics.items()):
                    if value is not None:
                        if isinstance(value, float):
                            print(f"  {key}: {value:.3f}")
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

    elif args.collector == "ec2":
        try:
            metrics = run_ec2_collector(push=args.push, verbose=args.verbose, json_output=args.json)

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

    elif args.collector == "wandb":
        try:
            metrics = run_wandb_collector(push=args.push, verbose=args.verbose, json_output=args.json)

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
        print("Available collectors: github, skypilot, asana, health_fom, ec2, wandb", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
