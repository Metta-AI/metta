#!/usr/bin/env -S uv run
import argparse
import logging
import os
import subprocess
import time

import sky
import sky.exceptions
import yaml

import gitta as git
from devops.skypilot.utils.cost_monitor import get_instance_cost
from devops.skypilot.utils.job_helpers import set_task_secrets
from metta.common.util.cli import spinner
from metta.common.util.text_styles import blue, bold, cyan, green, red, yellow


class CredentialWarningHandler(logging.Handler):
    """Custom handler to intercept and reformat botocore credential warnings."""

    def emit(self, record):
        if record.levelno == logging.WARNING and "credential" in record.getMessage().lower():
            # Extract just the warning message without the traceback
            message = record.getMessage()

            # Check for specific credential-related messages
            if "refresh failed" in message or "Token has expired" in message:
                print(f"\n{yellow('⚠️  AWS credentials need refresh')}")
                print("   Your AWS session is expiring but still functional")
                print(f"   Run {green('aws sso login')} when convenient to refresh\n")


# Set up custom logging for botocore.credentials
credentials_logger = logging.getLogger("botocore.credentials")
credentials_logger.setLevel(logging.WARNING)
credentials_logger.addHandler(CredentialWarningHandler())
# Prevent propagation to avoid duplicate output
credentials_logger.propagate = False


def get_existing_clusters():
    with spinner("Fetching existing clusters", style=cyan):
        request_id = sky.status()
        cluster_records = sky.get(request_id)
    return cluster_records


def get_next_name(cluster_records):
    names = [record["name"] for record in cluster_records]
    username = os.environ["USER"]
    for i in range(1, 100):
        name = f"{username}-sandbox-{i}"
        if name not in names:
            return name
    raise ValueError("No available sandbox name found")


def load_sandbox_config(config_path: str):
    """Load the sandbox YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_current_git_ref():
    """Get the current git branch or commit hash."""
    try:
        return git.get_current_branch()
    except (git.GitError, ValueError):
        return "main"  # Fallback to main


def get_gpu_instance_info(num_gpus: int, gpu_type: str = "L4", region: str = "us-east-1", cloud: str = "aws"):
    """
    Determine the instance type and cost for GPU instances.

    Args:
        num_gpus: Number of GPUs requested
        gpu_type: Type of GPU (default: L4)
        region: Cloud region
        cloud: Cloud provider (default: aws)

    Returns:
        Tuple of (instance_type, region, hourly_cost)
    """
    if cloud.lower() != "aws":
        print(f"Warning: Cost calculation only supported for AWS, not {cloud}")
        return None, region, None

    # Map GPU configurations to typical AWS instance types
    gpu_instance_map = {
        ("L4", 1): "g6.xlarge",
        ("L4", 2): "g6.2xlarge",
        ("L4", 4): "g6.4xlarge",
        ("L4", 8): "g6.8xlarge",
    }

    # Get the instance type based on GPU configuration
    instance_type = gpu_instance_map.get((gpu_type, num_gpus))
    if not instance_type:
        print(f"Warning: No instance mapping for {num_gpus} {gpu_type} GPU(s), using g6.xlarge as estimate")
        instance_type = "g6.xlarge"
        estimated_multiplier = num_gpus
    else:
        estimated_multiplier = 1

    # Try to calculate cost
    hourly_cost = None
    try:
        with spinner(f"Calculating cost for {instance_type}", style=cyan):
            hourly_cost = get_instance_cost(instance_type=instance_type, region=region, use_spot=False)

        if hourly_cost is not None:
            hourly_cost *= estimated_multiplier

    except Exception as e:
        print(f"\n{yellow('⚠️  Unable to calculate cost:')} {str(e)}")
        print("   Continuing without cost information...\n")

    return instance_type, region, hourly_cost


def main():
    parser = argparse.ArgumentParser(
        prog="sandbox.py",
        description="""
Manage GPU development sandboxes on SkyPilot.

When run without arguments, displays existing sandboxes and management commands.
Use --new to launch a new sandbox cluster.
        """.strip(),
        epilog="""
Examples:
  %(prog)s              # Show existing sandboxes and management commands
  %(prog)s --check      # Check for active sandboxes and exit
  %(prog)s --new        # Launch a new sandbox with 1 GPU
  %(prog)s --new --gpus 4  # Launch a new sandbox with 4 GPUs
  %(prog)s --new --git-ref feature-branch  # Launch with specific git branch
  %(prog)s --new --wait-timeout 600  # Wait up to 10 minutes for cluster to be ready

Common management commands:
  ssh <sandbox-name>     # Connect to a running sandbox
  sky stop <name>        # Stop a sandbox (keeps data)
  sky start <name>       # Restart a stopped sandbox
  sky down <name>        # Delete a sandbox completely
  sky logs <name>        # Check cluster logs
  sky launch -c <name> --no-setup  # Retry launch for stuck clusters
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--git-ref", type=str, default=None, help="Git branch or commit to deploy (default: current branch)"
    )
    parser.add_argument("--new", action="store_true", help="Launch a new sandbox cluster")
    parser.add_argument("--check", action="store_true", help="Check for existing sandboxes and exit")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument(
        "--retry-until-up", action="store_true", help="Keep retrying until cluster is successfully launched"
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=300,
        help="Timeout in seconds to wait for cluster to reach UP state (default: 300)",
    )

    args = parser.parse_args()

    # Get git ref - use current branch/commit if not specified
    git_ref = args.git_ref or get_current_git_ref()

    existing_clusters = get_existing_clusters()

    # Handle --check mode
    if args.check:
        username = os.environ.get("USER", "unknown")
        user_sandboxes = [c for c in existing_clusters if c["name"].startswith(f"{username}-sandbox-")]

        if not user_sandboxes:
            print(f"{green('✓')} No active sandboxes found for user {bold(username)}")
            return 0

        print(f"{bold(f'Found {len(user_sandboxes)} sandbox(es) for user {username}:')}")

        active_count = 0
        stopped_count = 0
        init_count = 0

        for cluster in user_sandboxes:
            status = cluster["status"].name
            if status == "UP":
                active_count += 1
                color_func = green
                status_msg = "running"
            elif status == "STOPPED":
                stopped_count += 1
                color_func = red
                status_msg = "stopped"
            elif status == "INIT":
                init_count += 1
                color_func = cyan
                status_msg = "launching"
            else:
                color_func = yellow
                status_msg = status.lower()

            gpu_info = ""
            if "handle" in cluster and cluster["handle"]:
                resources = cluster["handle"].launched_resources
                if resources and resources.accelerators:
                    gpu_info = f" [{resources.accelerators}]"

            print(f"  • {color_func(cluster['name'])} ({status_msg}){gpu_info}")

        # Summary
        print(f"\n{bold('Summary:')}")
        if active_count > 0:
            print(f"  {green(f'{active_count} running')}")
        if stopped_count > 0:
            print(f"  {red(f'{stopped_count} stopped')}")
        if init_count > 0:
            print(f"  {cyan(f'{init_count} launching')}")

        # Return exit code: 0 if any sandboxes exist, 1 if none
        return 0

    if existing_clusters and not args.new:
        print(f"You already have {len(existing_clusters)} sandbox(es) running:")
        for cluster in existing_clusters:
            message = ""
            color_func = yellow  # default color
            if cluster["status"].name == "INIT":
                message = " (launching - may take several minutes)"
                color_func = cyan
            elif cluster["status"].name == "STOPPED":
                message = " (stopped)"
                color_func = red
            elif cluster["status"].name == "UP":
                message = " (running)"
                color_func = green

            gpu_info = ""
            if "handle" in cluster and cluster["handle"]:
                resources = cluster["handle"].launched_resources
                if resources and resources.accelerators:
                    gpu_info = f" [{resources.accelerators}]"

            print(f"  {color_func(cluster['name'])}{message}{gpu_info}")

            # Additional guidance for INIT state clusters
            if cluster["status"].name == "INIT":
                cluster_name_ref = cluster["name"]
                print(
                    f"    💡 If stuck in INIT for >10min, try: {green(f'sky launch -c {cluster_name_ref} --no-setup')}"
                )
                print(f"    📊 Check logs: {green(f'sky logs {cluster_name_ref}')}")

        first_cluster_name = existing_clusters[0]["name"]
        first_stopped_cluster_name = next(
            (cluster["name"] for cluster in existing_clusters if cluster["status"].name == "STOPPED"), None
        )

        print("\n📦 Manage sandboxes:")
        print(f"  Launch new:     {green('./devops/skypilot/sandbox.py --new')}")
        print(f"  Connect:        {green(f'ssh {first_cluster_name}')}")
        if first_stopped_cluster_name:
            print(f"  Restart:        {green(f'sky start {first_stopped_cluster_name}')}")
        print(f"  Stop:           {green(f'sky stop {first_cluster_name}')}")
        print(f"  Delete:         {red(f'sky down {first_cluster_name}')}")

        return 0

    cluster_name = get_next_name(existing_clusters)
    print(f"\n🚀 Launching {blue(cluster_name)} with {bold(str(args.gpus))} L4 GPU(s)")
    print(f"🔌 Git ref: {cyan(git_ref)}")

    # Load the sandbox configuration
    config_path = "./devops/skypilot/launch/sandbox.yaml"
    config = load_sandbox_config(config_path)

    # Extract cloud configuration
    resources = config.get("resources", {})
    cloud = resources.get("cloud", "aws")
    region = resources.get("region", "us-east-1")

    # Parse GPU type from the config (e.g., "L4:1" -> "L4")
    accelerators_str = resources.get("accelerators", "L4:1")
    gpu_type = accelerators_str.split(":")[0]

    # Get instance type and calculate cost
    instance_type, region, hourly_cost = get_gpu_instance_info(args.gpus, gpu_type, region, cloud)

    if instance_type:
        print(f"Instance type: {bold(instance_type)} in {bold(region)}")

    if hourly_cost is not None:
        print(f"Approximate cost: {green(f'~${hourly_cost:.2f}/hour')} (on-demand pricing)")
    else:
        # Provide a rough estimate when we can't calculate exact cost
        gpu_cost_estimates = {
            1: "~$0.70-0.90/hour",
            2: "~$1.40-1.80/hour",
            4: "~$2.80-3.60/hour",
            8: "~$5.60-7.20/hour",
        }
        estimate = gpu_cost_estimates.get(args.gpus, f"~${0.70 * args.gpus:.2f}-{0.90 * args.gpus:.2f}/hour")
        print(f"Approximate cost: {yellow(estimate)} (estimated for {args.gpus} L4 GPU{'s' if args.gpus > 1 else ''})")

    autostop_hours = 48

    with spinner("Preparing task configuration", style=cyan):
        task = sky.Task.from_yaml(config_path)
        set_task_secrets(task)
        task.set_resources_override({"accelerators": f"{gpu_type}:{args.gpus}"})

        # Set the git ref in the environment variables
        task.update_envs({"METTA_GIT_REF": git_ref})

        time.sleep(1)

    print("\n⏳ This will take a few minutes...")

    try:
        sky_info = sky.api_info()
        if sky_info["status"] == "healthy" and sky_info["user"] is None:
            print(red("✗ You are not authenticated with SkyPilot."))
            print(f"  {green('metta install skypilot')}")
            print("to authenticate before launching a sandbox.")
            return 1

        request_id = sky.launch(
            task,
            cluster_name=cluster_name,
            idle_minutes_to_autostop=autostop_hours * 60,
            retry_until_up=args.retry_until_up,
        )

        # Stream the launch logs (no spinner here - let SkyPilot control the output)
        _result = sky.stream_and_get(request_id)

    except sky.exceptions.ResourcesUnavailableError as e:
        print(f"\n{red('✗ Failed to provision resources')}")
        print(f"\n{yellow('Tip:')} The requested resources are not available in {region}.")
        print("You can try:")
        print(f"  • Run with {green('--retry-until-up')} flag to keep retrying")
        print(f"  • Try a different region by modifying {cyan('sandbox.yaml')}")
        print("  • Use a different GPU type or instance size")
        print(f"\nError details: {str(e)}")
        return 1
    except Exception as e:
        print(f"\n{red('✗ Launch failed:')} {str(e)}")
        return 1

    # Wait for cluster to be fully ready before attempting SSH setup
    print("⏳ Waiting for cluster to be fully ready...")
    max_wait_seconds = args.wait_timeout
    wait_interval = 10  # Check every 10 seconds
    max_wait_attempts = max_wait_seconds // wait_interval

    for wait_attempt in range(max_wait_attempts):
        # Wait between checks, but not on the first attempt
        if wait_attempt > 0:
            time.sleep(wait_interval)

        try:
            with spinner(f"Checking cluster status (attempt {wait_attempt + 1}/{max_wait_attempts})", style=cyan):
                request_id = sky.status()
                cluster_records = sky.get(request_id)

            # Find our cluster in the status
            cluster_status = None
            for cluster in cluster_records:
                if cluster["name"] == cluster_name:
                    cluster_status = cluster["status"].name
                    break

            if cluster_status == "UP":
                print(f"{green('✓')} Cluster is now UP and ready")
                break
            elif cluster_status == "INIT":
                print(f"{yellow('⏳')} Cluster still initializing... (attempt {wait_attempt + 1}/{max_wait_attempts})")
            elif cluster_status is None:
                print(f"{red('✗')} Cluster not found in status output")
                return 1
            else:
                print(f"{red('✗')} Cluster in unexpected state: {cluster_status}")
                return 1

        except Exception as e:
            print(f"{yellow('⚠')} Error checking cluster status: {str(e)}, retrying...")

    else:
        # This else clause runs if the for loop completed without breaking (timeout)
        print(f"\n{red('✗')} Cluster did not reach UP state within {max_wait_seconds} seconds")
        print("Current status might still be INIT. You can:")
        print(f"  • Check status manually: {green(f'sky status {cluster_name}')}")
        print(f"  • Try connecting anyway: {green(f'ssh {cluster_name}')}")
        print(f"  • Re-run launch to retry: {green(f'sky launch -c {cluster_name} --no-setup')}")
        print(f"  • Increase timeout: {green(f'--wait-timeout {max_wait_seconds + 300}')}")
        return 1

    # Don't use spinner during log tailing - it interferes with output
    print("⚙️ Running setup job...")
    try:
        setup_result = sky.tail_logs(cluster_name, job_id=1, follow=True)
        if setup_result != 0:
            print(f"{red('✗')} Setup job failed with exit code {setup_result}")
            print(f"You can check logs with: {green(f'sky logs {cluster_name}')}")
            return 1
    except Exception as e:
        print(f"{red('✗')} Failed to tail setup logs: {str(e)}")
        print(f"You can check logs manually with: {green(f'sky logs {cluster_name}')}")
        return 1

    # Configure SSH access after cluster is fully ready
    with spinner("Configuring SSH access", style=cyan):
        try:
            subprocess.run(["sky", "status", cluster_name], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"\n{yellow('⚠')} SSH setup may not be complete. You can try:")
            print(f"  • Manual SSH: {green(f'ssh {cluster_name}')}")
            print(f"  • Update SSH config: {green(f'sky status {cluster_name}')}")
            print(f"Error: {str(e)}")

    print(f"\n{green('✓')} Sandbox is ready!")
    print("\nConnect to your sandbox:")
    print(f"  {green(f'ssh {cluster_name}')}")
    print(f"\n\n⚠️ The cluster will be automatically stopped after {bold(str(autostop_hours))} hours.")
    print("To disable autostop:")
    print(f"  {green(f'sky autostop --cancel {cluster_name}')}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main() or 0)
