#!/usr/bin/env -S uv run
import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import glob
import re

# Suppress Pydantic warnings from SkyPilot dependencies before importing sky
# SkyPilot v0.10.3.post2 with Pydantic 2.12.3 generates UnsupportedFieldAttributeWarning
# for 'repr' and 'frozen' attributes used in Field() definitions that have no effect.
# This is a known issue in Pydantic 2.12+ affecting multiple projects (wandb, pytorch, etc.)
# See: https://github.com/pydantic/pydantic/issues/10497
# These warnings are harmless and come from upstream dependencies, not our code.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import sky  # noqa: E402
import sky.exceptions  # noqa: E402
import yaml  # noqa: E402

import gitta as git  # noqa: E402
from devops.skypilot.utils.cost_monitor import get_instance_cost  # noqa: E402
from devops.skypilot.utils.job_helpers import set_task_secrets  # noqa: E402
from metta.common.util.cli import spinner  # noqa: E402
from metta.common.util.retry import retry_function  # noqa: E402
from metta.common.util.text_styles import blue, bold, cyan, green, red, yellow  # noqa: E402


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


def _build_incluster_persist_script(cluster_name: str) -> str:
    """Return a one-liner to persist cluster name and node count on head (rank 0).

    If a Docker container is running, prefer persisting inside the container; otherwise,
    fall back to the host path. This preserves behavior across both docker: runtime and
    AMI+docker-manual setups.
    """
    quoted_name = shlex.quote(cluster_name)
    return (
        "set -e; "
        'if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then exit 0; fi; '
        # Try to find our named container first, then any running container
        'CID=""; '
        'if command -v docker >/dev/null 2>&1; then '
        '  if docker ps -a --format "{{.Names}}" | grep -qx "metta"; then CID=metta; '
        '  else CID=$(docker ps -q | head -n1); fi; '
        'fi; '
        'if [ -n "$CID" ]; then '
        "  docker exec -i \"$CID\" bash -lc "
        + shlex.quote(
            "set -e; "
            "mkdir -p /workspace/metta/.cluster; "
            f"printf %s\\n {quoted_name} > /workspace/metta/.cluster/name; "
            "printf %s\\n \"${SKYPILOT_NUM_NODES:-1}\" > /workspace/metta/.cluster/num_nodes"
        )
        + "; "
        'else '
        "  mkdir -p /workspace/metta/.cluster; "
        f"  printf %s\\n {quoted_name} > /workspace/metta/.cluster/name; "
        '  printf %s\\n "${SKYPILOT_NUM_NODES:-1}" > /workspace/metta/.cluster/num_nodes; '
        'fi'
    )


def get_existing_clusters():
    """Get existing clusters."""
    with spinner("Fetching existing clusters", style=cyan):
        request_id = sky.status()
        cluster_records = sky.get(request_id)
    return cluster_records


def get_user_sandboxes(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter clusters to only show user's sandboxes."""
    username = os.environ.get("USER", "unknown")
    return [c for c in clusters if c["name"].startswith(f"{username}-sandbox-")]


def get_next_sandbox_name(cluster_records: List[Dict[str, Any]]) -> str:
    """Get the next available sandbox name for the current user."""
    names = [record["name"] for record in cluster_records]
    username = os.environ["USER"]
    for i in range(1, 100):
        name = f"{username}-sandbox-{i}"
        if name not in names:
            return name
    raise ValueError("No available sandbox name found")


def load_sandbox_config(config_path: str) -> Dict[str, Any]:
    """Load the sandbox YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_current_git_ref():
    """Get the current git branch or commit hash."""
    try:
        return git.get_current_branch()
    except (git.GitError, ValueError):
        return "main"  # Fallback to main


def format_cluster_info(cluster: Dict[str, Any]):
    """Format cluster information for display."""
    status = cluster["status"].name

    # Status formatting
    status_map = {
        "UP": (green, "running"),
        "STOPPED": (red, "stopped"),
        "INIT": (cyan, "launching"),
    }
    color_func, status_msg = status_map.get(status, (yellow, status.lower()))

    # GPU info
    gpu_info = ""
    if "handle" in cluster and cluster["handle"]:
        resources = cluster["handle"].launched_resources
        if resources and resources.accelerators:
            gpu_info = f" [{resources.accelerators}]"

    return color_func, status_msg, gpu_info


def print_cluster_status(clusters: List[Dict[str, Any]], title: Optional[str] = None) -> None:
    """Print cluster status in a consistent format."""
    if title:
        print(title)

    for cluster in clusters:
        color_func, status_msg, gpu_info = format_cluster_info(cluster)
        print(f"  {color_func(cluster['name'])} ({status_msg}){gpu_info}")

        # Additional guidance for INIT state clusters
        if cluster["status"].name == "INIT":
            cluster_name = cluster["name"]
            print(
                f"    💡 If stuck in INIT for >10min, try: {green(f'uv run sky launch -c {cluster_name} --no-setup')}"
            )
            print(f"    📊 Check logs: {green(f'uv run sky logs {cluster_name}')}")


def print_management_commands(clusters: List[Dict[str, Any]]) -> None:
    """Print helpful management commands."""
    if not clusters:
        return

    first_cluster_name = clusters[0]["name"]
    first_stopped_cluster = next((c["name"] for c in clusters if c["status"].name == "STOPPED"), None)

    print("\n📦 Manage sandboxes:")
    print(f"  Launch new:     {green('./devops/skypilot/sandbox.py --new')}")
    print(f"  Connect:        {green(f'ssh {first_cluster_name}')}")
    if first_stopped_cluster:
        print(f"  Restart:        {green(f'uv run sky start {first_stopped_cluster}')}")
    print(f"  Stop:           {green(f'uv run sky stop {first_cluster_name}')}")
    print(f"  Delete:         {red(f'uv run sky down {first_cluster_name}')}")


def get_gpu_instance_info(
    num_gpus: int,
    gpu_type: str = "L4",
    region: str = "us-east-1",
    cloud: str = "aws",
) -> Tuple[Optional[str], str, Optional[float]]:
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
    gpu_type_norm = gpu_type.upper()

    gpu_instance_map: Dict[Tuple[str, int], str] = {
        ("L4", 1): "g6.xlarge",
        ("L4", 2): "g6.2xlarge",
        ("L4", 4): "g6.4xlarge",
        ("L4", 8): "g6.8xlarge",
    }

    # Get the instance type based on GPU configuration
    instance_type = gpu_instance_map.get((gpu_type_norm, num_gpus))
    if not instance_type:
        print(f"Warning: No instance mapping for {num_gpus} {gpu_type_norm} GPU(s), using g6.xlarge as estimate")
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


def print_cost_info(hourly_cost: Optional[float], num_gpus: int) -> None:
    """Print cost information in a consistent format."""
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
        estimate = gpu_cost_estimates.get(num_gpus, f"~${0.70 * num_gpus:.2f}-{0.90 * num_gpus:.2f}/hour")
        print(f"Approximate cost: {yellow(estimate)} (estimated for {num_gpus} L4 GPU{'s' if num_gpus > 1 else ''})")


def check_cluster_status(cluster_name: str) -> Optional[str]:
    """Check the status of a specific cluster.

    Returns:
        The status name (e.g., "UP", "INIT", "STOPPED") or None if not found.
    """
    request_id = sky.status()
    cluster_records = sky.get(request_id)

    for cluster in cluster_records:
        if cluster["name"] == cluster_name:
            return cluster["status"].name

    return None


def wait_for_cluster_ready(cluster_name: str, timeout_seconds: int = 300) -> bool:
    """Wait for cluster to reach UP state using retry utility.

    Returns:
        True if cluster reached UP state, False if timeout or error.
    """
    print("⏳ Waiting for cluster to be fully ready...")
    start_time = time.time()

    def check_and_validate_status():
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 0:
            raise TimeoutError(f"Cluster did not reach UP state within {timeout_seconds} seconds")

        with spinner(f"Checking cluster status (remaining: {int(remaining)}s)", style=cyan):
            status = check_cluster_status(cluster_name)

        if status == "UP":
            return True
        elif status == "INIT":
            print(f"{yellow('⏳')} Cluster still initializing...")
            raise Exception("Cluster still in INIT state")  # Will trigger retry
        elif status is None:
            raise Exception("Cluster not found in status output")
        else:
            raise Exception(f"Cluster in unexpected state: {status}")

    try:
        retry_function(
            check_and_validate_status,
            max_retries=timeout_seconds // 5,
            max_delay=5.0,
            exceptions=(Exception,),
        )
        print(f"{green('✓')} Cluster is now UP and ready")
        return True
    except Exception as e:
        print(f"\n{red('✗')} {str(e)}")
        return False


def _find_sky_identity_for_cluster(cluster_name: str) -> Optional[str]:
    """Return the IdentityFile path used by SkyPilot for the given cluster.

    Searches files included via '~/.sky/generated/ssh/*' to find a block like:

      Host <cluster_name>
        IdentityFile /path/to/key

    Returns the IdentityFile path if found, else None.
    """
    try:
        ssh_include_dir = Path.home() / ".sky" / "generated" / "ssh"
        if not ssh_include_dir.exists():
            return None

        for path_str in glob.glob(str(ssh_include_dir / "*")):
            p = Path(path_str)
            try:
                content = p.read_text()
            except Exception:
                continue

            # Find the Host <cluster_name> block
            # Simple regex to capture lines between 'Host <name>' and next 'Host ' or EOF
            host_re = re.compile(rf"^Host\s+{re.escape(cluster_name)}\n(?P<body>(?:^(?!Host\s).*$\n?)*)", re.MULTILINE)
            m = host_re.search(content)
            if not m:
                continue
            body = m.group("body")
            for line in body.splitlines():
                if line.strip().lower().startswith("identityfile"):
                    # Return the remainder of the line after the keyword
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        return os.path.expanduser(parts[1])
        return None
    except Exception:
        return None


def _write_container_ssh_config(cluster_name: str, port: int = 2222, user: str = "root") -> Optional[str]:
    """Create/update a local SSH config entry to access the container via ProxyJump.

    Writes to '~/.sky/generated/ssh/<cluster_name>-container.conf' so it is auto-included
    by the main SSH config (which includes ~/.sky/generated/ssh/*).

    Returns the created host alias (e.g., '<cluster_name>-ctr') on success.
    """
    try:
        alias = f"{cluster_name}-ctr"
        ssh_dir = Path.home() / ".sky" / "generated" / "ssh"
        ssh_dir.mkdir(parents=True, exist_ok=True)

        identity = _find_sky_identity_for_cluster(cluster_name)

        lines: List[str] = [
            f"Host {alias}",
            "  HostName 127.0.0.1",
            f"  Port {port}",
            f"  User {user}",
            f"  ProxyJump {cluster_name}",
            "  StrictHostKeyChecking no",
            "  UserKnownHostsFile /dev/null",
        ]
        if identity:
            lines.append(f"  IdentityFile {identity}")
            lines.append("  IdentitiesOnly yes")

        lines.append("")
        content = "\n".join(lines)
        out_path = ssh_dir / f"{cluster_name}-container.conf"
        out_path.write_text(content)
        return alias
    except Exception:
        return None


def handle_check_mode(clusters):
    """Handle --check mode to display user's sandboxes."""
    username = os.environ.get("USER", "unknown")
    user_sandboxes = get_user_sandboxes(clusters)

    if not user_sandboxes:
        print(f"{green('✓')} No active sandboxes found for user {bold(username)}")
        print("\nLaunch your first sandbox:")
        print(f"  {green('./devops/skypilot/sandbox.py --new')}")
        return 0

    print(f"{bold(f'Found {len(user_sandboxes)} sandbox(es) for user {username}:')}")

    # Count by status
    status_counts = {"UP": 0, "STOPPED": 0, "INIT": 0}

    for cluster in user_sandboxes:
        color_func, status_msg, gpu_info = format_cluster_info(cluster)
        status = cluster["status"].name
        if status in status_counts:
            status_counts[status] += 1

        print(f"  • {color_func(cluster['name'])} ({status_msg}){gpu_info}")

    # Summary
    print(f"\n{bold('Summary:')}")
    if status_counts["UP"] > 0:
        print(f"  {green(str(status_counts['UP']) + ' running')}")
    if status_counts["STOPPED"] > 0:
        print(f"  {red(str(status_counts['STOPPED']) + ' stopped')}")
    if status_counts["INIT"] > 0:
        print(f"  {cyan(str(status_counts['INIT']) + ' launching')}")

    print_management_commands(user_sandboxes)
    return 0


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
  %(prog)s --new --gpus 4  # Launch a new sandbox with 4 GPUs on 1 node
  %(prog)s --new --nodes 4 --gpus 8  # Launch multi-node sandbox (4 nodes, 8 GPUs/node)
  %(prog)s --new --sweep-controller  # Launch a CPU-only sandbox (uses config instance_type)
  %(prog)s --new --git-ref feature-branch  # Launch with specific git branch
  %(prog)s --new --wait-timeout 600  # Wait up to 10 minutes for cluster to be ready

Common management commands:
  ssh <sandbox-name>                 # Connect to a running sandbox
  uv run sky stop <name>             # Stop a sandbox (keeps data)
  uv run sky start <name>            # Restart a stopped sandbox
  uv run sky down <name>             # Delete a sandbox completely
  uv run sky logs <name>             # Check cluster logs
  uv run sky launch -c <name> --no-setup  # Retry launch for stuck clusters
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--git-ref", type=str, default=None, help="Git branch or commit to deploy (default: current branch)"
    )
    parser.add_argument("--new", action="store_true", help="Launch a new sandbox cluster")
    parser.add_argument("--name", type=str, default=None, help="Custom sandbox cluster name (e.g., my-sandbox)")
    parser.add_argument("--check", action="store_true", help="Check for existing sandboxes and exit")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per node (default: 1)")
    parser.add_argument(
        "--gpu-type",
        type=str,
        choices=["L4", "A10G", "A100", "H100", "A100-80GB"],
        help="GPU type to use (overrides config accelerators)",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        help="Explicit cloud instance type (e.g., p5.4xlarge). Overrides automatic mapping.",
    )
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1)")
    parser.add_argument(
        "--retry-until-up", action="store_true", help="Keep retrying until cluster is successfully launched"
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=300,
        help="Timeout in seconds to wait for cluster to reach UP state (default: 300)",
    )
    parser.add_argument(
        "--sweep-controller",
        action="store_true",
        help="Launch a sweep-controller CPU-only sandbox (instance type from config)",
    )

    args = parser.parse_args()

    # Validate conflicting arguments
    if args.sweep_controller and (args.gpus > 1 or args.nodes > 1):
        print(f"{red('✗')} Error: --sweep-controller mode is CPU-only and cannot use GPUs.")
        print(
            f"  Either use --sweep-controller without --gpus/--nodes, "
            f"or use regular mode with --nodes {args.nodes} and --gpus {args.gpus}"
        )
        return 1

    # Get git ref - use current branch/commit if not specified
    git_ref = args.git_ref or get_current_git_ref()

    try:
        existing_clusters = get_existing_clusters()
    except Exception as e:
        print(f"{red('✗')} Failed to fetch existing clusters: {str(e)}")
        return 1

    # Handle --check mode
    if args.check:
        return handle_check_mode(existing_clusters)

    # Show existing clusters if not launching new
    if existing_clusters and not args.new:
        print(f"You already have {len(existing_clusters)} sandbox(es) running:")
        print_cluster_status(existing_clusters)
        print_management_commands(existing_clusters)
        return 0

    # Launch new sandbox
    if args.name:
        cluster_name = args.name
        if any(c["name"] == cluster_name for c in existing_clusters):
            print(red(f"✗ A sandbox named '{cluster_name}' already exists. Choose a different name or stop/delete it."))
            return 1
    else:
        cluster_name = get_next_sandbox_name(existing_clusters)

    # Determine configuration based on --sweep-controller flag
    if args.sweep_controller:
        print(f"\n🚀 Launching {blue(cluster_name)} in {bold('CPU-ONLY MODE')}")
        config_path = "./devops/skypilot/config/sandbox_cheap.yaml"
    else:
        nodes_msg = f"{args.nodes} node{'s' if args.nodes != 1 else ''}"
        # GPU type placeholder; will finalize after reading config and args
        print(f"\n🚀 Launching {blue(cluster_name)} with {bold(nodes_msg)} × GPUs per node: {bold(str(args.gpus))}")
        config_path = "./devops/skypilot/config/sandbox.yaml"

    print(f"🔌 Git ref: {cyan(git_ref)}")

    # Load configuration
    config = load_sandbox_config(config_path)

    # Extract cloud configuration
    resources = config.get("resources", {})
    cloud = resources.get("cloud", "aws")
    region = resources.get("region", "us-east-1")

    if args.sweep_controller:
        # For CPU-only mode, read the instance type from config
        instance_type = resources.get("instance_type", "m6i.2xlarge")
        print(f"Instance type: {bold(instance_type)} in {bold(region)}")

        # Try to calculate on-demand cost dynamically
        hourly_cost = None
        try:
            with spinner(f"Calculating cost for {instance_type}", style=cyan):
                hourly_cost = get_instance_cost(instance_type=instance_type, region=region, use_spot=False)
        except Exception:
            hourly_cost = None

        if hourly_cost is not None:
            print(f"Approximate cost: {green(f'~${hourly_cost:.3f}/hour')} (on-demand pricing)")
        else:
            # Fallback hint when cost API is unavailable
            if instance_type == "m6i.2xlarge":
                print(f"Approximate cost: {green('~$0.384/hour')} (on-demand pricing, us-east-1)")
            else:
                print("Approximate cost: (unavailable) – check AWS pricing for your region.")
    else:
        # Determine GPU type: CLI overrides config
        accelerators_str = resources.get("accelerators", "L4:1")
        default_gpu_type = accelerators_str.split(":")[0]
        gpu_type = (args.gpu_type or default_gpu_type).upper()

        print(f"GPU configuration: {bold(str(args.gpus))} {bold(gpu_type)} GPU{'s' if args.gpus != 1 else ''} per node")

        # For A100/H100 families, switch to AMI-host + manual docker recipe
        if gpu_type in ["A100", "H100", "A100-80GB"]:
            config_path = "./devops/skypilot/config/sandbox_a100.yaml"

        # Get instance type and calculate per-node cost
        instance_type, region, hourly_cost = get_gpu_instance_info(
            args.gpus,
            gpu_type,
            region,
            cloud,
        )

        if instance_type:
            print(f"Instance type (per node): {bold(instance_type)} in {bold(region)}")

        # Cost output (per node and total)
        if hourly_cost is not None:
            total_cost = hourly_cost * max(1, int(args.nodes))
            print(
                f"Approximate cost: {green(f'~${hourly_cost:.2f}/hour per node')} | "
                f"{green(f'~${total_cost:.2f}/hour total for {args.nodes} node(s)')}"
            )
        else:
            # Fallback estimate when cost API unavailable
            gpu_cost_estimates = {
                1: "~$0.70-0.90/hour",
                2: "~$1.40-1.80/hour",
                4: "~$2.80-3.60/hour",
                8: "~$5.60-7.20/hour",
            }
            per_node_est = gpu_cost_estimates.get(args.gpus, f"~${0.70 * args.gpus:.2f}-{0.90 * args.gpus:.2f}/hour")
            print(f"Approximate cost (per node): {yellow(per_node_est)}")
            print(f"Approximate total (very rough): {yellow(per_node_est)} × {args.nodes} nodes")

    autostop_hours = 48

    # Prepare task
    with spinner("Preparing task configuration", style=cyan):
        task = sky.Task.from_yaml(config_path)
        set_task_secrets(task)

        if not args.sweep_controller:
            # Only override GPU resources for non-cheap mode
            # Use computed gpu_type (from CLI or config)
            accelerators_override = f"{gpu_type}:{args.gpus}"
            overrides: Dict[str, Any] = {"accelerators": accelerators_override}
            if args.instance_type:
                overrides["instance_type"] = args.instance_type

            # Use custom AMI for A100 and H100 instances
            if gpu_type in ["A100", "H100", "A100-80GB"]:
                overrides["image_id"] = "ami-06161ef177f46b7d3"
                print(f"Using custom AMI: {cyan('ami-06161ef177f46b7d3')} for {gpu_type}")

            task.set_resources_override(overrides)

            # Multi-node support
            if args.nodes and args.nodes > 1:
                task.num_nodes = int(args.nodes)

        task.update_envs({"METTA_GIT_REF": git_ref})
        time.sleep(1)

    print("\n⏳ This will take a few minutes...")

    # Check authentication
    try:
        sky_info = sky.api_info()
        if sky_info["status"] == "healthy" and sky_info["user"] is None:
            print(red("✗ You are not authenticated with SkyPilot."))
            print(f"  {green('metta install skypilot')}")
            print("to authenticate before launching a sandbox.")
            return 1

        # Launch cluster
        request_id = sky.launch(
            task,
            cluster_name=cluster_name,
            idle_minutes_to_autostop=autostop_hours * 60,
            retry_until_up=args.retry_until_up,
        )

        # Stream the launch logs
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

    # Wait for cluster to be ready using retry utility
    if not wait_for_cluster_ready(cluster_name, args.wait_timeout):
        print("Current status might still be INIT. You can:")
        print(f"  • Check status manually: {green(f'uv run sky status {cluster_name}')}")
        print(f"  • Try connecting anyway: {green(f'ssh {cluster_name}')}")
        print(f"  • Re-run launch to retry: {green(f'uv run sky launch -c {cluster_name} --no-setup')}")
        print(f"  • Increase timeout: {green(f'--wait-timeout {args.wait_timeout + 300}')}")
        return 1

    # Run setup job
    print("⚙️ Running setup job...")
    try:
        setup_result = sky.tail_logs(cluster_name, job_id=1, follow=True)
        if setup_result != 0:
            print(f"{red('✗')} Setup job failed with exit code {setup_result}")
            print(f"You can check logs with: {green(f'uv run sky logs {cluster_name}')}")
            return 1
    except Exception as e:
        print(f"{red('✗')} Failed to tail setup logs: {str(e)}")
        print(f"You can check logs manually with: {green(f'uv run sky logs {cluster_name}')}")
        return 1

    # Configure SSH access
    with spinner("Configuring SSH access", style=cyan):
        try:
            subprocess.run(["sky", "status", cluster_name], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"\n{yellow('⚠')} SSH setup may not be complete. You can try:")
            print(f"  • Manual SSH: {green(f'ssh {cluster_name}')}")
            print(f"  • Update SSH config: {green(f'uv run sky status {cluster_name}')}")
            print(f"Error: {str(e)}")

    # Always persist cluster metadata on master for in-SSH launches
    with spinner("Persisting cluster metadata on master", style=cyan):
        try:
            persist_script = _build_incluster_persist_script(cluster_name)
            subprocess.run(
                [
                    "sky",
                    "exec",
                    cluster_name,
                    "--num-nodes",
                    str(max(1, int(args.nodes or 1))),
                    "--",
                    persist_script,
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"\n{yellow('⚠')} Failed to persist cluster metadata on master: {str(e)}")
            print("   You can still run jobs, but auto-detection may not work in SSH sessions.")

    # Create a local SSH alias to connect directly to the container on the head node
    with spinner("Creating SSH alias for container", style=cyan):
        alias = _write_container_ssh_config(cluster_name, port=2222, user="root")
        if alias:
            print(f"  {green('✓')} Added SSH alias: {bold(alias)}")
            print(f"     Use: {green(f'ssh {alias}')} (proxies via {cluster_name})")
        else:
            print(f"  {yellow('⚠')} Failed to write SSH alias for container. You can still:")
            print(f"     {green(f'ssh -J {cluster_name} -p 2222 root@127.0.0.1')}")

    # For multi-GPU sandboxes (multi-node OR >1 GPU per node) and not sweep-controller, copy helpful credentials
    do_bootstrap_incluster = (not args.sweep_controller) and (
        (args.nodes and int(args.nodes) > 1) or (args.gpus and int(args.gpus) > 1)
    )
    if do_bootstrap_incluster:
        # Transfer helpful credentials and persist cluster metadata to enable in-cluster launching.
        print("\n📤 Transferring credentials for in-cluster SkyPilot usage...")
        scp_success = True

        # Transfer .sky folder (SkyPilot client state/keys)
        with spinner("Copying ~/.sky folder", style=cyan):
            try:
                sky_path = os.path.expanduser("~/.sky")
                if os.path.exists(sky_path):
                    subprocess.run(
                        ["scp", "-rq", sky_path, f"{cluster_name}:~/"],
                        check=True,
                        capture_output=True,
                    )
                    print(f"  {green('✓')} ~/.sky folder transferred")
                else:
                    print(f"  {yellow('⚠')} ~/.sky folder not found locally")
                    scp_success = False
            except subprocess.CalledProcessError as e:
                print(f"  {red('✗')} Failed to transfer ~/.sky folder: {str(e)}")
                scp_success = False

        # Transfer .aws folder (for AWS CLI configuration and SSO)
        with spinner("Copying ~/.aws folder", style=cyan):
            try:
                aws_path = os.path.expanduser("~/.aws")
                if os.path.exists(aws_path):
                    subprocess.run(
                        ["scp", "-rq", aws_path, f"{cluster_name}:~/"],
                        check=True,
                        capture_output=True,
                    )
                    print(f"  {green('✓')} ~/.aws folder transferred")
                    # Check if SSO is configured
                    config_path = os.path.join(aws_path, "config")
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                            if "sso_session" in f.read() or "sso_start_url" in f.read():
                                print(f"    {yellow('Note:')} AWS SSO detected")
                                print("    Run 'aws sso login' if tokens expired")
                else:
                    print(f"  {yellow('⚠')} ~/.aws folder not found locally")
                    print("    AWS credentials will need to be configured via environment variables")
            except subprocess.CalledProcessError as e:
                print(f"  {red('✗')} Failed to transfer ~/.aws folder: {str(e)}")
                scp_success = False

        # Transfer observatory tokens
        with spinner("Copying ~/.metta/config.yaml", style=cyan):
            try:
                obs_path = os.path.expanduser("~/.metta/config.yaml")
                if os.path.exists(obs_path):
                    subprocess.run(
                        ["scp", "-q", obs_path, f"{cluster_name}:~/.metta/config.yaml"],
                        check=True,
                        capture_output=True,
                    )
                    print(f"  {green('✓')} Observatory tokens transferred")
                else:
                    print(f"  {yellow('⚠')} Observatory tokens not found locally")
            except subprocess.CalledProcessError as e:
                print(f"  {red('✗')} Failed to transfer observatory tokens: {str(e)}")
                scp_success = False

        if not scp_success:
            print(f"\n{yellow('⚠')} Some files failed to transfer.")
            print("  You can manually copy them later with:")
            print(f"    {green(f'scp -r ~/.sky {cluster_name}:~/')}")
            print(f"    {green(f'scp ~/.metta/config.yaml {cluster_name}:~/.metta/')}")

    # Success!
    print(f"\n{green('✓')} Sandbox is ready!")
    print("\nConnect to your sandbox:")
    print(f"  {green(f'ssh {cluster_name}')}")
    # If we created the container SSH alias earlier, surface it here as well
    container_alias = f"{cluster_name}-ctr"
    container_conf = Path.home() / ".sky" / "generated" / "ssh" / f"{cluster_name}-container.conf"
    if container_conf.exists():
        print(f"  {green(f'ssh {container_alias}')}  # connect directly to the container")
    print(f"\n\n⚠️ The cluster will be automatically stopped after {bold(str(autostop_hours))} hours.")
    print("To disable autostop:")
    print(f"  {green(f'uv run sky autostop --cancel {cluster_name}')}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
