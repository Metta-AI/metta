#!/usr/bin/env -S uv run
# ruff: noqa: E402
from metta.common.util.log_config import suppress_noisy_logs

suppress_noisy_logs()

import os
import re
import subprocess
import time
from contextlib import contextmanager
from typing import Annotated, Optional

import sky
import sky.exceptions
import typer
import yaml
from sky import skypilot_config
from sky.schemas.api.responses import StatusResponse
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from typer import rich_utils

import gitta as git
from devops.skypilot.utils.cost_monitor import get_instance_cost
from devops.skypilot.utils.task_helpers import set_task_secrets
from metta.common.util.cli import spinner
from metta.common.util.log_config import init_logging
from metta.common.util.text_styles import blue, bold, cyan, green, red, yellow


def get_existing_clusters():
    """Get existing clusters."""
    with spinner("Fetching existing clusters", style=cyan):
        request_id = sky.status()
        cluster_records = sky.get(request_id)
    return cluster_records


def get_user_sandboxes(clusters):
    """Filter clusters to only show user's sandboxes."""
    username = os.environ.get("USER", "unknown")
    return [c for c in clusters if c["name"].startswith(f"{username}-sandbox-")]


def get_next_sandbox_name(cluster_records):
    """Get the next available sandbox name for the current user."""
    names = [record["name"] for record in cluster_records]
    username = os.environ["USER"]
    for i in range(1, 100):
        name = f"{username}-sandbox-{i}"
        if name not in names:
            return name
    raise ValueError("No available sandbox name found")


_CLUSTER_NAME_RE = re.compile(r"^[a-zA-Z]([-_.a-zA-Z0-9]*[a-zA-Z0-9])?$")


def suggest_valid_cluster_name(name: str, username: str) -> str:
    if name and not name[0].isalpha():
        name = f"{username}-{name}"
    name = re.sub(r"[^-_.a-zA-Z0-9]", "-", name)
    name = name.strip("-_.")
    if not name:
        return f"{username}-sandbox"
    if not name[0].isalpha():
        name = f"{username}-{name}"
    return name


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


def format_cluster_info(cluster: StatusResponse):
    """Format cluster information for display."""
    status = cluster.status.name

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


def print_cluster_status(clusters: list[StatusResponse]):
    """Print cluster status in a consistent format."""
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


def print_management_commands(clusters: list[StatusResponse]):
    """Print helpful management commands."""
    if not clusters:
        return

    first_cluster_name = clusters[0].name
    first_stopped_cluster = next((c.name for c in clusters if c.status.name == "STOPPED"), None)

    print("\n📦 Manage sandboxes:")
    print(f"  Launch new:     {green('./devops/skypilot/sandbox.py new')}")
    print(f"  Connect:        {green(f'ssh {first_cluster_name}')}")
    if first_stopped_cluster:
        print(f"  Restart:        {green(f'uv run sky start {first_stopped_cluster}')}")
    print(f"  Stop:           {green(f'uv run sky stop {first_cluster_name}')}")
    print(f"  Delete:         {red(f'uv run sky down {first_cluster_name}')}")


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


def print_cost_info(hourly_cost: float | None, num_gpus: int):
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


@contextmanager
def maybe_set_capacity_reservation(capacity_reservation_id: str | None):
    if capacity_reservation_id is None:
        yield
        return

    new_configs = skypilot_config.to_dict()
    new_configs.set_nested(("aws", "specific_reservations"), [capacity_reservation_id])
    with skypilot_config.replace_skypilot_config(new_configs):
        yield


def check_cluster_status(cluster_name: str) -> Optional[str]:
    """Check the status of a specific cluster.

    Returns:
        The status name (e.g., "UP", "INIT", "STOPPED") or None if not found.
    """
    request_id = sky.status()
    cluster_records = sky.get(request_id)

    for cluster in cluster_records:
        if cluster.name == cluster_name:
            return cluster.status.name

    return None


def wait_for_cluster_ready(cluster_name: str, timeout_seconds: int = 300) -> bool:
    """Wait for cluster to reach UP state using tenacity retry.

    Returns:
        True if cluster reached UP state, False if timeout or error.
    """
    print("⏳ Waiting for cluster to be fully ready...")
    start_time = time.time()

    @retry(
        stop=stop_after_attempt(timeout_seconds // 5 + 1),
        wait=wait_exponential_jitter(initial=1.0, max=5.0),
        reraise=True,
    )
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
        check_and_validate_status()
        print(f"{green('✓')} Cluster is now UP and ready")
        return True
    except Exception as e:
        print(f"\n{red('✗')} {str(e)}")
        return False


def handle_check_mode(clusters: list[StatusResponse]):
    """Handle --check mode to display user's sandboxes."""
    username = os.environ.get("USER", "unknown")
    user_sandboxes = get_user_sandboxes(clusters)

    if not user_sandboxes:
        print(f"{green('✓')} No active sandboxes found for user {bold(username)}")
        print("\nLaunch your first sandbox:")
        print(f"  {green('./devops/skypilot/sandbox.py new')}")
        return

    print(f"{bold(f'Found {len(user_sandboxes)} sandbox(es) for user {username}:')}")

    # Count by status
    status_counts = {"UP": 0, "STOPPED": 0, "INIT": 0}

    for cluster in user_sandboxes:
        color_func, status_msg, gpu_info = format_cluster_info(cluster)
        status = cluster.status.name
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
    return


app = typer.Typer(
    rich_markup_mode="rich",
)

rich_utils.STYLE_HELPTEXT = ""  # don't gray out help text - https://github.com/fastapi/typer/issues/437


@app.callback(invoke_without_command=True)
def check():
    """
    Manage GPU development sandboxes on SkyPilot.

    When run without arguments, displays existing sandboxes and management commands.
    """
    existing_clusters = get_existing_clusters()
    handle_check_mode(existing_clusters)


@app.command()
def new(
    git_ref: Annotated[
        Optional[str], typer.Option(help="Git branch or commit to deploy (default: current branch)")
    ] = None,
    name: Annotated[
        Optional[str],
        typer.Option("--name", help="Sandbox/cluster name (default: next available <user>-sandbox-N)"),
    ] = None,
    config_path_override: Annotated[
        Optional[str], typer.Option("--config", help="SkyPilot task YAML to launch (default: sandbox.yaml)")
    ] = None,
    gpus: Annotated[Optional[int], typer.Option("--gpus", help="Number of GPUs to use")] = None,
    image_id: Annotated[Optional[str], typer.Option("--image-id", help="Override AMI ID (AWS)")] = None,
    capacity_reservation_id: Annotated[
        Optional[str],
        typer.Option(
            "--capacity-reservation-id",
            help="AWS EC2 capacity reservation ID to target (e.g. cr-01fc61d2a5e290e58)",
        ),
    ] = None,
    retry_until_up: Annotated[
        bool, typer.Option("--retry-until-up", help="Keep retrying until cluster is successfully launched")
    ] = False,
    wait_timeout: Annotated[
        int, typer.Option("--wait-timeout", help="Timeout in seconds to wait for cluster to reach UP state")
    ] = 300,
    sweep_controller: Annotated[
        bool,
        typer.Option(
            "--sweep-controller", help="Launch a sweep-controller CPU-only sandbox (instance type from config)"
        ),
    ] = False,
):
    """
    Launch a new sandbox.

    Examples:
        [bold]sandbox.py[/bold]                               # Show existing sandboxes and management commands
        [bold]sandbox.py new --sweep-controller[/bold]        # Launch a CPU-only sandbox (uses config instance_type)
        [bold]sandbox.py new --git-ref feature-branch[/bold]  # Launch with specific git branch
        [bold]sandbox.py new --name my-sandbox[/bold]
        [bold]sandbox.py new --config ./devops/skypilot/config/sandbox_p5.yaml[/bold]
        [bold]sandbox.py new --wait-timeout 600[/bold]        # Wait up to 10 minutes for cluster to be ready

    Common management commands:
        ssh <sandbox-name>                 # Connect to a running sandbox
        uv run sky stop <name>             # Stop a sandbox (keeps data)
        uv run sky start <name>            # Restart a stopped sandbox
        uv run sky down <name>             # Delete a sandbox completely
        uv run sky logs <name>             # Check cluster logs
        uv run sky launch -c <name> --no-setup  # Retry launch for stuck clusters
    """

    # Get git ref - use current branch/commit if not specified
    git_ref = git_ref or get_current_git_ref()

    try:
        existing_clusters = get_existing_clusters()
    except Exception as e:
        print(f"{red('✗')} Failed to fetch existing clusters: {str(e)}")
        raise typer.Exit(1) from None

    # Launch new sandbox
    if name is not None:
        username = os.environ.get("USER", "unknown")
        if _CLUSTER_NAME_RE.fullmatch(name) is None:
            suggested = suggest_valid_cluster_name(name, username)
            print(f"{red('✗')} Cluster name is invalid: {cyan(name)}")
            print(f"  Try: {green(f'--name {suggested}')}")
            raise typer.Exit(1)

        existing_names = {record["name"] for record in existing_clusters}
        if name in existing_names:
            print(f"{red('✗')} Cluster already exists: {cyan(name)}")
            raise typer.Exit(1)
        cluster_name = name
    else:
        cluster_name = get_next_sandbox_name(existing_clusters)

    if config_path_override is not None and not os.path.exists(config_path_override):
        print(f"{red('✗')} Config not found: {cyan(config_path_override)}")
        raise typer.Exit(1)

    if config_path_override is None:
        if sweep_controller:
            config_path = "./devops/skypilot/config/sandbox_cheap.yaml"
        else:
            config_path = "./devops/skypilot/config/sandbox.yaml"
    else:
        config_path = config_path_override

    if sweep_controller:
        if gpus is not None:
            print(f"{red('✗')} Error: --sweep-controller mode is CPU-only and cannot use GPUs.")
            print(f"  Either use --sweep-controller without --gpus, or use regular mode with --gpus {gpus}")
            raise typer.Exit(1)
        print(f"\n🚀 Launching {blue(cluster_name)} in {bold('CPU-ONLY MODE')}")
    else:
        if gpus is None and config_path_override is None:
            gpus = 1

    print(f"🔌 Git ref: {cyan(git_ref)}")

    # Load configuration
    sandbox_config = load_sandbox_config(config_path)

    # Extract cloud configuration
    resources = sandbox_config.get("resources", {})
    cloud = resources.get("cloud", "aws")
    region = resources.get("region", "us-east-1")

    if sweep_controller:
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
        accelerators_str = resources.get("accelerators", "L4:1")
        gpu_type, gpus_in_config_str = accelerators_str.split(":")
        gpus_in_config = int(gpus_in_config_str)
        gpus_to_use = gpus if gpus is not None else gpus_in_config

        print(f"\n🚀 Launching {blue(cluster_name)} with {bold(str(gpus_to_use))} {gpu_type} GPU(s)")

        instance_type_in_config = resources.get("instance_type")
        if instance_type_in_config:
            print(f"Instance type: {bold(instance_type_in_config)} in {bold(region)}")
        else:
            instance_type, region, hourly_cost = get_gpu_instance_info(gpus_to_use, gpu_type, region, cloud)
            if instance_type:
                print(f"Instance type: {bold(instance_type)} in {bold(region)}")
            print_cost_info(hourly_cost, gpus_to_use)

    autostop_hours = 48

    # Prepare task
    with spinner("Preparing task configuration", style=cyan):
        task = sky.Task.from_yaml(config_path)
        set_task_secrets(task)

        if not sweep_controller and gpus is not None:
            task.set_resources_override({"accelerators": f"{gpu_type}:{gpus}"})

        if image_id is not None:
            task.set_resources_override({"image_id": image_id})

        if git_ref:
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
            raise typer.Exit(1) from None

        if capacity_reservation_id is not None and cloud.lower() != "aws":
            print(f"{red('✗')} --capacity-reservation-id is only supported for AWS (cloud={cloud})")
            raise typer.Exit(1) from None

        if capacity_reservation_id is not None:
            print(f"📌 Capacity reservation: {cyan(capacity_reservation_id)}")
        if image_id is not None:
            print(f"🖼️  Image ID override: {cyan(image_id)}")

        with maybe_set_capacity_reservation(capacity_reservation_id):
            request_id = sky.launch(
                task,
                cluster_name=cluster_name,
                idle_minutes_to_autostop=autostop_hours * 60,
                retry_until_up=retry_until_up,
            )

            _result = sky.stream_and_get(request_id)
    except typer.Exit as e:
        raise e from None
    except sky.exceptions.ResourcesUnavailableError as e:
        print(f"\n{red('✗ Failed to provision resources')}")
        print(f"\n{yellow('Tip:')} The requested resources are not available in {region}.")
        print("You can try:")
        print(f"  • Run with {green('--retry-until-up')} flag to keep retrying")
        print(f"  • Try a different region by modifying {cyan('sandbox.yaml')}")
        print("  • Use a different GPU type or instance size")
        print(f"\nError details: {str(e)}")
        raise typer.Exit(1) from None
    except Exception as e:
        print(f"\n{red('✗ Launch failed:')} {str(e)}")
        raise typer.Exit(1) from None

    # Wait for cluster to be ready using retry utility
    if not wait_for_cluster_ready(cluster_name, wait_timeout):
        print("Current status might still be INIT. You can:")
        print(f"  • Check status manually: {green(f'uv run sky status {cluster_name}')}")
        print(f"  • Try connecting anyway: {green(f'ssh {cluster_name}')}")
        print(f"  • Re-run launch to retry: {green(f'uv run sky launch -c {cluster_name} --no-setup')}")
        print(f"  • Increase timeout: {green(f'--wait-timeout {wait_timeout + 300}')}")
        raise typer.Exit(1)

    # Run setup job
    print("⚙️ Running setup job...")
    try:
        setup_result = sky.tail_logs(cluster_name, job_id=1, follow=True)
        if setup_result != 0:
            print(f"{red('✗')} Setup job failed with exit code {setup_result}")
            print(f"You can check logs with: {green(f'uv run sky logs {cluster_name}')}")
            raise typer.Exit(1)
    except Exception as e:
        print(f"{red('✗')} Failed to tail setup logs: {str(e)}")
        print(f"You can check logs manually with: {green(f'uv run sky logs {cluster_name}')}")
        raise typer.Exit(1) from None

    # Configure SSH access
    with spinner("Configuring SSH access", style=cyan):
        try:
            subprocess.run(["sky", "status", cluster_name], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"\n{yellow('⚠')} SSH setup may not be complete. You can try:")
            print(f"  • Manual SSH: {green(f'ssh {cluster_name}')}")
            print(f"  • Update SSH config: {green(f'uv run sky status {cluster_name}')}")
            print(f"Error: {str(e)}")

    # For CPU-only mode, SCP the additional files over
    if sweep_controller:
        print("\n📤 Transferring additional files to sandbox...")
        scp_success = True

        failed_dirs: list[str] = []

        for folder in [".sky", ".aws", ".metta"]:
            with spinner(f"Copying ~/.{folder} folder", style=cyan):
                try:
                    folder_path = os.path.expanduser(f"~/{folder}")
                    if os.path.exists(folder_path):
                        subprocess.run(
                            ["scp", "-rq", folder_path, f"{cluster_name}:~/"],
                            check=True,
                            capture_output=True,
                        )
                        print(f"  {green('✓')} ~/{folder} folder transferred")
                    else:
                        print(f"  {yellow('⚠')} ~/{folder} folder not found locally")
                        print("    AWS credentials will need to be configured via environment variables")
                except subprocess.CalledProcessError as e:
                    print(f"  {red('✗')} Failed to transfer ~/.aws folder: {str(e)}")
                    failed_dirs.append(folder)
                    scp_success = False

        if not scp_success:
            print(f"\n{yellow('⚠')} Some files failed to transfer.")
            print("  You can manually copy them later with:")
            for folder in failed_dirs:
                print(f"    {green(f'scp -r ~/{folder} {cluster_name}:~/')}")

        # Check if SSO is configured
        config_path = os.path.join(os.path.expanduser("~/.aws"), "config")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                if "sso_session" in f.read() or "sso_start_url" in f.read():
                    print(f"    {yellow('Note:')} AWS SSO detected")
                    print("    Run 'aws sso login' if tokens expired")

    # Success!
    print(f"\n{green('✓')} Sandbox is ready!")
    print("\nConnect to your sandbox:")
    print(f"  {green(f'ssh {cluster_name}')}")
    print(f"\n\n⚠️ The cluster will be automatically stopped after {bold(str(autostop_hours))} hours.")
    print("To disable autostop:")
    print(f"  {green(f'uv run sky autostop --cancel {cluster_name}')}")


if __name__ == "__main__":
    init_logging()
    app()
