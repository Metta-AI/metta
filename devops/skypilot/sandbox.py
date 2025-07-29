#!/usr/bin/env -S uv run
import argparse
import os
import subprocess
import time

import sky
import sky.exceptions
import yaml

from metta.common.util.cli import spinner
from metta.common.util.cost_monitor import get_instance_cost
from metta.common.util.text_styles import blue, bold, cyan, green, red, yellow


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
    # These are common mappings for L4 GPUs on AWS
    gpu_instance_map = {
        ("L4", 1): "g6.xlarge",
        ("L4", 2): "g6.2xlarge",
        ("L4", 4): "g6.4xlarge",
        ("L4", 8): "g6.8xlarge",
    }

    # Get the instance type based on GPU configuration
    instance_type = gpu_instance_map.get((gpu_type, num_gpus))
    if not instance_type:
        # Fallback to a reasonable default or raise an error
        print(f"Warning: No instance mapping for {num_gpus} {gpu_type} GPU(s), using g6.xlarge as estimate")
        instance_type = "g6.xlarge"
        estimated_multiplier = num_gpus  # Rough estimate
    else:
        estimated_multiplier = 1

    # Calculate cost for on-demand instances, since sandboxes don't use spot.
    with spinner(f"Calculating cost for {instance_type}", style=cyan):
        hourly_cost = get_instance_cost(instance_type=instance_type, region=region, use_spot=False)

    if hourly_cost is not None:
        hourly_cost *= estimated_multiplier

    return instance_type, region, hourly_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-ref", type=str, default=None)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--gpus", type=int, default=1, help="Number of L4 GPUs to use.")
    parser.add_argument("--retry-until-up", action="store_true", help="Keep retrying until cluster is up")
    args = parser.parse_args()

    existing_clusters = get_existing_clusters()

    if existing_clusters and not args.new:
        print(f"You already have {len(existing_clusters)} sandbox(es) running:")
        for cluster in existing_clusters:
            message = ""
            if cluster["status"].name == "INIT":
                message = " (launching)"
            elif cluster["status"].name == "STOPPED":
                message = " (stopped)"
            elif cluster["status"].name == "UP":
                message = " (running)"

            gpu_info = ""
            if "handle" in cluster and cluster["handle"]:
                resources = cluster["handle"].launched_resources
                if resources and resources.accelerators:
                    gpu_info = f" [{resources.accelerators}]"

            print(f"  {yellow(cluster['name'])}{message}{gpu_info}")

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

        return

    cluster_name = get_next_name(existing_clusters)
    print(f"\n🚀 Launching {blue(cluster_name)} with {bold(str(args.gpus))} L4 GPU(s)")

    # Load the sandbox configuration
    config_path = "./devops/skypilot/config/sandbox.yaml"
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

    if hourly_cost is not None:
        print(f"Instance type: {bold(instance_type)} in {bold(region)}")
        print(f"Approximate cost: {green(f'~${hourly_cost:.2f}/hour')} (on-demand pricing)")
    else:
        print("Unable to determine exact cost, but typical L4 on-demand instances cost ~$0.70-$2.00/hour per GPU")

    autostop_hours = 48

    with spinner("Preparing task configuration", style=cyan):
        task = sky.Task.from_yaml(config_path)
        task.set_resources_override({"accelerators": f"{gpu_type}:{args.gpus}"})
        time.sleep(1)

    print("\n⏳ This will take a few minutes...")

    try:
        sky_info = sky.api_info()
        if sky_info["status"] == "healthy" and sky_info["user"] is None:
            print(red("✗ You are not authenticated with SkyPilot."))
            print(f"  {green('metta install skypilot')}")
            print("to authenticate before launching a sandbox.")
            return

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
        return
    except Exception as e:
        print(f"\n{red('✗ Launch failed:')} {str(e)}")
        return

    # Don't use spinner during log tailing - it interferes with output
    print("⚙️ Running setup job...")
    try:
        setup_result = sky.tail_logs(cluster_name, job_id=1, follow=True)
        if setup_result != 0:
            print(f"{red('✗')} Setup job failed with exit code {setup_result}")
            return
    except Exception as e:
        print(f"{red('✗')} Failed to tail setup logs: {str(e)}")
        return

    # Force ssh setup
    with spinner("Configuring SSH access", style=cyan):
        subprocess.run(["sky", "status", cluster_name], check=True, capture_output=True)

    print(f"\n{green('✓')} Sandbox is ready!")
    print("\nConnect to your sandbox:")
    print(f"  {green(f'ssh {cluster_name}')}")
    print(f"\n\n⚠️ The cluster will be automatically stopped after {bold(str(autostop_hours))} hours.")
    print("To disable autostop:")
    print(f"  {green(f'sky autostop --cancel {cluster_name}')}")


if __name__ == "__main__":
    main()
