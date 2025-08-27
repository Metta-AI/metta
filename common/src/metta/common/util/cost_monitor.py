#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "skypilot",
#     "requests",
# ]
# ///
import json
import logging
import os
import sys
from typing import Any

# Remove the current directory from sys.path to avoid circular import with local colorama.py
sys.path = [p for p in sys.path if p not in ("", ".", os.path.dirname(__file__))]

import requests  # noqa: E402
import sky  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def get_instance_cost(instance_type: str, region: str, zone: str | None = None, use_spot: bool = False) -> float:
    """
    Get the hourly cost for a specific instance type.

    Args:
        instance_type: AWS instance type (e.g., 't2.micro')
        region: AWS region (e.g., 'us-west-2')
        zone: Optional AWS availability zone
        use_spot: Whether to calculate spot instance pricing

    Returns:
        Hourly cost as a float

    Raises:
        RuntimeError: If unable to calculate cost
    """
    try:
        cloud = sky.clouds.AWS()
        instance_hourly_cost = cloud.instance_type_to_hourly_cost(
            instance_type, use_spot=use_spot, region=region, zone=zone
        )
        return instance_hourly_cost
    except Exception as e:
        raise RuntimeError(f"Failed to calculate hourly cost for {instance_type}: {e}") from e


def get_running_instance_info() -> tuple[str, str, str, bool]:
    """
    Retrieve instance metadata for the currently running EC2 instance.

    Returns:
        Tuple of (instance_type, region, zone, use_spot)

    Raises:
        RuntimeError: If unable to determine instance info
    """
    # Get region and zone from SkyPilot cluster info
    if "SKYPILOT_CLUSTER_INFO" not in os.environ:
        raise RuntimeError("SKYPILOT_CLUSTER_INFO not set. Cannot determine instance info.")

    try:
        cluster_info = json.loads(os.environ["SKYPILOT_CLUSTER_INFO"])
        region = cluster_info.get("region")
        zone = cluster_info.get("zone")
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Error parsing SKYPILOT_CLUSTER_INFO: {e}")

    if not region:
        raise RuntimeError("Region not found in SKYPILOT_CLUSTER_INFO.")

    try:
        # Query AWS metadata (IMDSv2 for security)
        token = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=2,
        ).text
        headers = {"X-aws-ec2-metadata-token": token}

        instance_type = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type", headers=headers, timeout=2
        ).text

        life_cycle = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-life-cycle", headers=headers, timeout=2
        ).text

        use_spot = life_cycle == "spot"

        return instance_type, region, zone, use_spot

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error querying AWS metadata service. Not running on AWS EC2? Error: {e}")


def get_cost_info() -> dict[str, Any]:
    """
    Retrieves instance and cost information from the cloud environment.

    Returns:
        Dictionary with cost information

    Raises:
        RuntimeError: If unable to determine cost info
    """
    instance_type, region, zone, use_spot = get_running_instance_info()
    instance_hourly_cost = get_instance_cost(instance_type=instance_type, region=region, zone=zone, use_spot=use_spot)

    return {
        "instance_hourly_cost": instance_hourly_cost,
        "instance_type": instance_type,
        "region": region,
        "zone": zone,
        "use_spot": use_spot,
    }


def main():
    """
    Calculates the total hourly cost for the cluster and prints it to stdout.
    Additional info goes to stderr for debugging.
    """
    try:
        cost_info = get_cost_info()

        num_nodes_env = os.environ.get("SKYPILOT_NUM_NODES")
        if num_nodes_env is None:
            raise RuntimeError("SKYPILOT_NUM_NODES environment variable not set")

        num_nodes = int(num_nodes_env)
        instance_hourly_cost = cost_info["instance_hourly_cost"]
        total_hourly_cost = instance_hourly_cost * num_nodes

        # Log details to stderr for visibility in SkyPilot logs
        logger.info(f"Instance Type: {cost_info['instance_type']}")
        logger.info(f"Spot Instance: {cost_info['use_spot']}")
        logger.info(f"Region: {cost_info['region']}")
        logger.info(
            f"Total Hourly Cost for {num_nodes} node(s): ${total_hourly_cost:.4f} "
            f"(${instance_hourly_cost:.4f}/hr per instance)"
        )

        # Print the value to stdout for shell script consumption
        print(total_hourly_cost)
        return 0

    except Exception as e:
        logger.error(f"Failed to get cost info: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
