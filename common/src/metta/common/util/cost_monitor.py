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
from typing import Any, Tuple

# Remove the current directory from sys.path to avoid circular import with local colorama.py
sys.path = [p for p in sys.path if p not in ("", ".", os.path.dirname(__file__))]

import requests  # noqa: E402
import sky  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def get_instance_cost(instance_type: str, region: str, zone: str | None = None, use_spot: bool = False) -> float | None:
    """
    Get the hourly cost for a specific instance type.

    Args:
        instance_type: AWS instance type (e.g., 't2.micro')
        region: AWS region (e.g., 'us-west-2')
        zone: Optional AWS availability zone
        use_spot: Whether to calculate spot instance pricing

    Returns:
        Hourly cost as a float, or None if unable to calculate
    """
    try:
        cloud = sky.clouds.AWS()
        instance_hourly_cost = cloud.instance_type_to_hourly_cost(
            instance_type, use_spot=use_spot, region=region, zone=zone
        )
        return instance_hourly_cost
    except Exception as e:
        logger.error(f"Error calculating hourly cost for {instance_type}: {e}")
        return None


def get_running_instance_info() -> Tuple[str, str, str, bool] | None:
    """
    Retrieve instance metadata for the currently running EC2 instance.

    Returns:
        Tuple of (instance_type, region, zone, use_spot) or None if not on EC2
    """
    # Get region and zone from SkyPilot cluster info
    if "SKYPILOT_CLUSTER_INFO" not in os.environ:
        logger.warning("SKYPILOT_CLUSTER_INFO not set. Cannot determine instance info.")
        return None

    try:
        cluster_info = json.loads(os.environ["SKYPILOT_CLUSTER_INFO"])
        region = cluster_info.get("region")
        zone = cluster_info.get("zone")
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing SKYPILOT_CLUSTER_INFO: {e}")
        return None

    if not region:
        logger.error("Region not found in SKYPILOT_CLUSTER_INFO.")
        return None

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
        logger.error(f"Error querying AWS metadata service. This is expected if not on an AWS EC2 instance. Error: {e}")
        return None


def get_cost_info() -> dict[str, Any] | None:
    """
    Retrieves instance and cost information from the cloud environment.

    Returns:
        Dictionary with cost information or None if unable to determine
    """
    instance_info = get_running_instance_info()
    if not instance_info:
        return None

    instance_type, region, zone, use_spot = instance_info

    instance_hourly_cost = get_instance_cost(instance_type=instance_type, region=region, zone=zone, use_spot=use_spot)

    if instance_hourly_cost is None:
        return None

    return {
        "instance_hourly_cost": instance_hourly_cost,
        "instance_type": instance_type,
        "region": region,
        "zone": zone,
        "use_spot": use_spot,
    }


def main():
    """
    Calculates the total hourly cost for the cluster and sets METTA_HOURLY_COST
    environment variable by appending to the shell's environment.
    """
    cost_info = get_cost_info()
    if cost_info:
        instance_hourly_cost = cost_info["instance_hourly_cost"]
        num_nodes_env = os.environ.get("SKYPILOT_NUM_NODES")
        if num_nodes_env is None:
            logger.warning("SKYPILOT_NUM_NODES environment variable not set; cost info will not be provided.")
            return
        num_nodes = int(num_nodes_env)
        total_hourly_cost = instance_hourly_cost * num_nodes

        # Set the environment variable for the current session
        os.environ["METTA_HOURLY_COST"] = str(total_hourly_cost)

        # Also append to bashrc to persist for child processes
        bashrc_path = os.path.expanduser("~/.bashrc")
        with open(bashrc_path, "a") as f:
            f.write(f"\nexport METTA_HOURLY_COST={total_hourly_cost}\n")

        # Log details to stderr for visibility in SkyPilot logs
        logger.info(f"Instance Type: {cost_info['instance_type']}")
        logger.info(f"Spot Instance: {cost_info['use_spot']}")
        logger.info(f"Region: {cost_info['region']}")
        logger.info(
            f"Total Hourly Cost for {num_nodes} node(s): ${total_hourly_cost:.4f} "
            f"(${instance_hourly_cost:.4f}/hr per instance)"
        )
        logger.info(f"Set METTA_HOURLY_COST={total_hourly_cost}")
    else:
        logger.warning("Could not determine hourly cost. METTA_HOURLY_COST will not be set.")


if __name__ == "__main__":
    main()
