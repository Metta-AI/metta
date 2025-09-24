#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "skypilot",
#     "requests",
# ]
# ///
"""Calculate hourly cost for running SkyPilot cluster."""

import json
import logging
import os
import sys
from typing import Any

# Remove current directory from sys.path to avoid circular imports
sys.path = [p for p in sys.path if p not in ("", ".", os.path.dirname(__file__))]

import requests  # noqa: E402
import sky  # noqa: E402
import sky.clouds  # noqa: E402

logger = logging.getLogger(__name__)

# AWS Instance Metadata Service (IMDS) endpoint
AWS_IMDS_ENDPOINT = "169.254.169.254"


def get_instance_cost(instance_type: str, region: str, zone: str | None = None, use_spot: bool = False) -> float | None:
    """
    Get the hourly cost for a specific instance type.

    Args:
        instance_type: AWS instance type (e.g., 't2.micro')
        region: AWS region (e.g., 'us-west-2')
        zone: Optional AWS availability zone
        use_spot: Whether to calculate spot instance pricing

    Returns:
        Hourly cost in USD as a float

    """
    try:
        cloud = sky.clouds.AWS()
        return cloud.instance_type_to_hourly_cost(instance_type, use_spot=use_spot, region=region, zone=zone)
    except Exception as e:
        logger.error(f"Error calculating hourly cost for {instance_type}: {e}")
        return None


def get_running_instance_info() -> tuple[str, str, str, bool]:
    """
    Retrieve instance metadata for the currently running EC2 instance.

    Returns:
        Tuple of (instance_type, region, zone, use_spot)
    """
    # Get region and zone from SkyPilot cluster info
    if "SKYPILOT_CLUSTER_INFO" not in os.environ:
        raise RuntimeError("SKYPILOT_CLUSTER_INFO not set. Cannot determine instance info.")

    try:
        cluster_info = json.loads(os.environ["SKYPILOT_CLUSTER_INFO"])
        region = cluster_info.get("region")
        zone = cluster_info.get("zone")
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Error parsing SKYPILOT_CLUSTER_INFO: {e}") from e

    if not region:
        raise RuntimeError("Region not found in SKYPILOT_CLUSTER_INFO.")

    try:
        # Query AWS metadata (IMDSv2 for security)
        token = requests.put(
            f"http://{AWS_IMDS_ENDPOINT}/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=2,
        ).text
        headers = {"X-aws-ec2-metadata-token": token}

        instance_type = requests.get(
            f"http://{AWS_IMDS_ENDPOINT}/latest/meta-data/instance-type", headers=headers, timeout=2
        ).text

        life_cycle = requests.get(
            f"http://{AWS_IMDS_ENDPOINT}/latest/meta-data/instance-life-cycle", headers=headers, timeout=2
        ).text

        use_spot = life_cycle == "spot"

        return instance_type, region, zone, use_spot

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error querying AWS metadata service. Not running on AWS EC2? Error: {e}") from e


def get_cost_info() -> dict[str, Any]:
    """Get cost information for the running cluster."""
    instance_type, region, zone, use_spot = get_running_instance_info()

    num_nodes_env = os.environ.get("SKYPILOT_NUM_NODES")
    if not num_nodes_env:
        raise RuntimeError("SKYPILOT_NUM_NODES environment variable not set")

    num_nodes = int(num_nodes_env)
    instance_hourly_cost = get_instance_cost(instance_type=instance_type, region=region, zone=zone, use_spot=use_spot)

    if instance_hourly_cost is None:
        raise RuntimeError(f"Could not determine hourly cost for {instance_type}")

    return {
        "instance_type": instance_type,
        "region": region,
        "zone": zone,
        "use_spot": use_spot,
        "instance_hourly_cost": instance_hourly_cost,
        "num_nodes": num_nodes,
        "total_hourly_cost": instance_hourly_cost * num_nodes,
    }


def main():
    """
    Calculates the total hourly cost for the cluster and sets METTA_HOURLY_COST environment variable by appending
    to the shell's environment. Also prints METTA_HOURLY_COST to stdout for shell script consumption. Other output
    goes to stderr.
    """
    try:
        cost_info = get_cost_info()
        total_hourly_cost = cost_info["total_hourly_cost"]
        num_nodes = cost_info["num_nodes"]
        instance_hourly_cost = cost_info["instance_hourly_cost"]

        # Set the environment variable for the current session
        os.environ["METTA_HOURLY_COST"] = str(total_hourly_cost)

        # Also append to METTA_ENV_FILE to persist for parent process
        metta_env_file = os.path.expanduser("~/.metta_env_path")
        try:
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(metta_env_file), exist_ok=True)
            with open(metta_env_file, "a") as f:
                f.write(f"\nexport METTA_HOURLY_COST={total_hourly_cost}\n")
        except FileNotFoundError:
            print(f"Warning: Directory for {metta_env_file} could not be created", file=sys.stderr)
        except PermissionError:
            print(f"Warning: No permission to write to {metta_env_file}", file=sys.stderr)
        except IOError as e:
            print(f"Warning: Failed to write to {metta_env_file}: {e}", file=sys.stderr)

        # Log details to stderr for visibility in SkyPilot logs
        logger.info(f"Instance Type: {cost_info['instance_type']}")
        logger.info(f"Spot Instance: {cost_info['use_spot']}")
        logger.info(f"Region: {cost_info['region']}")
        logger.info(
            f"Total Hourly Cost for {num_nodes} node(s): ${total_hourly_cost:.4f} "
            f"(${instance_hourly_cost:.4f}/hr per instance)"
        )
        logger.info(f"Set METTA_HOURLY_COST={total_hourly_cost}")

        # Print the value to stdout for shell script consumption
        print(total_hourly_cost)
        return 0
    except Exception as e:
        logger.error(f"Failed to get cost information: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
