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

# Remove the current directory from sys.path to avoid circular import with local colorama.py
sys.path = [p for p in sys.path if p not in ("", ".", os.path.dirname(__file__))]

import requests  # noqa: E402
import sky  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def get_cost_info():
    """
    Retrieves instance and cost information from the cloud environment.
    """
    if "SKYPILOT_CLUSTER_INFO" not in os.environ:
        logger.warning("SKYPILOT_CLUSTER_INFO not set. Cannot determine cost.")
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
            "http://169.254.169.254/latest/api/token", headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
        ).text
        headers = {"X-aws-ec2-metadata-token": token}
        instance_type = requests.get("http://169.254.169.254/latest/meta-data/instance-type", headers=headers).text
        life_cycle = requests.get("http://169.254.169.254/latest/meta-data/instance-life-cycle", headers=headers).text
        use_spot = life_cycle == "spot"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying AWS metadata service. This is expected if not on an AWS EC2 instance. Error: {e}")
        return None

    try:
        # Calculate cost for a single instance
        cloud = sky.clouds.AWS()
        instance_hourly_cost = cloud.instance_type_to_hourly_cost(
            instance_type, use_spot=use_spot, region=region, zone=zone
        )
    except Exception as e:
        logger.error(f"Error calculating hourly cost with sky: {e}")
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
        num_nodes = int(os.environ.get("NUM_NODES", 1))
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
