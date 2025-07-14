"""
AWS pricing utilities for performance threshold tracking.
"""

import json
import logging
import os
import subprocess
from typing import Dict, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class AWSPricingClient:
    """Client for querying AWS pricing information."""

    def __init__(self, region: str = "us-east-1", profile: str = "softmax"):
        """Initialize AWS pricing client.

        Args:
            region: AWS region to query pricing for
            profile: AWS profile to use for credentials
        """
        self.region = region
        self.profile = profile
        self.pricing_client = None
        self._init_client()

    def _init_client(self):
        """Initialize the AWS pricing client."""
        try:
            session = boto3.Session(profile_name=self.profile, region_name=self.region)
            self.pricing_client = session.client("pricing")
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"Failed to initialize AWS pricing client: {e}")
            logger.warning("Will use fallback pricing data")
            self.pricing_client = None

    def get_instance_pricing(self, instance_type: str, use_spot: bool = False) -> Optional[float]:
        """Get pricing for a specific instance type.

        Args:
            instance_type: AWS instance type (e.g., 'g5.4xlarge')
            use_spot: Whether to get spot pricing (True) or on-demand pricing (False)

        Returns:
            Price per hour in USD, or None if not available
        """
        if not self.pricing_client:
            return self._get_fallback_pricing(instance_type, use_spot)

        try:
            # Query AWS Pricing API
            filters = [
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            ]

            if use_spot:
                filters.append({"Type": "TERM_MATCH", "Field": "marketoption", "Value": "Spot"})
            else:
                filters.append({"Type": "TERM_MATCH", "Field": "marketoption", "Value": "OnDemand"})

            response = self.pricing_client.get_products(
                ServiceCode="AmazonEC2",
                Filters=filters,
                MaxResults=1,
            )

            if response["PriceList"]:
                price_data = json.loads(response["PriceList"][0])
                # Extract the price from the complex pricing structure
                price = self._extract_price_from_response(price_data)
                return price

        except Exception as e:
            logger.warning(f"Failed to get pricing for {instance_type}: {e}")

        return self._get_fallback_pricing(instance_type, use_spot)

    def _extract_price_from_response(self, price_data: Dict) -> float:
        """Extract hourly price from AWS pricing response."""
        try:
            # Navigate through the pricing structure
            terms = price_data.get("terms", {})

            # For on-demand, look in the OnDemand terms
            if "OnDemand" in terms:
                term_id = list(terms["OnDemand"].keys())[0]
                price_dimensions = terms["OnDemand"][term_id]["priceDimensions"]
                price_dimension_id = list(price_dimensions.keys())[0]
                price_per_unit = price_dimensions[price_dimension_id]["pricePerUnit"]["USD"]
                return float(price_per_unit)

            # For spot, look in the Spot terms
            elif "Spot" in terms:
                term_id = list(terms["Spot"].keys())[0]
                price_dimensions = terms["Spot"][term_id]["priceDimensions"]
                price_dimension_id = list(price_dimensions.keys())[0]
                price_per_unit = price_dimensions[price_dimension_id]["pricePerUnit"]["USD"]
                return float(price_per_unit)

        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Failed to extract price from response: {e}")

        return 0.0

    def _get_fallback_pricing(self, instance_type: str, use_spot: bool) -> Optional[float]:
        """Get fallback pricing from hardcoded data."""
        # Fallback pricing data (updated periodically)
        fallback_pricing = {
            "g4dn.xlarge": {"on_demand": 0.526, "spot": 0.1578},
            "g5.xlarge": {"on_demand": 1.006, "spot": 0.3018},
            "g5.2xlarge": {"on_demand": 1.212, "spot": 0.3636},
            "g5.4xlarge": {"on_demand": 2.424, "spot": 0.7272},
            "g5.8xlarge": {"on_demand": 4.848, "spot": 1.4544},
            "g5.12xlarge": {"on_demand": 7.272, "spot": 2.1816},
            "g5.24xlarge": {"on_demand": 14.544, "spot": 4.3632},
            "p3.2xlarge": {"on_demand": 3.06, "spot": 0.918},
            "p3.8xlarge": {"on_demand": 12.24, "spot": 3.672},
            "p3.16xlarge": {"on_demand": 24.48, "spot": 7.344},
            # Add more instance types as needed
        }

        if instance_type in fallback_pricing:
            pricing = fallback_pricing[instance_type]
            return pricing["spot"] if use_spot else pricing["on_demand"]

        logger.warning(f"No pricing data available for {instance_type}")
        return None


class SkyPilotInstanceInfo:
    """Extract instance information from SkyPilot environment."""

    @staticmethod
    def get_instance_info() -> Tuple[str, bool, int, int]:
        """Get instance information from SkyPilot environment variables.

        Returns:
            Tuple of (instance_type, use_spot, num_nodes, num_gpus_per_node)
        """
        # Try to get instance type from environment
        instance_type = os.environ.get("SKYPILOT_INSTANCE_TYPE", "g5.4xlarge")

        # Determine if using spot instances
        use_spot = os.environ.get("SKYPILOT_USE_SPOT", "true").lower() == "true"

        # Get number of nodes
        num_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", "1"))

        # Get GPUs per node
        num_gpus_per_node = int(os.environ.get("SKYPILOT_NUM_GPUS_PER_NODE", "1"))

        return instance_type, use_spot, num_nodes, num_gpus_per_node

    @staticmethod
    def get_instance_info_from_task(task_yaml_path: str) -> Tuple[str, bool, int, int]:
        """Extract instance information from SkyPilot task YAML.

        Args:
            task_yaml_path: Path to SkyPilot task YAML file

        Returns:
            Tuple of (instance_type, use_spot, num_nodes, num_gpus_per_node)
        """
        try:
            import yaml

            with open(task_yaml_path, "r") as f:
                task_config = yaml.safe_load(f)

            resources = task_config.get("resources", {})

            # Extract instance type from resources
            instance_type = "g5.4xlarge"  # default
            use_spot = False
            num_nodes = 1
            num_gpus_per_node = 1

            # Look for instance type in any_of resources
            if "any_of" in resources:
                for resource in resources["any_of"]:
                    if "instance_type" in resource:
                        instance_type = resource["instance_type"]
                    if "use_spot" in resource:
                        use_spot = resource["use_spot"]
                    if "accelerators" in resource:
                        # Parse accelerator string like "L4:1" or "A10G:2"
                        accel_str = resource["accelerators"]
                        if ":" in accel_str:
                            num_gpus_per_node = int(accel_str.split(":")[1])

            # Get node count
            if "num_nodes" in task_config:
                num_nodes = task_config["num_nodes"]

            return instance_type, use_spot, num_nodes, num_gpus_per_node

        except Exception as e:
            logger.warning(f"Failed to parse task YAML {task_yaml_path}: {e}")
            return "g5.4xlarge", False, 1, 1

    @staticmethod
    def get_instance_info_from_aws_cli() -> Tuple[str, bool, int, int]:
        """Get instance information by querying AWS CLI.

        This method tries to find the current instance and get its details.

        Returns:
            Tuple of (instance_type, use_spot, num_nodes, num_gpus_per_node)
        """
        try:
            # Get instance metadata
            result = subprocess.run(
                ["curl", "-s", "http://169.254.169.254/latest/meta-data/instance-type"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                instance_type = result.stdout.strip()

                # Check if it's a spot instance
                spot_result = subprocess.run(
                    ["curl", "-s", "http://169.254.169.254/latest/meta-data/spot/termination-time"],
                    capture_output=True,
                )
                use_spot = spot_result.returncode == 0

                # For now, assume single node and single GPU
                # This could be enhanced by checking environment variables or other metadata
                num_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", "1"))
                num_gpus_per_node = int(os.environ.get("SKYPILOT_NUM_GPUS_PER_NODE", "1"))

                return instance_type, use_spot, num_nodes, num_gpus_per_node

        except Exception as e:
            logger.warning(f"Failed to get instance info from AWS CLI: {e}")

        return "g5.4xlarge", False, 1, 1


def calculate_total_cost(
    hours: float,
    instance_type: str = None,
    use_spot: bool = None,
    num_nodes: int = None,
    num_gpus_per_node: int = None,
    region: str = "us-east-1",
    profile: str = "softmax",
) -> float:
    """Calculate total cost for a training run.

    Args:
        hours: Number of hours the instance(s) ran
        instance_type: AWS instance type (if None, will be detected)
        use_spot: Whether using spot instances (if None, will be detected)
        num_nodes: Number of nodes (if None, will be detected)
        num_gpus_per_node: Number of GPUs per node (if None, will be detected)
        region: AWS region
        profile: AWS profile to use

    Returns:
        Total cost in USD
    """
    # Detect instance information if not provided
    if instance_type is None or use_spot is None or num_nodes is None or num_gpus_per_node is None:
        detected_instance, detected_spot, detected_nodes, detected_gpus = SkyPilotInstanceInfo.get_instance_info()

        instance_type = instance_type or detected_instance
        use_spot = use_spot if use_spot is not None else detected_spot
        num_nodes = num_nodes or detected_nodes
        num_gpus_per_node = num_gpus_per_node or detected_gpus

    # Get pricing client
    pricing_client = AWSPricingClient(region=region, profile=profile)

    # Get price per hour for single instance
    price_per_hour = pricing_client.get_instance_pricing(instance_type, use_spot)

    if price_per_hour is None:
        logger.warning(f"Could not determine pricing for {instance_type}, using fallback")
        price_per_hour = 2.424  # g5.4xlarge on-demand fallback

    # Calculate total cost
    total_cost = price_per_hour * hours * num_nodes

    logger.info(f"Cost calculation: {hours:.2f}h × ${price_per_hour:.3f}/h × {num_nodes} nodes = ${total_cost:.2f}")

    return total_cost
