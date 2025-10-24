"""EC2 metrics collector for Datadog monitoring."""

import datetime
from typing import Any

import boto3

from devops.datadog.utils.base import BaseCollector


class EC2Collector(BaseCollector):
    """Collector for EC2 instance and infrastructure metrics.

    Collects comprehensive metrics about EC2 instances, costs, utilization,
    EBS volumes, and other infrastructure resources.
    """

    def __init__(self, region: str = "us-east-1"):
        """Initialize EC2 collector.

        Args:
            region: AWS region to collect metrics from
        """
        super().__init__(name="ec2")
        self.region = region
        self.ec2_client = boto3.client("ec2", region_name=region)
        self.cloudwatch_client = boto3.client("cloudwatch", region_name=region)

    def collect_metrics(self) -> dict[str, Any]:
        """Collect all EC2 metrics."""
        metrics = {}

        # Collect instance metrics
        metrics.update(self._collect_instance_metrics())

        # Collect EBS metrics
        metrics.update(self._collect_ebs_metrics())

        # Collect cost estimates
        metrics.update(self._collect_cost_metrics())

        return metrics

    def _collect_instance_metrics(self) -> dict[str, Any]:
        """Collect EC2 instance metrics."""
        metrics = {
            # Instance counts
            "ec2.instances.total": 0,
            "ec2.instances.running": 0,
            "ec2.instances.stopped": 0,
            "ec2.instances.spot": 0,
            "ec2.instances.ondemand": 0,
            # Instance types
            "ec2.instances.gpu_instances": 0,
            "ec2.instances.gpu_count": 0,
            "ec2.instances.cpu_count": 0,
            # Utilization
            "ec2.instances.idle": 0,
            # Age
            "ec2.instances.avg_age_days": None,
            "ec2.instances.oldest_age_days": None,
        }

        # GPU counts per instance type
        # Based on AWS documentation for common GPU instance types
        gpu_counts = {
            # P2 instances (NVIDIA K80)
            "p2.xlarge": 1,
            "p2.8xlarge": 8,
            "p2.16xlarge": 16,
            # P3 instances (NVIDIA V100)
            "p3.2xlarge": 1,
            "p3.8xlarge": 4,
            "p3.16xlarge": 8,
            "p3dn.24xlarge": 8,
            # P4 instances (NVIDIA A100)
            "p4d.24xlarge": 8,
            "p4de.24xlarge": 8,
            # P5 instances (NVIDIA H100)
            "p5.48xlarge": 8,
            # G4 instances (NVIDIA T4)
            "g4dn.xlarge": 1,
            "g4dn.2xlarge": 1,
            "g4dn.4xlarge": 1,
            "g4dn.8xlarge": 1,
            "g4dn.12xlarge": 4,
            "g4dn.16xlarge": 1,
            "g4ad.xlarge": 1,
            "g4ad.2xlarge": 1,
            "g4ad.4xlarge": 1,
            "g4ad.8xlarge": 2,
            "g4ad.16xlarge": 4,
            # G5 instances (NVIDIA A10G)
            "g5.xlarge": 1,
            "g5.2xlarge": 1,
            "g5.4xlarge": 1,
            "g5.8xlarge": 1,
            "g5.12xlarge": 4,
            "g5.16xlarge": 1,
            "g5.24xlarge": 4,
            "g5.48xlarge": 8,
            "g5g.xlarge": 1,
            "g5g.2xlarge": 1,
            "g5g.4xlarge": 1,
            "g5g.8xlarge": 1,
            "g5g.16xlarge": 2,
        }

        try:
            # Get all instances
            response = self.ec2_client.describe_instances()

            instance_ages = []
            now = datetime.datetime.now(datetime.timezone.utc)

            for reservation in response["Reservations"]:
                for instance in reservation["Instances"]:
                    metrics["ec2.instances.total"] += 1

                    # State tracking
                    state = instance["State"]["Name"]
                    if state == "running":
                        metrics["ec2.instances.running"] += 1
                    elif state == "stopped":
                        metrics["ec2.instances.stopped"] += 1

                    # Spot vs On-Demand
                    lifecycle = instance.get("InstanceLifecycle", "")
                    if lifecycle == "spot":
                        metrics["ec2.instances.spot"] += 1
                    else:
                        metrics["ec2.instances.ondemand"] += 1

                    # Instance type analysis
                    instance_type = instance.get("InstanceType", "")

                    # Count GPU instances and total GPUs
                    if instance_type in gpu_counts:
                        metrics["ec2.instances.gpu_instances"] += 1
                        metrics["ec2.instances.gpu_count"] += gpu_counts[instance_type]

                    # CPU count (approximate based on instance type)
                    # This is a simplified mapping - could be enhanced
                    if "." in instance_type:
                        size = instance_type.split(".")[1]
                        if "xlarge" in size:
                            # Rough estimate: small=2, medium=2, large=2, xlarge=4, 2xlarge=8, etc.
                            multiplier = 1
                            if size.startswith("2x"):
                                multiplier = 2
                            elif size.startswith("4x"):
                                multiplier = 4
                            elif size.startswith("8x"):
                                multiplier = 8
                            elif size.startswith("12x"):
                                multiplier = 12
                            elif size.startswith("16x"):
                                multiplier = 16
                            metrics["ec2.instances.cpu_count"] += 4 * multiplier
                        else:
                            metrics["ec2.instances.cpu_count"] += 2

                    # Age calculation
                    launch_time = instance.get("LaunchTime")
                    if launch_time:
                        age_days = (now - launch_time).total_seconds() / 86400
                        instance_ages.append(age_days)

            # Calculate age statistics
            if instance_ages:
                metrics["ec2.instances.avg_age_days"] = sum(instance_ages) / len(instance_ages)
                metrics["ec2.instances.oldest_age_days"] = max(instance_ages)

        except Exception as e:
            self.logger.error(f"Failed to collect instance metrics: {e}")
            for key in metrics:
                metrics[key] = None

        return metrics

    def _collect_ebs_metrics(self) -> dict[str, Any]:
        """Collect EBS volume metrics."""
        metrics = {
            "ec2.ebs.volumes.total": 0,
            "ec2.ebs.volumes.attached": 0,
            "ec2.ebs.volumes.unattached": 0,
            "ec2.ebs.volumes.size_gb": 0,
            "ec2.ebs.snapshots.total": 0,
            "ec2.ebs.snapshots.size_gb": 0,
        }

        try:
            # Get all volumes
            volumes_response = self.ec2_client.describe_volumes()

            for volume in volumes_response["Volumes"]:
                metrics["ec2.ebs.volumes.total"] += 1
                metrics["ec2.ebs.volumes.size_gb"] += volume["Size"]

                # Check attachment status
                if volume.get("Attachments"):
                    metrics["ec2.ebs.volumes.attached"] += 1
                else:
                    metrics["ec2.ebs.volumes.unattached"] += 1

            # Get snapshots
            snapshots_response = self.ec2_client.describe_snapshots(OwnerIds=["self"])

            for snapshot in snapshots_response["Snapshots"]:
                metrics["ec2.ebs.snapshots.total"] += 1
                metrics["ec2.ebs.snapshots.size_gb"] += snapshot["VolumeSize"]

        except Exception as e:
            self.logger.error(f"Failed to collect EBS metrics: {e}")
            for key in metrics:
                metrics[key] = None

        return metrics

    def _collect_cost_metrics(self) -> dict[str, Any]:
        """Collect cost estimate metrics."""
        metrics = {
            "ec2.cost.running_hourly_estimate": None,
            "ec2.cost.monthly_estimate": None,
            "ec2.cost.spot_savings_pct": None,
        }

        # Note: This is a simplified cost estimation
        # For production, you'd want to use AWS Cost Explorer API or pricing API
        # This is just a placeholder showing the structure

        try:
            # Simplified cost estimation based on instance counts
            # Real implementation would use AWS Pricing API
            running_count = 0
            spot_count = 0

            response = self.ec2_client.describe_instances(
                Filters=[{"Name": "instance-state-name", "Values": ["running"]}]
            )

            for reservation in response["Reservations"]:
                for instance in reservation["Instances"]:
                    running_count += 1
                    if instance.get("InstanceLifecycle") == "spot":
                        spot_count += 1

            # Rough estimate: $0.10/hour per instance (this is very approximate)
            # Real implementation should use actual pricing
            estimated_hourly = running_count * 0.10
            metrics["ec2.cost.running_hourly_estimate"] = estimated_hourly
            metrics["ec2.cost.monthly_estimate"] = estimated_hourly * 24 * 30

            # Spot savings estimate (spot typically 70% cheaper)
            if running_count > 0:
                metrics["ec2.cost.spot_savings_pct"] = (spot_count / running_count) * 70.0

        except Exception as e:
            self.logger.error(f"Failed to collect cost metrics: {e}")
            # Leave as None on error

        return metrics
