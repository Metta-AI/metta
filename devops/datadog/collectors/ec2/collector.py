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
        super().__init__(name="ec2")
        self.region = region
        self.ec2_client = boto3.client("ec2", region_name=region)
        self.cloudwatch_client = boto3.client("cloudwatch", region_name=region)

    def collect_metrics(self) -> dict[str, Any]:
        metrics = {}

        metrics.update(self._collect_instance_metrics())
        metrics.update(self._collect_ebs_metrics())
        metrics.update(self._collect_cost_metrics())

        return metrics

    def _collect_instance_metrics(self) -> dict[str, Any]:
        metrics = {}

        gpu_counts = {
            "p2.xlarge": 1,
            "p2.8xlarge": 8,
            "p2.16xlarge": 16,
            "p3.2xlarge": 1,
            "p3.8xlarge": 4,
            "p3.16xlarge": 8,
            "p3dn.24xlarge": 8,
            "p4d.24xlarge": 8,
            "p4de.24xlarge": 8,
            "p5.48xlarge": 8,
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
            response = self.ec2_client.describe_instances()

            total_count = 0
            running_count = 0
            stopped_count = 0
            spot_count = 0
            ondemand_count = 0
            gpu_instances = 0
            total_gpus = 0
            total_cpus = 0
            instance_ages = []

            now = datetime.datetime.now(datetime.timezone.utc)

            for reservation in response["Reservations"]:
                for instance in reservation["Instances"]:
                    total_count += 1

                    state = instance["State"]["Name"]
                    if state == "running":
                        running_count += 1
                    elif state == "stopped":
                        stopped_count += 1

                    lifecycle = instance.get("InstanceLifecycle", "")
                    if lifecycle == "spot":
                        spot_count += 1
                    else:
                        ondemand_count += 1

                    instance_type = instance.get("InstanceType", "")

                    if instance_type in gpu_counts:
                        gpu_instances += 1
                        total_gpus += gpu_counts[instance_type]

                    if "." in instance_type:
                        size = instance_type.split(".")[1]
                        if "xlarge" in size:
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
                            total_cpus += 4 * multiplier
                        else:
                            total_cpus += 2

                    launch_time = instance.get("LaunchTime")
                    if launch_time:
                        age_days = (now - launch_time).total_seconds() / 86400
                        instance_ages.append(age_days)

            metrics["ec2.instances"] = [
                (total_count, ["status:total"]),
                (running_count, ["status:running"]),
                (stopped_count, ["status:stopped"]),
                (spot_count, ["pricing:spot"]),
                (ondemand_count, ["pricing:ondemand"]),
                (gpu_instances, ["type:gpu"]),
            ]

            metrics["ec2.resources.gpus"] = [(total_gpus, [])]
            metrics["ec2.resources.cpus"] = [(total_cpus, [])]

            if instance_ages:
                avg_age = sum(instance_ages) / len(instance_ages)
                max_age = max(instance_ages)
                metrics["ec2.instances.age_days"] = [
                    (avg_age, ["metric:avg"]),
                    (max_age, ["metric:max"]),
                ]

        except Exception as e:
            self.logger.error(f"Failed to collect instance metrics: {e}")

        return metrics

    def _collect_ebs_metrics(self) -> dict[str, Any]:
        metrics = {}

        try:
            volumes_response = self.ec2_client.describe_volumes()

            total_volumes = 0
            attached_volumes = 0
            unattached_volumes = 0
            total_volume_size = 0

            for volume in volumes_response["Volumes"]:
                total_volumes += 1
                total_volume_size += volume["Size"]

                if volume.get("Attachments"):
                    attached_volumes += 1
                else:
                    unattached_volumes += 1

            metrics["ec2.ebs.volumes"] = [
                (total_volumes, ["status:total"]),
                (attached_volumes, ["status:attached"]),
                (unattached_volumes, ["status:unattached"]),
            ]

            metrics["ec2.ebs.volumes.size_gb"] = [(total_volume_size, [])]

            snapshots_response = self.ec2_client.describe_snapshots(OwnerIds=["self"])

            total_snapshots = 0
            total_snapshot_size = 0

            for snapshot in snapshots_response["Snapshots"]:
                total_snapshots += 1
                total_snapshot_size += snapshot["VolumeSize"]

            metrics["ec2.ebs.snapshots"] = [(total_snapshots, [])]
            metrics["ec2.ebs.snapshots.size_gb"] = [(total_snapshot_size, [])]

        except Exception as e:
            self.logger.error(f"Failed to collect EBS metrics: {e}")

        return metrics

    def _collect_cost_metrics(self) -> dict[str, Any]:
        metrics = {}

        try:
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

            estimated_hourly = running_count * 0.10
            estimated_monthly = estimated_hourly * 24 * 30

            metrics["ec2.cost"] = [
                (estimated_hourly, ["metric:hourly_estimate"]),
                (estimated_monthly, ["metric:monthly_estimate"]),
            ]

            if running_count > 0:
                spot_savings_pct = (spot_count / running_count) * 70.0
                metrics["ec2.cost.spot_savings_pct"] = [(spot_savings_pct, [])]

        except Exception as e:
            self.logger.error(f"Failed to collect cost metrics: {e}")

        return metrics
