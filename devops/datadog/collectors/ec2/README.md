# AWS EC2 Collector

Collects EC2 instance and cost metrics using AWS boto3 SDK.

**Status**: 📋 **Planned**

## Metrics to Collect

### Instance Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `ec2.instances.running` | Currently running instances | count | High |
| `ec2.instances.stopped` | Stopped instances | count | Medium |
| `ec2.instances.total` | Total instances (all states) | count | Low |
| `ec2.instances.spot` | Running spot instances | count | Medium |
| `ec2.instances.on_demand` | Running on-demand instances | count | Medium |
| `ec2.instances.reserved` | Running reserved instances | count | Low |

### Instance Type Distribution

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `ec2.instances.gpu_count` | Instances with GPUs (p3, p4, g4, etc.) | count | High |
| `ec2.instances.cpu_only_count` | CPU-only instances | count | Medium |
| `ec2.instances.by_type.{type}` | Count by instance type (e.g., p3.2xlarge) | count | Low |

### Cost Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `ec2.cost.running_hourly` | Estimated hourly cost for running instances | usd | High |
| `ec2.cost.daily_estimate` | Estimated daily cost | usd | High |
| `ec2.cost.monthly_estimate` | Projected monthly cost | usd | High |
| `ec2.cost.spot_savings` | Savings from spot vs on-demand | usd | Medium |

### Utilization Metrics (via CloudWatch)

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `ec2.utilization.avg_cpu_percent` | Average CPU utilization across instances | percent | High |
| `ec2.utilization.max_cpu_percent` | Highest CPU utilization | percent | Medium |
| `ec2.utilization.avg_memory_percent` | Average memory utilization (requires CloudWatch agent) | percent | Medium |
| `ec2.utilization.idle_instances` | Instances with <10% CPU for >1 hour | count | High |

### Network Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `ec2.network.total_egress_gb_7d` | Total network egress in 7 days | gb | Low |
| `ec2.network.total_ingress_gb_7d` | Total network ingress in 7 days | gb | Low |

### Storage Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `ec2.ebs.volumes_total` | Total EBS volumes | count | Low |
| `ec2.ebs.storage_total_gb` | Total EBS storage provisioned | gb | Medium |
| `ec2.ebs.cost_monthly_estimate` | Estimated monthly EBS cost | usd | Medium |
| `ec2.ebs.unattached_volumes` | Volumes not attached to instances | count | High |

### Region Distribution

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `ec2.instances.by_region.us_east_1` | Instances in us-east-1 | count | Low |
| `ec2.instances.by_region.us_west_2` | Instances in us-west-2 | count | Low |
| *(repeat for monitored regions)* | | | |

## Configuration

### Required Secrets (AWS Secrets Manager)

- `ec2/access-key-id` - AWS Access Key ID (or use IAM role)
- `ec2/secret-access-key` - AWS Secret Access Key (or use IAM role)

**Note**: Prefer IAM role for EKS ServiceAccount over static credentials.

### Environment Variables

- `AWS_REGION` - Default region for queries (default: us-east-1)
- `AWS_REGIONS` - Comma-separated list of regions to monitor (default: us-east-1,us-west-2)
- `EC2_TAG_FILTER` - Filter instances by tag (e.g., "Environment=production")

### Collection Schedule

**Recommended**: Every 5 minutes (`*/5 * * * *`)

- Instance states change frequently
- Cost tracking benefits from frequent updates
- Utilization metrics need regular sampling

## Implementation Notes

### Sample Metric Implementation

```python
from datetime import datetime, timedelta, timezone
import boto3

from devops.datadog.common.decorators import metric
from devops.datadog.common.secrets import get_secret


def _get_ec2_client(region: str = "us-east-1"):
    """Create boto3 EC2 client."""
    # If using IAM role, credentials not needed
    return boto3.client("ec2", region_name=region)


@metric("ec2.instances.running", unit="count")
def get_running_instances() -> int:
    """Number of currently running EC2 instances."""
    client = _get_ec2_client()

    response = client.describe_instances(
        Filters=[{"Name": "instance-state-name", "Values": ["running"]}]
    )

    count = 0
    for reservation in response["Reservations"]:
        count += len(reservation["Instances"])

    return count


@metric("ec2.instances.spot", unit="count")
def get_spot_instances() -> int:
    """Number of running spot instances."""
    client = _get_ec2_client()

    response = client.describe_instances(
        Filters=[
            {"Name": "instance-state-name", "Values": ["running"]},
            {"Name": "instance-lifecycle", "Values": ["spot"]},
        ]
    )

    count = 0
    for reservation in response["Reservations"]:
        count += len(reservation["Instances"])

    return count


@metric("ec2.cost.running_hourly", unit="usd")
def get_running_hourly_cost() -> float:
    """Estimated hourly cost for all running instances."""
    client = _get_ec2_client()

    response = client.describe_instances(
        Filters=[{"Name": "instance-state-name", "Values": ["running"]}]
    )

    total_hourly_cost = 0.0

    # Pricing data (simplified - could fetch from AWS Pricing API)
    INSTANCE_PRICING = {
        "p3.2xlarge": 3.06,
        "p3.8xlarge": 12.24,
        "g4dn.xlarge": 0.526,
        "c5.2xlarge": 0.34,
        # ... more types
    }

    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            instance_type = instance["InstanceType"]
            hourly_rate = INSTANCE_PRICING.get(instance_type, 0.1)  # Default fallback
            total_hourly_cost += hourly_rate

    return total_hourly_cost


@metric("ec2.utilization.idle_instances", unit="count")
def get_idle_instances() -> int:
    """Instances with <10% CPU utilization for >1 hour."""
    ec2_client = _get_ec2_client()
    cloudwatch = boto3.client("cloudwatch")

    # Get running instances
    response = ec2_client.describe_instances(
        Filters=[{"Name": "instance-state-name", "Values": ["running"]}]
    )

    idle_count = 0
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=1)

    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            instance_id = instance["InstanceId"]

            # Get CPU utilization from CloudWatch
            metrics = cloudwatch.get_metric_statistics(
                Namespace="AWS/EC2",
                MetricName="CPUUtilization",
                Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=["Average"],
            )

            if metrics["Datapoints"]:
                avg_cpu = metrics["Datapoints"][0]["Average"]
                if avg_cpu < 10:
                    idle_count += 1

    return idle_count
```

## Dashboard Widgets

Recommended visualizations:

1. **Instance Overview**
   - Query value: Running instances
   - Query value: Spot vs On-Demand
   - Timeseries: Instance count over time

2. **Cost Tracking**
   - Query value: Hourly cost
   - Query value: Daily estimate
   - Query value: Monthly projection
   - Timeseries: Cost trend

3. **Utilization**
   - Query value: Average CPU %
   - Query value: Idle instances
   - Heatmap: CPU utilization by instance

4. **Resource Distribution**
   - Pie chart: Instances by type
   - Pie chart: Instances by region
   - Query value: GPU instance count

## Alerting Recommendations

```yaml
# High cost alert
Query: avg(last_1h):sum:ec2.cost.running_hourly{} > 50
Alert: EC2 hourly cost exceeds $50

# Idle instances
Query: avg(last_2h):sum:ec2.utilization.idle_instances{} > 5
Alert: More than 5 idle instances for 2+ hours

# Unattached volumes
Query: avg(last_1h):sum:ec2.ebs.unattached_volumes{} > 10
Alert: More than 10 unattached EBS volumes (waste)

# High monthly projection
Query: avg(last_24h):sum:ec2.cost.monthly_estimate{} > 10000
Alert: Projected monthly EC2 cost exceeds $10,000
```

## Dependencies

- `boto3` - AWS SDK for Python
- `botocore` - Low-level AWS API access

## Challenges & Considerations

1. **Multi-Region**: Need to query multiple regions, aggregate results
2. **IAM Permissions**: Requires `ec2:Describe*` and `cloudwatch:GetMetricStatistics`
3. **Pricing Data**: AWS Pricing API is complex, may use static pricing table
4. **Reserved Instances**: Tracking RI utilization is complex
5. **Spot Instance Interruptions**: May want to track interruption rates

## IAM Policy Required

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeVolumes",
        "ec2:DescribeRegions",
        "cloudwatch:GetMetricStatistics",
        "pricing:GetProducts"
      ],
      "Resource": "*"
    }
  ]
}
```

## Related Documentation

- [AWS EC2 API Reference](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/)
- [boto3 EC2 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2.html)
- [Collectors Architecture](../../docs/COLLECTORS_ARCHITECTURE.md)
- [Adding New Collector](../../docs/ADDING_NEW_COLLECTOR.md)

## Next Steps

1. Configure IAM role for EKS ServiceAccount
2. Test boto3 queries in staging
3. Implement collector following template
4. Test with multiple regions
5. Verify cost calculations
6. Deploy to production
7. Set up cost alerts

## Maintenance

- **Owner**: DevOps Team / Infrastructure
- **Priority**: High (cost visibility critical)
- **API Version**: boto3 latest
- **Estimated Effort**: 4-6 hours (multi-region complexity)
- **Last Updated**: 2025-10-22
