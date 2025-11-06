# EC2 Collector

Monitors AWS EC2 infrastructure and costs.

## Metrics

| Metric | Type | Tags | Description |
|--------|------|------|-------------|
| `ec2.instances` | gauge | `status:total\|running\|stopped`, `pricing:spot\|ondemand`, `type:gpu` | Instance counts |
| `ec2.instances.age_days` | gauge | `metric:avg\|max` | Instance age statistics |
| `ec2.resources.gpus` | gauge | - | Total GPU count |
| `ec2.resources.cpus` | gauge | - | Total CPU count |
| `ec2.ebs.volumes` | gauge | `status:total\|attached\|unattached` | EBS volume counts |
| `ec2.ebs.volumes.size_gb` | gauge | - | Total volume size |
| `ec2.ebs.snapshots` | gauge | - | Snapshot count |
| `ec2.ebs.snapshots.size_gb` | gauge | - | Total snapshot size |
| `ec2.cost` | gauge | `metric:hourly_estimate\|monthly_estimate` | Cost estimates |
| `ec2.cost.spot_savings_pct` | gauge | - | Spot instance savings percentage |

**Total**: 10 metric names with dimensional tags

## Configuration

```bash
# Environment Variables
AWS_REGION=us-east-1    # AWS region to monitor
```

Requires AWS credentials with EC2 read permissions.

## Usage

```bash
# Test locally
uv run python devops/datadog/scripts/run_collector.py ec2 --verbose

# Push to Datadog
uv run python devops/datadog/scripts/run_collector.py ec2 --push
```

## Dashboard Queries

```python
# Running instances by type
sum:ec2.instances{status:running,pricing:spot}

# Monthly cost estimate
sum:ec2.cost{metric:monthly_estimate}

# Spot savings
avg:ec2.cost.spot_savings_pct{}
```
