# Datadog Collectors

Automated metric collection from multiple services for Datadog monitoring.

## Quick Links

- **[Collectors Architecture](docs/COLLECTORS_ARCHITECTURE.md)** - System design and principles
- **[Adding New Collector](docs/ADDING_NEW_COLLECTOR.md)** - Step-by-step implementation guide
- **[CI/CD Metrics](docs/CI_CD_METRICS.md)** - GitHub metrics catalog
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production and dev deployment

## System Overview

Automated metric collection from multiple services via Kubernetes CronJobs: **Current Collectors:**

- **GitHub** ✅ **(Production)**: 28 metrics - PRs, commits, branches, CI/CD workflows, developer activity
- **Skypilot** ✅ **(Production)**: 30 metrics - Jobs, clusters, resource utilization
- **Asana** ✅ **(Production)**: 14 metrics - Project health, bugs tracking, team velocity
- **EC2** ✅ **(Production)**: 19 metrics - Instances, costs, utilization, EBS volumes
- **WandB** ✅ **(Production)**: 11 per-run metrics - Training runs with per-run tags (duration, SPS, hearts, latency)
  for PTM/sweep/all runs
- **Kubernetes** ✅ **(Production)**: 15 metrics - Resource efficiency, pod health, waste tracking
- **Health FoM** ✅ **(Production)**: 14 metrics - Normalized 0.0-1.0 health scores for CI/CD and training metrics All
  collectors:
- Run together every 15 minutes via unified CronJob
- Use AWS Secrets Manager for credentials
- Emit health metrics about themselves
- Handle errors gracefully (continue on individual failures)
- Timeout protection (120s per collector) prevents indefinite hangs

## Requirements

### Python Dependencies

Managed via `uv` (already configured in the project):

```bash
# Install/update dependencies
metta install
```

### AWS Credentials

Collectors require AWS credentials for Secrets Manager access:

```bash
# Configure AWS CLI (if not already done)
aws configure
# Verify access to secrets
aws secretsmanager get-secret-value --secret-id github/dashboard-token --region us-east-1
```

### Datadog API Keys & Secrets

All secrets are stored in AWS Secrets Manager. See [docs/SECRETS_SETUP.md](docs/SECRETS_SETUP.md) for complete setup
guide. **Both Production and Local Development** use AWS Secrets Manager:

```bash
# Configure AWS CLI (one-time setup)
aws configure
# Verify access to secrets
uv run python scripts/validate_secrets.py
```

**Production**: Uses IRSA (IAM Roles for Service Accounts) - automatic credential access **Local Development**: Uses
your personal AWS CLI credentials

## Quick Start

### Local Development & Testing

```bash
# Run all collectors locally (pushes metrics to Datadog)
cd devops/datadog
uv run python scripts/run_all_collectors.py
# Run individual collector
uv run python scripts/run_collector.py github --verbose
```

### View Collected Metrics

```bash
# Run all collectors (recommended - used by CronJob)
uv run python devops/datadog/scripts/run_all_collectors.py
# Run individual collector
uv run python devops/datadog/scripts/run_collector.py github --verbose
uv run python devops/datadog/scripts/run_collector.py kubernetes --verbose
uv run python devops/datadog/scripts/run_collector.py ec2 --verbose
# Push metrics to Datadog
uv run python devops/datadog/scripts/run_all_collectors.py  # Pushes all
uv run python devops/datadog/scripts/run_collector.py github --push  # Individual
```

## Add a New Collector

See [Adding New Collector](docs/ADDING_NEW_COLLECTOR.md) for complete guide.

```bash
# 1. Create collector structure
mkdir -p collectors/skypilot
touch collectors/skypilot/{__init__.py,collector.py,metrics.py,README.md}
# 2. Implement collector (see guide)
# 3. Add secrets to AWS Secrets Manager
# 4. Configure Helm chart in devops/charts/
# 5. Deploy
cd ../charts
helm upgrade --install datadog-collectors ./datadog-collectors
```

## Directory Structure

```
devops/datadog/
├── collectors/         # Data collector modules
│   ├── base.py        # Base collector class (shared interface)
│   ├── github/        # GitHub metrics (PRs, commits, CI/CD)
│   ├── skypilot/      # Skypilot jobs and compute costs
│   ├── wandb/         # WandB training runs and experiments
│   ├── ec2/           # AWS EC2 instances and costs
│   ├── asana/         # Asana tasks and project velocity
│   ├── kubernetes/    # Kubernetes resource efficiency and pod health
│   └── health_fom/    # Normalized health scores (0.0-1.0)
│
├── utils/             # Shared utilities
│   ├── datadog_client.py  # Datadog metric submission
│   ├── registry.py        # Metric registry and auto-discovery
│   └── decorators.py      # @metric decorator
│
├── scripts/            # Management scripts
│   └── run_collector.py    # Run any collector
│
├── docs/               # Documentation
│   ├── COLLECTORS_ARCHITECTURE.md
│   ├── ADDING_NEW_COLLECTOR.md
│   └── CI_CD_METRICS.md
│
└── tests/              # Test suite
    └── collectors/     # Collector tests
```

## Documentation

- **[Collectors Architecture](docs/COLLECTORS_ARCHITECTURE.md)** - System design, patterns, deployment
- **[Adding New Collector](docs/ADDING_NEW_COLLECTOR.md)** - Step-by-step implementation guide
- **[Metric Best Practices](docs/METRIC_BEST_PRACTICES.md)** - Guide for choosing effective metrics
- **[Datadog Integration Analysis](docs/DATADOG_INTEGRATION_ANALYSIS.md)** - Current integration architecture
- **[CI/CD Metrics](docs/CI_CD_METRICS.md)** - Complete GitHub metrics catalog

## Common Commands

```bash
# Collector management (all collectors run together every 15 minutes)
uv run python devops/datadog/scripts/run_all_collectors.py              # Run all collectors
uv run python devops/datadog/scripts/run_collector.py github --verbose  # Run individual collector
uv run python devops/datadog/scripts/run_collector.py kubernetes --push # Push individual collector
```

## Architecture Philosophy

**Modular, self-contained collectors** that scale as you add services:

1. Each collector focuses on one service (GitHub, Skypilot, WandB, etc.)
2. Common base class provides health monitoring and error handling
3. Metrics defined via decorators, automatically registered
4. Deployed as Kubernetes CronJobs via Helm
5. Credentials managed via AWS Secrets Manager

---

**Ready to get started?**

- Run collectors: `uv run python devops/datadog/scripts/run_all_collectors.py`
- Add collectors: [Adding New Collector](docs/ADDING_NEW_COLLECTOR.md)
