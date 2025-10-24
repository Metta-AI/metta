# Datadog Management System

Complete Datadog integration for Metta, including:

- **Data Collectors**: Automated metric collection from multiple services
- **Dashboard Management**: Jsonnet-based dashboard configuration

## Quick Links

### Data Collectors

- **[Collectors Architecture](docs/COLLECTORS_ARCHITECTURE.md)** - System design and principles
- **[Adding New Collector](docs/ADDING_NEW_COLLECTOR.md)** - Step-by-step implementation guide
- **[CI/CD Metrics](docs/CI_CD_METRICS.md)** - GitHub metrics catalog

### Dashboards

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 10 minutes
- **[Jsonnet Design](docs/JSONNET_DESIGN.md)** - Architecture and patterns
- **[Widget Reference](docs/DATADOG_WIDGET_REFERENCE.md)** - Available widgets

## System Overview

### Data Collectors

Automated metric collection from multiple services via Kubernetes CronJobs:

**Current Collectors:**

- **GitHub** ✅ **(Production)**: PRs, commits, branches, CI/CD workflows, developer activity
- **Skypilot** ✅ **(Production)**: Jobs, clusters, resource utilization
- **Asana** ✅ **(Production)**: Project health, bugs tracking, team velocity
- **EC2** ✅ **(Production)**: Instances, costs, utilization, EBS volumes
- **WandB** ✅ **(Production)**: Training runs, model performance, GPU hours
- **Kubernetes** ✅ **(Production)**: Resource efficiency, pod health, waste tracking
- **Health FoM** ✅ **(Production)**: Normalized 0.0-1.0 health scores for CI/CD metrics

All collectors:

- Run together every 15 minutes via unified CronJob
- Use AWS Secrets Manager for credentials
- Emit health metrics about themselves
- Handle errors gracefully (continue on individual failures)

### Dashboard Management

**Jsonnet-based dashboard configuration** (like Grafana's Grafonnet):

**Benefits:**

- Define widgets once, use in multiple dashboards
- Mix and match components
- Grid layouts with automatic positioning
- 10 lines of code instead of 200 lines of JSON
- Version control of modular components

## Requirements

### System Dependencies

```bash
# macOS
brew install jsonnet

# Ubuntu/Debian
apt-get install jsonnet

# Or install Go version (recommended for latest features)
go install github.com/google/go-jsonnet/cmd/jsonnet@latest
```

### Python Dependencies

Managed via `uv` (already configured in the project):

```bash
# Install/update dependencies
metta install

# Verify installation
metta softmax-system-health --help
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

All secrets are stored in AWS Secrets Manager. See [docs/SECRETS_SETUP.md](docs/SECRETS_SETUP.md) for complete setup guide.

**Both Production and Local Development** use AWS Secrets Manager:

```bash
# Configure AWS CLI (one-time setup)
aws configure

# Verify access to secrets
uv run python scripts/validate_secrets.py
```

**Production**: Uses IRSA (IAM Roles for Service Accounts) - automatic credential access
**Local Development**: Uses your personal AWS CLI credentials

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

### Deploy a Dashboard

```bash
# Setup (one time)
cd devops/datadog
# Configure secrets (see docs/SECRETS_SETUP.md for complete guide)
uv run python scripts/validate_secrets.py  # Validate configuration

# Daily workflow
vim dashboards/components/ci.libsonnet       # Edit widget components
vim dashboards/sources/my_dashboard.jsonnet  # Compose dashboard
metta datadog dashboard build                 # Build JSON from Jsonnet
metta datadog dashboard diff                  # Review changes
metta datadog dashboard push                  # Upload to Datadog
```

### Add a New Collector

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
│   ├── secrets.py         # AWS Secrets Manager access
│   ├── registry.py        # Metric registry and auto-discovery
│   └── decorators.py      # @metric decorator
│
├── dashboards/         # Dashboard files (grouped)
│   ├── lib/           # Jsonnet library (widget builders)
│   ├── components/    # Reusable widget collections
│   ├── sources/       # Dashboard definitions (SOURCE)
│   └── templates/     # Generated JSON (OUTPUT, gitignored)
├── scripts/            # Management scripts
│   ├── run_collector.py    # Run any collector
│   └── push_dashboard.py   # Push dashboards to Datadog
│
├── docs/               # Documentation
│   ├── COLLECTORS_ARCHITECTURE.md
│   ├── ADDING_NEW_COLLECTOR.md
│   ├── CI_CD_METRICS.md
│   └── ...
│
└── tests/              # Test suite
    └── collectors/     # Collector tests
```

## Documentation

### Data Collectors

- **[Collectors Architecture](docs/COLLECTORS_ARCHITECTURE.md)** - System design, patterns, deployment
- **[Adding New Collector](docs/ADDING_NEW_COLLECTOR.md)** - Step-by-step implementation guide
- **[Datadog Integration Analysis](docs/DATADOG_INTEGRATION_ANALYSIS.md)** - Current integration architecture
- **[CI/CD Metrics](docs/CI_CD_METRICS.md)** - Complete GitHub metrics catalog

### Dashboards

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 10 minutes
- **[Complete Guide](docs/README.md)** - Comprehensive documentation
- **[Jsonnet Design](docs/JSONNET_DESIGN.md)** - Architecture and patterns
- **[Modular Workflow](docs/MODULAR_WORKFLOW.md)** - Component-based approach
- **[Widget Reference](docs/DATADOG_WIDGET_REFERENCE.md)** - Available widgets
- **[Status](docs/STATUS.md)** - Implementation status

## Common Commands

```bash
# Collector management (all collectors run together every 15 minutes)
uv run python devops/datadog/scripts/run_all_collectors.py              # Run all collectors
uv run python devops/datadog/scripts/run_collector.py github --verbose  # Run individual collector
uv run python devops/datadog/scripts/run_collector.py kubernetes --push # Push individual collector

# Dashboard management
metta datadog dashboard build        # Build all dashboards from Jsonnet
metta datadog dashboard push         # Upload dashboards to Datadog
metta datadog dashboard pull         # Download dashboards (for reference)
metta datadog dashboard list         # List dashboards in Datadog
metta datadog dashboard metrics      # Discover available metrics
metta datadog dashboard diff         # Show git diff of changes
metta datadog dashboard clean        # Remove generated JSON files

# Utility
metta datadog env                    # Check environment variables
```

## Architecture Philosophy

### Data Collectors

**Modular, self-contained collectors** that scale as you add services:

1. Each collector focuses on one service (GitHub, Skypilot, WandB, etc.)
2. Common base class provides health monitoring and error handling
3. Metrics defined via decorators, automatically registered
4. Deployed as Kubernetes CronJobs via Helm
5. Credentials managed via AWS Secrets Manager

### Dashboards

**Inspired by Grafana's Grafonnet** - composable dashboard code:

1. Define reusable widget components
2. Compose dashboards by mixing and matching
3. Build JSON from Jsonnet sources
4. Push to Datadog
5. Version control the .jsonnet sources (not generated JSON)

---

**Ready to get started?**

- View metrics: `metta datadog collect github`
- Deploy dashboards: [Quick Start Guide](docs/QUICK_START.md)
- Add collectors: [Adding New Collector](docs/ADDING_NEW_COLLECTOR.md)
