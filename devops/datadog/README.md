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
- **GitHub**: PRs, commits, branches, CI/CD workflows, developer activity
- **Skypilot** *(planned)*: Jobs, clusters, compute costs
- **WandB** *(planned)*: Training runs, experiments
- **EC2** *(planned)*: Instances, costs, utilization
- **Asana** *(planned)*: Tasks, projects, velocity

Each collector:
- Runs on schedule (5-30 minute intervals)
- Uses AWS Secrets Manager for credentials
- Emits health metrics about itself
- Handles errors gracefully

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

### Datadog API Keys

For dashboard management, set up Datadog credentials:
```bash
cd devops/datadog
cp .env.sample .env
# Edit .env and add:
#   DD_API_KEY=your-api-key
#   DD_APP_KEY=your-app-key
#   DD_SITE=datadoghq.com
source ./load_env.sh
```

## Quick Start

### View Collected Metrics

```bash
# See all currently collected metrics
metta softmax-system-health report

# Output shows 17+ metrics across categories:
# - Pull Requests (open, merged, time to merge)
# - Branches (active branches)
# - Commits (total, per developer, hotfixes)
# - CI/CD (workflow runs, failures, duration)
# - Developers (active count, productivity)
```

### Deploy a Dashboard

```bash
# Setup (one time)
cd devops/datadog
cp .env.sample .env
# Edit .env with your Datadog API credentials
source ./load_env.sh

# Daily workflow
vim components/ci.libsonnet         # Edit widget components
vim dashboards/my_dashboard.jsonnet # Compose dashboard
make build                          # Build JSON from Jsonnet
make diff                           # Review changes
make push                           # Upload to Datadog
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
│   └── asana/         # Asana tasks and project velocity
│
├── common/             # Shared utilities
│   ├── datadog_client.py  # Datadog metric submission
│   ├── secrets.py         # AWS Secrets Manager access
│   ├── registry.py        # Metric registry and auto-discovery
│   └── decorators.py      # @metric decorator
│
├── lib/                # Jsonnet library (widget builders)
├── components/         # Reusable widget collections
├── dashboards/         # Dashboard definitions (SOURCE)
├── templates/          # Generated JSON (OUTPUT, gitignored)
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
make help           # Show all available commands
make list           # List dashboards in Datadog
make list-metrics   # Discover available metrics
make build          # Build all dashboards from Jsonnet
make push           # Upload dashboards to Datadog
make pull           # Download dashboards (for reference)
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
- View metrics: `metta softmax-system-health report`
- Deploy dashboards: [Quick Start Guide](docs/QUICK_START.md)
- Add collectors: [Adding New Collector](docs/ADDING_NEW_COLLECTOR.md)
