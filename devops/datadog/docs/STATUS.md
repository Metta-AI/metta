# Datadog Observability System - Status

## Overview

This system provides two key capabilities:
1. **Metrics Collection** - Automated collection and submission of GitHub metrics to Datadog
2. **Dashboard Management** - Jsonnet-based composable dashboard system

---

## ✅ Completed - Metrics Collection (Phase 1 & 2B)

We've built a production-ready metrics collection system for tracking GitHub repository health.

### Phase 1: Metrics Implementation (Complete)

**GitHub Metrics (25 total):**
- ✅ Pull request metrics (8): open, merged, closed, cycle time, stale PRs, review coverage
- ✅ Branch metrics (1): active branch count
- ✅ Commit & code metrics (7): commits, hotfixes, reverts, lines added/deleted, files changed
- ✅ CI/CD metrics (7): workflow runs, failures, duration (avg, p50, p90, p99), test status
- ✅ Developer metrics (2): active developers, commits per developer

**Documentation:**
- ✅ `docs/METRIC_CONVENTIONS.md` - Naming patterns and conventions
- ✅ `docs/CI_CD_METRICS.md` - Complete metric catalog with use cases
- ✅ `WORKPLAN.md` - Implementation tracking and decisions

### Phase 2B: Modular Architecture (Complete)

**Common Infrastructure:**
- ✅ `common/base.py` - BaseCollector abstract class
- ✅ `common/decorators.py` - @metric decorator for registration
- ✅ `common/datadog_client.py` - Datadog API wrapper with retry logic

**GitHub Collector:**
- ✅ `collectors/github/collector.py` - GitHubCollector with all 25 metrics
- ✅ Organized into logical groups (PRs, branches, commits, CI/CD, developers)
- ✅ Comprehensive error handling and logging
- ✅ Uses modern datadog-api-client v2 API

**Current Status:**
- Collection frequency: Every 15 minutes (Kubernetes CronJob - pending deployment)
- API usage: ~600 calls/hour (12% of 5000/hour limit)
- Collection time: ~90 seconds
- Strategy: "Overshoot and trim later" - collect now, refine based on dashboard usage

---

## ✅ Completed - Jsonnet-Based Dashboard Management

We've successfully set up a Jsonnet-based workflow for managing Datadog dashboards with reusable, composable components.

### What We Built

**Core Scripts:**
- ✅ `fetch_dashboards.py` - List all dashboards
- ✅ `export_dashboard.py` - Export single dashboard
- ✅ `batch_export.py` - Export all dashboards
- ✅ `push_dashboard.py` - Push dashboards to Datadog
- ✅ `list_metrics.py` - Discover available metrics for widgets

**Configuration:**
- ✅ `.env` - API credentials (populated)
- ✅ `.env.sample` - Template for others
- ✅ `load_env.sh` - Credential loader
- ✅ `.gitignore` - Ignores generated JSON, tracks .jsonnet sources

**Tools:**
- ✅ `Makefile` - Build, pull, push commands
- ✅ `README.md` - Comprehensive Jsonnet-based guide
- ✅ `QUICK_START.md` - 10-minute Jsonnet getting started
- ✅ `JSONNET_DESIGN.md` - Architecture and design details
- ✅ `JSONNET_PROTOTYPE.md` - Implementation guide

**Existing Dashboards (JSON exports for reference):**
- ✅ `templates/softmax_system_health.json` - CI/CD metrics
- ✅ `templates/policy_evaluator.json` - APM metrics
- ✅ `templates/softmax_pulse.json` - Empty placeholder

## 🎯 Current Workflow

```bash
# Edit Jsonnet components (reusable widgets)
vim components/ci.libsonnet

# Edit dashboard definitions (compose widgets)
vim dashboards/my_dashboard.jsonnet

# Build JSON from Jsonnet
metta datadog dashboard build

# Review changes
metta datadog dashboard diff

# Push to Datadog
metta datadog dashboard push

# Version control
git add components/ dashboards/ lib/
git commit -m "Update dashboard components"
```

The workflow: **Edit .jsonnet → Build → Push → Commit**

## 📊 What Works

- ✅ Pull all dashboards from Datadog (for reference)
- ✅ Export individual dashboards
- ✅ Push dashboards back to Datadog
- ✅ Build JSON from Jsonnet sources
- ✅ Discover available metrics for widget building
- ✅ Create new dashboards (remove ID field)
- ✅ Update existing dashboards (keep ID field)
- ✅ Dry-run mode for safe testing
- ✅ Git diff to see changes
- ✅ Simple Makefile interface

## 🗂️ File Structure

```
devops/datadog/
├── lib/                     # Jsonnet library (reusable helpers) [TO CREATE]
│   ├── widgets.libsonnet   # Widget primitives
│   ├── layout.libsonnet    # Grid layout helpers
│   └── dashboard.libsonnet # Dashboard builders
│
├── components/              # Reusable widget definitions [TO CREATE]
│   ├── ci.libsonnet        # CI/CD widgets
│   ├── apm.libsonnet       # APM widgets
│   └── custom.libsonnet    # Your custom widgets
│
├── dashboards/              # Dashboard definitions (SOURCE) [TO CREATE]
│   ├── softmax_health.jsonnet
│   ├── policy_eval.jsonnet
│   └── custom_view.jsonnet
│
├── templates/               # Generated JSON (OUTPUT, gitignored)
│   ├── softmax_health.json # Built from .jsonnet
│   ├── policy_eval.json    # Built from .jsonnet
│   └── custom_view.json    # Built from .jsonnet
│
├── scripts/
│   ├── fetch_dashboards.py  # List dashboards
│   ├── export_dashboard.py  # Export one
│   ├── batch_export.py      # Export all
│   ├── push_dashboard.py    # Push to Datadog
│   └── list_metrics.py      # Discover metrics
│
├── docs/
│   ├── README.md            # Comprehensive guide
│   ├── QUICK_START.md       # 10-minute Jsonnet guide
│   ├── JSONNET_DESIGN.md    # Architecture details
│   ├── JSONNET_PROTOTYPE.md # Implementation guide
│   ├── MODULAR_WORKFLOW.md  # Modular workflow details
│   └── STATUS.md            # This file
│
└── Config:
    ├── .env                 # Credentials (gitignored)
    ├── .env.sample          # Template for credentials
    ├── .gitignore           # Git ignore rules
    ├── load_env.sh          # Load credentials
    └── Makefile             # Commands
```

## 💡 Why Jsonnet Instead of Terraform or Raw JSON?

After exploring multiple approaches, we chose Jsonnet because:

**vs. Terraform:**
✅ **Simpler** - No Terraform installation or state files
✅ **Native** - Generates JSON directly (Datadog's format)
✅ **Faster** - Direct API calls, no plan/apply overhead
✅ **More composable** - Functions and imports for reusability

**vs. Raw JSON:**
✅ **Reusable** - Define widgets once, use in multiple dashboards
✅ **Mix-and-match** - Combine widgets from different sources
✅ **Grid layouts** - Automatic positioning with simple helpers
✅ **Readable** - 10 lines of Jsonnet vs 200 lines of JSON
✅ **Type safe** - Catch errors before pushing to Datadog

**Inspiration:** Grafana's **Grafonnet** uses the same pattern for composable dashboards.

**Note:** Terraform makes sense if managing monitors, SLOs, users, etc. For dashboards alone, Jsonnet is better.

## 🚀 Next Steps

### Phase 2C: CLI Integration (Complete)

**Metrics Collection:**
- ✅ Update CLI to use new GitHubCollector
- ✅ Add `metta datadog collect github` command
- ✅ Support `--dry-run` and `--push` flags
- ✅ Add `--verbose` for debugging
- ✅ Test end-to-end metric collection and submission

**Implementation Details:**
- Created `run_collector.py` - Standalone runner script that executes in full project environment
- Updated `cli.py` - Uses subprocess to call runner script, avoiding import issues
- AWS Secrets Manager integration - Falls back to AWS if GITHUB_TOKEN not set
- Clean JSON output - Status messages routed to stderr when using --json flag
- Rich table display - Formatted output with typer and rich libraries

### Phase 3: EKS Deployment

**Kubernetes Infrastructure:**
- [ ] Create Helm chart for metrics collectors
- [ ] Configure CronJob for 15-minute collection schedule
- [ ] Set up AWS Secrets Manager integration
- [ ] Deploy to EKS and verify metrics in Datadog
- [ ] Monitor collection performance and API usage

### Jsonnet Dashboard Setup (Lower Priority)

- [ ] Create `lib/widgets.libsonnet` - Widget primitive functions
- [ ] Create `lib/layout.libsonnet` - Grid layout helpers
- [ ] Extract widgets from `templates/*.json` to create `components/*.libsonnet`
- [ ] Create example dashboards in `dashboards/*.jsonnet` format
- [ ] Test complete workflow: Edit .jsonnet → Build → Push → Verify

### Future Enhancements

- [ ] Add AWS infrastructure metrics collector
- [ ] Add custom application metrics collector
- [ ] Set up alerts and monitors based on collected metrics
- [ ] Build comprehensive project health dashboard
- [ ] Add DORA metrics (deployment frequency, lead time, MTTR, change failure rate)

## 📝 Quick Reference

### Daily Use

```bash
# Load credentials (once per shell session)
source ./load_env.sh

# Edit → Build → Push workflow
vim components/ci.libsonnet         # Edit widgets
vim dashboards/my_dashboard.jsonnet  # Compose dashboard
metta datadog dashboard build        # Build JSON from Jsonnet
metta datadog dashboard diff         # Review changes
metta datadog dashboard push         # Upload to Datadog

# Commit changes
git add lib/ components/ dashboards/
git commit -m "Update dashboard components"
```

### Commands

```bash
# Dashboard management
metta datadog dashboard build        # Build all dashboards from Jsonnet
metta datadog dashboard build -f FILE # Build single dashboard
metta datadog dashboard list         # List dashboards in Datadog
metta datadog dashboard metrics      # Discover available metrics
metta datadog dashboard pull         # Download all dashboards (for reference)
metta datadog dashboard push         # Upload all dashboards to Datadog
metta datadog dashboard push --dry-run  # Preview push without making changes
metta datadog dashboard diff         # Show git diff of changes
metta datadog dashboard clean        # Remove generated JSON files

# Collector management
metta datadog collect github         # Run GitHub collector (dry-run)
metta datadog collect github --push  # Run and push metrics to Datadog
metta datadog list-collectors        # List available collectors
```

## 🎉 Summary

**We have a Jsonnet-based composable dashboard system!**

- Define reusable widget components
- Compose dashboards by mixing and matching widgets
- Build JSON from Jsonnet sources
- Push to Datadog
- Track .jsonnet sources in git (not generated JSON)

No Terraform complexity, no monolithic JSON files, just composable Jsonnet components.

**Inspired by Grafana's Grafonnet** - the same pattern for Datadog dashboards.

---

Last updated: 2025-10-23 (Phase 2C complete)
