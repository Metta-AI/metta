# Datadog Dashboard Management - Status

## ✅ Completed - Jsonnet-Based Approach

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
make build

# Review changes
make diff

# Push to Datadog
make push

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

### Immediate (To Complete Jsonnet Setup)

- [ ] Create `lib/widgets.libsonnet` - Widget primitive functions
- [ ] Create `lib/layout.libsonnet` - Grid layout helpers
- [ ] Create `lib/dashboard.libsonnet` - Dashboard builder functions
- [ ] Extract widgets from `templates/*.json` to create `components/*.libsonnet`
- [ ] Create example dashboards in `dashboards/*.jsonnet` format
- [ ] Test complete workflow: Edit .jsonnet → Build → Push → Verify

### Optional Enhancements

- [ ] Set up CI/CD to validate Jsonnet before merge
- [ ] Add pre-commit hook to validate Jsonnet compilation
- [ ] Document team widget library conventions
- [ ] Create widget templates for common patterns
- [ ] Build more sophisticated layout helpers (beyond simple grid)

## 📝 Quick Reference

### Daily Use

```bash
# Load credentials (once per shell session)
source ./load_env.sh

# Edit → Build → Push workflow
vim components/ci.libsonnet        # Edit widgets
vim dashboards/my_dashboard.jsonnet # Compose dashboard
make build                         # Build JSON from Jsonnet
make diff                          # Review changes
make push                          # Upload to Datadog

# Commit changes
git add lib/ components/ dashboards/
git commit -m "Update dashboard components"
```

### Commands

```bash
make help           # Show all commands
make list           # List dashboards in Datadog
make list-metrics   # Discover available metrics
make build          # Build all dashboards from Jsonnet
make build-one      # Build single dashboard
make pull           # Download all dashboards (for reference)
make push           # Upload all dashboards to Datadog
make diff           # Show git diff of changes
make dry-run        # Preview push without making changes
make clean          # Remove generated JSON files
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

Last updated: 2025-10-22
