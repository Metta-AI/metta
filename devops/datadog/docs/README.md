# Datadog Dashboard Management Guide

## Overview

This guide covers managing Datadog dashboards as code using **Jsonnet** for composable, reusable components. By treating dashboards as modular components, we gain:

- **Reusability** - Define widgets once, use everywhere
- **Mix-and-match** - Combine widgets from different sources
- **Grid layouts** - Simple N×M tiling
- **Version control** - Track individual components
- **Type safety** - Jsonnet catches errors at build time

**Approach:** We use **Jsonnet** (like Grafana's Grafonnet) to build dashboards from reusable widget components, then generate JSON for Datadog's API.

## Table of Contents

- [Why Jsonnet?](#why-jsonnet)
- [Quick Start](#quick-start)
- [Workflow Overview](#workflow-overview)
- [Directory Structure](#directory-structure)
- [Creating Widgets](#creating-widgets)
- [Building Dashboards](#building-dashboards)
- [Common Operations](#common-operations)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Why Jsonnet?

We use **Jsonnet** (instead of raw JSON or Terraform) because:

✅ **Composable** - Build dashboards from reusable widget components
✅ **DRY** - Define widgets once, use in multiple dashboards
✅ **Grid Layouts** - Easy N×M tiling with automatic positioning
✅ **Industry Standard** - Same approach as Grafana (Grafonnet)
✅ **Type Safe** - Catch errors before pushing to Datadog
✅ **Readable** - 10 lines of Jsonnet vs 200 lines of JSON

**Inspiration:** Grafana's **Grafonnet** library uses Jsonnet for composable dashboards. We apply the same pattern to Datadog.

---

## Quick Start

### Prerequisites

1. **Get Datadog API credentials**
   - Log in to Datadog
   - Go to Organization Settings → API Keys (create/copy)
   - Go to Organization Settings → Application Keys (create with `dashboards_read` permission)

2. **Set up environment**
   ```bash
   cd devops/datadog

   # Copy and edit .env with your credentials
   cp .env.sample .env
   vim .env

   # Load credentials
   source ./load_env.sh
   ```

### Complete Workflow

```bash
# 1. Pull dashboards from Datadog
metta datadog dashboard pull

# 2. Edit JSON files
vim templates/softmax_system_health.json

# 3. See what changed
metta datadog dashboard diff

# 4. Push to Datadog
metta datadog dashboard push

# 5. Commit to git
git add templates/
git commit -m "Update dashboard thresholds"
```

---

## Workflow Overview

### The Jsonnet Workflow

```bash
# 1. Edit Jsonnet components (reusable widgets)
vim components/ci.libsonnet

# 2. Edit dashboard definitions (compose widgets)
vim dashboards/softmax_health.jsonnet

# 3. Build JSON from Jsonnet
metta datadog dashboard build

# 4. Push to Datadog
metta datadog dashboard push

# 5. Version control
git commit
```

### Visual Workflow

```
components/*.libsonnet (reusable widgets)
        ↓
dashboards/*.jsonnet (compose dashboards)
        ↓
metta datadog dashboard build (generate JSON with jsonnet)
        ↓
templates/*.json (generated output)
        ↓
metta datadog dashboard push (upload to Datadog)
        ↓
Datadog UI (updated!)
        ↓
git commit (version control .jsonnet files)
```

### Example: From Component to Dashboard

**Widget Component** (`components/ci.libsonnet`):
```jsonnet
testsPassingWidget()::
  widgets.timeseries(
    'Tests are passing on main',
    'avg:ci.tests_passing_on_main{}'
  )
```

**Dashboard** (`dashboards/health.jsonnet`):
```jsonnet
local ci = import '../components/ci.libsonnet';

{
  title: 'System Health',
  widgets: [ci.testsPassingWidget()],
}
```

**Result**: Reusable widget in multiple dashboards!

---

## Directory Structure

```
devops/datadog/
├── lib/                          # Jsonnet library (reusable helpers)
│   ├── widgets.libsonnet        # Widget primitives
│   ├── layout.libsonnet         # Grid layout helpers
│   └── dashboard.libsonnet      # Dashboard builders
│
├── components/                   # Reusable widget definitions
│   ├── ci.libsonnet             # CI/CD widgets
│   ├── apm.libsonnet            # APM widgets
│   └── custom.libsonnet         # Your custom widgets
│
├── dashboards/                   # Dashboard definitions (SOURCE)
│   ├── softmax_health.jsonnet   # Dashboard composition
│   ├── policy_eval.jsonnet      # Dashboard composition
│   └── custom_view.jsonnet      # Mix and match!
│
├── templates/                    # Generated JSON (OUTPUT, gitignored)
│   ├── softmax_health.json      # Built from .jsonnet
│   ├── policy_eval.json         # Built from .jsonnet
│   └── custom_view.json         # Built from .jsonnet
│
├── docs/                         # Documentation
│   ├── README.md                # This file
│   ├── QUICK_START.md           # Getting started
│   ├── JSONNET_DESIGN.md        # Jsonnet design details
│   ├── JSONNET_PROTOTYPE.md     # Implementation guide
│   ├── MODULAR_WORKFLOW.md      # Modular workflow details
│   └── STATUS.md                # Current status
│
├── scripts/                     # Python scripts
│   ├── fetch_dashboards.py      # List dashboards
│   ├── export_dashboard.py      # Export one dashboard
│   ├── batch_export.py          # Export all dashboards
│   ├── push_dashboard.py        # Push to Datadog
│   └── list_metrics.py          # Discover metrics
│
└── Config:
    ├── .env                     # API credentials (gitignored)
    ├── .env.sample              # Template for credentials
    ├── .gitignore               # Git ignore rules
    ├── Makefile                 # Build & deploy commands
    └── load_env.sh              # Load credentials
```

### Key Points

- **`lib/`** - Reusable Jsonnet library (like Grafonnet)
- **`components/`** - Your widget definitions (VERSION CONTROLLED)
- **`dashboards/`** - Dashboard compositions (VERSION CONTROLLED)
- **`templates/`** - Generated JSON (gitignored, rebuilt from .jsonnet)
- **`.env`** - Never commit this (contains secrets)

---

## Creating Widgets

Widgets are defined in `components/*.libsonnet` files as reusable functions.

### Example: CI Widget Component

`components/ci.libsonnet`:
```jsonnet
local widgets = import '../lib/widgets.libsonnet';

{
  testsPassingWidget()::
    widgets.timeseries(
      title='Tests are passing on main',
      query='avg:ci.tests_passing_on_main{source:softmax-system-health}',
      options={
        markers: [{
          label: 'Unit-test jobs should be passing on main',
          value: 'y = 1',
          display_type: 'info',
        }],
      }
    ),

  revertsCountWidget()::
    widgets.timeseries(
      title='Number of reverts in the last 7 days',
      query='avg:commits.reverts{source:softmax-system-health}'
    ),
}
```

### Discover Available Metrics

```bash
# List all metrics
metta datadog dashboard metrics

# Search for specific metrics
uv run python devops/datadog/scripts/list_metrics.py --search=cpu

# Find metrics to use in your widgets
uv run python devops/datadog/scripts/list_metrics.py --search=commits
```

---

## Building Dashboards

Dashboards are defined in `dashboards/*.jsonnet` files by composing widgets.

### Simple Dashboard

`dashboards/my_dashboard.jsonnet`:
```jsonnet
local ci = import '../components/ci.libsonnet';

{
  title: 'My Dashboard',
  description: 'CI/CD monitoring',
  layout_type: 'ordered',
  widgets: [
    ci.testsPassingWidget(),
    ci.revertsCountWidget(),
  ],
}
```

### Grid Layout Dashboard

```jsonnet
local layout = import '../lib/layout.libsonnet';
local ci = import '../components/ci.libsonnet';

{
  title: 'Grid Dashboard',
  layout_type: 'free',
  widgets: layout.grid([
    ci.testsPassingWidget(),
    ci.revertsCountWidget(),
    ci.hotfixCountWidget(),
  ], cols=3),  // 3 columns
}
```

### Mix and Match Widgets

```jsonnet
local ci = import '../components/ci.libsonnet';
local apm = import '../components/apm.libsonnet';

{
  title: 'Custom View',
  widgets: [
    ci.testsPassingWidget(),
    apm.orchestratorLatencyWidget(),
    ci.hotfixCountWidget(),
  ],
}
```

### Build and Deploy

```bash
# Build JSON from Jsonnet
metta datadog dashboard build

# Review generated JSON
cat templates/my_dashboard.json

# Push to Datadog
metta datadog dashboard push
```

---

## Common Operations

### List All Dashboards

```bash
metta datadog dashboard list
```

Shows all dashboards in your Datadog account with IDs, titles, and URLs.

### Pull All Dashboards

```bash
metta datadog dashboard pull
```

Downloads all dashboards as JSON files to `templates/`.

### Export Specific Dashboard

```bash
metta datadog dashboard export abc-123-def
```

Downloads one dashboard by ID to `templates/dashboard_abc-123-def.json`.

### Edit a Dashboard

```bash
# Option 1: Edit JSON directly
vim templates/softmax_system_health.json

# Option 2: Edit in Datadog UI, then pull
# (Make changes in UI)
metta datadog dashboard pull  # Sync changes to local JSON
```

### Review Changes

```bash
metta datadog dashboard diff
```

Shows git diff of all JSON changes.

### Push Changes to Datadog

```bash
# Dry run first (safe - no changes)
metta datadog dashboard push --dry-run

# Actually push
metta datadog dashboard push
```

### Create New Dashboard

**Recommended:** Create in Datadog UI first, then pull:

```bash
# 1. Create dashboard in Datadog UI
# 2. Pull it down
metta datadog dashboard pull

# 3. Find the new JSON file
ls -lt templates/  # Shows newest first

# 4. Commit to git
git add templates/new_dashboard.json
git commit -m "Add new dashboard"
```

### Delete Dashboard

```bash
# 1. Remove JSON file
rm templates/old_dashboard.json

# 2. Remove from Datadog UI
# (or push will recreate it if ID still matches)

# 3. Commit deletion
git rm templates/old_dashboard.json
git commit -m "Remove deprecated dashboard"
```

### Clone Dashboard

```bash
# 1. Export the dashboard you want to clone
metta datadog dashboard export abc-123-def

# 2. Copy and edit
cp templates/dashboard_abc-123-def.json templates/my_new_dashboard.json

# 3. Edit the copy
vim templates/my_new_dashboard.json
# Remove the "id" field (so it creates new dashboard)
# Change the "title" field

# 4. Push to create new dashboard
metta datadog dashboard push
```

---

## Best Practices

### 1. Pull Before Editing

Always pull latest from Datadog before making changes:

```bash
metta datadog dashboard pull
vim templates/my_dashboard.json
metta datadog dashboard push
```

### 2. Use metta datadog dashboard diff

Review changes before pushing:

```bash
metta datadog dashboard diff    # See what changed
metta datadog dashboard push    # Only if changes look good
```

### 3. Small, Focused Changes

```bash
# Good - one dashboard at a time
vim templates/dashboard1.json
metta datadog dashboard diff
metta datadog dashboard push

# Less ideal - many changes at once
vim templates/*.json
metta datadog dashboard push  # Harder to debug if something fails
```

### 4. Descriptive Commit Messages

```bash
# Good
git commit -m "Update softmax health dashboard: add hotfix threshold"

# Less helpful
git commit -m "Update dashboard"
```

### 5. Keep JSON Formatted

JSON files should be properly indented (2 spaces):

```bash
# Format JSON nicely
jq . templates/my_dashboard.json > tmp.json && mv tmp.json templates/my_dashboard.json
```

The export scripts already format JSON with 2-space indents.

### 6. Don't Edit IDs

The `"id"` field in JSON should not be changed - it identifies the dashboard in Datadog.

To create a new dashboard from existing one:
1. Copy the JSON
2. **Remove** the `"id"` field
3. Change the title
4. Push (will create new dashboard)

---

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `metta datadog dashboard --help` | Show all available commands |
| `metta datadog dashboard list` | List all dashboards in Datadog |
| `metta datadog dashboard pull` | Download all dashboards as JSON |
| `metta datadog dashboard push` | Upload all JSON dashboards to Datadog |
| `metta datadog dashboard export <id>` | Download specific dashboard |
| `metta datadog dashboard diff` | Show git diff of changes |
| `metta datadog dashboard push --dry-run` | Preview push without making changes |
| `metta datadog dashboard clean` | Remove generated JSON files |
| `metta datadog dashboard build` | Build dashboards from Jsonnet |
| `metta datadog dashboard metrics` | List available metrics |

---

## Troubleshooting

### "DD_API_KEY not set"

```bash
# Solution: Load credentials
source ./load_env.sh
```

### "No JSON files found"

```bash
# Forgot to pull first
metta datadog dashboard push
# Error: No JSON files found in templates/

# Solution:
metta datadog dashboard pull
metta datadog dashboard push
```

### Push Says "Created" Instead of "Updated"

This means the dashboard ID doesn't match any existing dashboard. Either:

1. Dashboard was deleted in Datadog (will create new one)
2. ID field is wrong/missing in JSON

Check the `"id"` field in your JSON file.

### JSON Syntax Error

```bash
# Invalid JSON
metta datadog dashboard push
# Error: Invalid JSON in my_dashboard.json

# Solution: Validate JSON
jq . templates/my_dashboard.json
# Fix the syntax error shown
```

### Changes Not Showing in Datadog

1. Check push output - did it succeed?
2. Refresh Datadog UI (hard refresh: Cmd+Shift+R)
3. Check you're looking at the right dashboard (ID matches)

---

## Advanced: Understanding Dashboard JSON

### Key Fields

```json
{
  "id": "abc-123-def",              // Dashboard ID (don't change)
  "title": "My Dashboard",          // Dashboard name
  "description": "...",             // Description
  "layout_type": "ordered",         // or "free"
  "template_variables": [],         // Filters
  "widgets": [                      // Dashboard widgets
    {
      "definition": {               // Widget configuration
        "type": "timeseries",       // Widget type
        "requests": [...]           // Queries
      },
      "layout": {                   // Position (if layout_type=free)
        "x": 0, "y": 0,
        "width": 4, "height": 2
      }
    }
  ]
}
```

### Common Widget Types

- `timeseries` - Line/area charts
- `query_value` - Single number
- `toplist` - Top N items
- `heatmap` - Heat map
- `note` - Text/markdown
- `alert_graph` - Monitor status
- `slo` - SLO widget

### Template Variables

Add filtering to dashboards:

```json
"template_variables": [
  {
    "name": "env",
    "prefix": "env",
    "default": "production"
  }
]
```

Use in queries: `avg:my.metric{$env}`

---

## Getting Help

- **Datadog API Docs**: https://docs.datadoghq.com/api/latest/dashboards/
- **Dashboard JSON Schema**: https://docs.datadoghq.com/dashboards/graphing_json/
- **Internal Team**: Check with team members who manage dashboards

---

Last updated: 2025-10-22
