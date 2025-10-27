# Datadog Dashboard Management - Quick Start

Get started managing Datadog dashboards with **Jsonnet** in 10 minutes.

## What is This?

We use **Jsonnet** (like Grafana's Grafonnet) to build Datadog dashboards from reusable components. Think of it as
composable widgets instead of monolithic JSON files.

**Benefits:**

- Define widgets once, use in multiple dashboards
- Mix and match components
- Grid layouts with automatic positioning
- 10 lines of code instead of 200 lines of JSON

## Prerequisites

1. **Jsonnet** (already installed! ✅)
2. **Datadog API credentials**

## Setup (One Time)

### 1. Get API Credentials

1. Log in to Datadog
2. Go to **Organization Settings** → **API Keys**
3. Create/copy an API key
4. Go to **Organization Settings** → **Application Keys**
5. Create an Application key with `dashboards_read` and `dashboards_write` permissions

### 2. Configure Environment

All credentials are stored in AWS Secrets Manager. See [SECRETS_SETUP.md](../SECRETS_SETUP.md) for complete setup.

```bash
cd devops/datadog

# Validate AWS secrets configuration
uv run python scripts/validate_secrets.py

# For local development, you can set environment variables:
export DD_API_KEY=your_api_key
export DD_APP_KEY=your_app_key
```

The validation script will verify:

- ✓ All required secrets exist in AWS Secrets Manager
- ✓ Secrets can be retrieved successfully

## Quick Test

Let's verify Jsonnet works:

```bash
# Simple test
echo '{ name: "test", value: 1 + 2 }' | jsonnet -

# Should output:
# {
#    "name": "test",
#    "value": 3
# }
```

## The Jsonnet Workflow

### Step 1: Create Widget Library

Create reusable widget components in `components/`:

```bash
mkdir -p lib components dashboards
```

`lib/widgets.libsonnet`:

```jsonnet
{
  timeseries(title, query, options={}):: {
    definition: {
      type: 'timeseries',
      title: title,
      requests: [{
        formulas: [{ formula: 'query1' }],
        queries: [{
          data_source: 'metrics',
          name: 'query1',
          query: query,
        }],
        display_type: 'line',
      }],
    },
  },
}
```

`components/ci.libsonnet`:

```jsonnet
local widgets = import '../lib/widgets.libsonnet';

{
  testsPassingWidget()::
    widgets.timeseries(
      'Tests are passing on main',
      'avg:ci.tests_passing_on_main{source:softmax-system-health}'
    ),
}
```

### Step 2: Create Dashboard

`dashboards/test.jsonnet`:

```jsonnet
local ci = import '../components/ci.libsonnet';

{
  title: 'Test Dashboard',
  description: 'Built with Jsonnet!',
  layout_type: 'ordered',
  widgets: [
    ci.testsPassingWidget(),
  ],
}
```

### Step 3: Build and Push

```bash
# Build JSON from Jsonnet
metta datadog dashboard build

# Review generated JSON
cat templates/test.json

# Push to Datadog
metta datadog dashboard push
```

## Daily Workflow

```bash
# 1. Edit components or dashboards
vim components/ci.libsonnet
vim dashboards/my_dashboard.jsonnet

# 2. Build JSON from Jsonnet
metta datadog dashboard build

# 3. Review changes
metta datadog dashboard diff

# 4. Push to Datadog
metta datadog dashboard push

# 5. Commit source files (.jsonnet, not .json!)
git add components/ dashboards/ lib/
git commit -m "Update dashboard components"
```

## Discover Available Metrics

Find metrics to use in your widgets:

```bash
# List all metrics
metta datadog dashboard metrics

# Search for specific metrics (using CLI directly)
uv run python devops/datadog/scripts/list_metrics.py --search=cpu
uv run python devops/datadog/scripts/list_metrics.py --search=commits
```

## Example: Grid Layout

Create a 2×2 grid of widgets:

`lib/layout.libsonnet`:

```jsonnet
{
  grid(widgets, cols=2):: [
    widget {
      layout: {
        x: (i % cols) * (12 / cols),
        y: std.floor(i / cols) * 3,
        width: 12 / cols,
        height: 3,
      },
    }
    for i in std.range(0, std.length(widgets) - 1)
    for widget in [widgets[i]]
  ],
}
```

`dashboards/grid_dashboard.jsonnet`:

```jsonnet
local layout = import '../lib/layout.libsonnet';
local ci = import '../components/ci.libsonnet';

{
  title: 'Grid Dashboard',
  layout_type: 'free',  // Need 'free' for manual positioning
  widgets: layout.grid([
    ci.testsPassingWidget(),
    ci.revertsCountWidget(),
    ci.hotfixCountWidget(),
  ], cols=3),  // 3 columns
}
```

## Common Commands

```bash
metta datadog dashboard --help     # Show all available commands
metta datadog dashboard metrics    # Discover metrics
metta datadog dashboard build      # Build dashboards from Jsonnet
metta datadog dashboard diff       # Show changes
metta datadog dashboard push       # Upload to Datadog
metta datadog dashboard list       # List dashboards in Datadog
```

## Example: Mix and Match

Combine widgets from different sources:

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

## Troubleshooting

### "No .jsonnet files found"

```bash
# Create dashboard directory
mkdir -p dashboards

# Add a test dashboard
vim dashboards/test.jsonnet
```

### "jsonnet: command not found"

```bash
# Install jsonnet
brew install jsonnet
```

### "DD_API_KEY not set"

```bash
# Set environment variables for local development
export DD_API_KEY=your_api_key
export DD_APP_KEY=your_app_key

# Or ensure AWS credentials are configured
aws configure
```

## Next Steps

1. **Read the design docs**:
   - [README.md](README.md) - Comprehensive guide
   - [JSONNET_DESIGN.md](JSONNET_DESIGN.md) - Architecture details
   - [JSONNET_PROTOTYPE.md](JSONNET_PROTOTYPE.md) - Implementation guide

2. **Extract existing dashboards**:

   ```bash
   # Pull current dashboards
   metta datadog dashboard pull

   # Convert to Jsonnet components
   # (manual extraction - see existing templates/)
   ```

3. **Build widget library**:
   - Extract widgets from `templates/*.json`
   - Create component files in `components/`
   - Build grid layout helpers in `lib/`

## Tips

1. **Start small** - Create one widget component first
2. **Use grid layouts** - Easier than manual positioning
3. **Mix and match** - Combine widgets from different components
4. **Version control .jsonnet** - Not .json (those are generated)
5. **Discover metrics** - Use `metta datadog dashboard metrics` to find data sources

---

**Remember:** The workflow is **Edit .jsonnet → Build → Push → Commit**

Happy dashboard building! 🎉
