# Dashboard UI Workflow: Editing in Datadog and Syncing to Version Control

This document describes the workflow for making dashboard changes in the Datadog UI and porting them back to our version-controlled jsonnet source files.

**Key Principle:** The Datadog UI is the editor, jsonnet is the source of truth for version control.

---

## Table of Contents
1. [Workflow Overview](#workflow-overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Process](#step-by-step-process)
4. [Understanding Dashboard JSON Structure](#understanding-dashboard-json-structure)
5. [Understanding Our Jsonnet System](#understanding-our-jsonnet-system)
6. [Comparing Changes](#comparing-changes)
7. [Updating Jsonnet Source](#updating-jsonnet-source)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## Workflow Overview

```
1. User edits dashboard in Datadog UI (fast, visual, immediate feedback)
   ↓
2. Export dashboard JSON using our scripts
   ↓
3. Compare exported JSON with previous version (analyze changes)
   ↓
4. Update jsonnet source files to match the new layout
   ↓
5. Generate dashboard JSON from jsonnet (verify it matches)
   ↓
6. Push to Datadog (optional, to verify)
   ↓
7. Commit jsonnet changes to git
```

**Why this workflow?**
- Datadog UI is faster for visual layout and styling
- Jsonnet is better for version control, reuse, and programmatic generation
- LLM can help bridge the gap by analyzing changes and updating jsonnet

---

## Prerequisites

### Tools Required
- Python 3.11+ with uv
- Access to Datadog API (credentials via env vars or AWS Secrets Manager)
- Git repository with jsonnet dashboard sources

### Environment Variables
```bash
# Datadog credentials (or use AWS Secrets Manager)
export DD_API_KEY="your_api_key"
export DD_APP_KEY="your_app_key"
export DD_SITE="datadoghq.com"  # optional, defaults to datadoghq.com

# For accessing WandB (if needed)
export WANDB_API_KEY="your_wandb_key"
```

### Key Scripts
All scripts are in `devops/datadog/scripts/`:
- `export_dashboard.py` - Export dashboard JSON from Datadog
- `push_dashboard.py` - Push dashboard JSON to Datadog
- `view_dashboard.py` - View live dashboard widget values (for debugging)

---

## Step-by-Step Process

### 1. Edit Dashboard in Datadog UI

1. Go to Datadog dashboard URL
2. Click "Edit" or the pencil icon
3. Make your changes:
   - Drag/drop widgets to rearrange
   - Resize widgets
   - Edit titles, queries, colors
   - Add/remove widgets
   - Change section headers
4. Save your changes in Datadog

**Do NOT:** manually edit the jsonnet files yet. Let the UI be your editor.

### 2. Export the Updated Dashboard

```bash
# Export dashboard by ID
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/export_dashboard.py <dashboard_id> \
  > /tmp/dashboard_updated.json 2>/dev/null

# Example: Export WandB dashboard
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/export_dashboard.py dr3-pdj-rrw \
  > /tmp/wandb_dashboard_updated.json 2>/dev/null
```

The exported JSON contains:
- Dashboard metadata (title, description)
- All widgets with their definitions
- Layout information (x, y, width, height)
- Queries, formulas, and styling

### 3. Compare with Previous Version

**Option A: Compare with last export**
```bash
# If you have the previous export
diff /tmp/dashboard_previous.json /tmp/dashboard_updated.json
```

**Option B: Compare with generated version from jsonnet**
```bash
# Generate current jsonnet version (if you have the jsonnet source)
jsonnet devops/datadog/dashboards/sources/wandb_metrics.jsonnet \
  > /tmp/dashboard_from_jsonnet.json

# Compare
diff /tmp/dashboard_from_jsonnet.json /tmp/dashboard_updated.json
```

**Option C: Use an LLM to analyze changes** (recommended)

See the "Comparing Changes" section below for detailed analysis techniques.

### 4. Update Jsonnet Source

This is where an LLM (like Claude Code) becomes valuable. The LLM should:

1. **Identify what changed:**
   - Widgets added/removed
   - Widget positions changed
   - Queries modified
   - Styling updated

2. **Map changes to jsonnet:**
   - Understand the jsonnet structure
   - Update widget definitions
   - Adjust layout coordinates
   - Modify queries/formulas

3. **Verify the update:**
   - Generate JSON from updated jsonnet
   - Compare with the exported dashboard
   - Ensure they match

See "Updating Jsonnet Source" section for detailed guidance.

### 5. Test the Updated Jsonnet

```bash
# Generate dashboard JSON from updated jsonnet
jsonnet devops/datadog/dashboards/sources/your_dashboard.jsonnet \
  > /tmp/dashboard_from_jsonnet_new.json

# Validate it matches the exported version
# (Some fields like IDs may differ - that's OK)
jq 'del(.id, .url, .created_at, .modified_at, .author_handle, .author_name, .widgets[].id)' \
  /tmp/dashboard_updated.json > /tmp/exported_normalized.json

jq 'del(.id, .url, .created_at, .modified_at, .author_handle, .author_name, .widgets[].id)' \
  /tmp/dashboard_from_jsonnet_new.json > /tmp/jsonnet_normalized.json

diff /tmp/exported_normalized.json /tmp/jsonnet_normalized.json
```

### 6. (Optional) Push to Datadog to Verify

```bash
# Push the jsonnet-generated version to Datadog
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/push_dashboard.py /tmp/dashboard_from_jsonnet_new.json

# View it in the UI to confirm it looks correct
```

### 7. Commit Changes

```bash
git add devops/datadog/dashboards/sources/your_dashboard.jsonnet
git commit -m "feat: update dashboard layout to match UI changes"
git push
```

---

## Understanding Dashboard JSON Structure

### Top-Level Structure
```json
{
  "id": "abc-123-xyz",
  "title": "Dashboard Title",
  "description": "Dashboard description",
  "layout_type": "ordered",
  "widgets": [
    // Array of widget objects
  ]
}
```

### Widget Structure

All widgets follow this pattern:
```json
{
  "id": 1234567890,  // Unique ID (auto-generated by Datadog)
  "definition": {
    "type": "query_value|timeseries|note|...",
    "title": "Widget Title",
    // Widget-specific configuration
  },
  "layout": {
    "x": 0,      // Column position (0-11)
    "y": 0,      // Row position
    "width": 6,  // Width in columns (1-12)
    "height": 4  // Height in rows
  }
}
```

### Widget Types

#### 1. Query Value (single metric)
```json
{
  "definition": {
    "type": "query_value",
    "title": "Active Runs",
    "requests": [{
      "response_format": "scalar",
      "queries": [{
        "name": "query1",
        "data_source": "metrics",
        "query": "avg:wandb.runs.active{*}",
        "aggregator": "last"
      }],
      "formulas": [{"formula": "query1"}]
    }],
    "precision": 2,
    "autoscale": true
  }
}
```

#### 2. Timeseries (line chart)
```json
{
  "definition": {
    "type": "timeseries",
    "title": "Training Throughput - 24h Trend",
    "requests": [{
      "response_format": "timeseries",
      "queries": [{
        "name": "query1",
        "data_source": "metrics",
        "query": "avg:wandb.push_to_main.overview.sps{*}"
      }],
      "formulas": [{"formula": "query1"}],
      "display_type": "line",
      "style": {
        "palette": "dog_classic",
        "line_type": "solid",
        "line_width": "normal"
      }
    }],
    "show_legend": false,
    "yaxis": {
      "include_zero": true,
      "scale": "linear"
    }
  }
}
```

#### 3. Note (text/header)
```json
{
  "definition": {
    "type": "note",
    "content": "## Section Header\nDescription text",
    "background_color": "blue",
    "font_size": "18",
    "text_align": "center",
    "show_tick": false,
    "has_padding": true
  }
}
```

### Layout Grid System

Datadog uses a 12-column grid:
- **x**: Column position (0-11)
- **y**: Row position (0+)
- **width**: Columns (1-12)
- **height**: Rows (1+)

Examples:
- Full width: `width: 12`
- Half width: `width: 6`
- Quarter width: `width: 3`

Typical layouts:
```
[12]                    - Full width
[6] [6]                 - Two half-width side-by-side
[3] [3] [3] [3]         - Four quarter-width
[4] [4] [4]             - Three third-width
```

---

## Understanding Our Jsonnet System

### Directory Structure
```
devops/datadog/
├── dashboards/
│   ├── sources/          # Jsonnet source files
│   │   ├── wandb_metrics.jsonnet
│   │   ├── github_ci.jsonnet
│   │   └── system_health.jsonnet
│   └── templates/        # JSON templates (if needed)
└── scripts/
    ├── export_dashboard.py
    ├── push_dashboard.py
    └── generate_dashboards.py  # (if you have a batch generator)
```

### Jsonnet Basics

Jsonnet is a data templating language that extends JSON with:
- **Variables**: `local my_var = "value";`
- **Functions**: `local create_widget(title, metric) = {...};`
- **Imports**: `local lib = import 'lib.libsonnet';`
- **String interpolation**: `"metric: %(name)s" % {name: "cpu"}`

### Common Patterns in Our Dashboards

#### Pattern 1: Widget Factory Functions
```jsonnet
local create_query_value(metric, title, x, y, width=3, height=2) = {
  definition: {
    type: 'query_value',
    title: title,
    requests: [{
      response_format: 'scalar',
      queries: [{
        name: 'query1',
        data_source: 'metrics',
        query: 'avg:%s{*}' % metric,
        aggregator: 'last',
      }],
      formulas: [{ formula: 'query1' }],
    }],
    autoscale: true,
    precision: 2,
  },
  layout: {
    x: x,
    y: y,
    width: width,
    height: height,
  },
};
```

#### Pattern 2: Metric Lists
```jsonnet
local metrics = {
  runs: {
    active: 'wandb.runs.active',
    completed: 'wandb.runs.completed_24h',
    failed: 'wandb.runs.failed_24h',
  },
  push_to_main: {
    sps: 'wandb.push_to_main.overview.sps',
    hearts: 'wandb.push_to_main.heart.gained',
    latency: 'wandb.push_to_main.skypilot.queue_latency_s',
    duration: 'wandb.push_to_main.avg_duration_hours',
  },
};
```

#### Pattern 3: Section Builders
```jsonnet
local build_ptm_section(start_y) = [
  // Header
  {
    definition: {
      type: 'note',
      content: '## Push-to-Main Baseline Performance\nGitHub CI runs',
      background_color: 'blue',
      font_size: '18',
      text_align: 'center',
    },
    layout: { x: 0, y: start_y, width: 12, height: 1 },
  },
  // Timeseries widgets
  create_timeseries(metrics.push_to_main.sps, 'Training Throughput (SPS)', 0, start_y + 1),
  create_timeseries(metrics.push_to_main.hearts, 'Score (Hearts Gained)', 6, start_y + 1),
  create_timeseries(metrics.push_to_main.latency, 'SkyPilot Queue Latency', 0, start_y + 5),
  create_timeseries(metrics.push_to_main.duration, 'Training Duration', 6, start_y + 5),
];
```

#### Pattern 4: Dashboard Assembly
```jsonnet
{
  title: 'WandB Training Metrics',
  description: 'Comprehensive WandB metrics dashboard',
  layout_type: 'ordered',
  widgets:
    build_overview_section(0) +
    build_ptm_section(7) +
    build_sweep_section(16),
}
```

---

## Comparing Changes

### Manual Comparison

#### 1. Extract Widget Summaries
```bash
# Create a Python script to summarize widgets
python3 << 'EOF'
import json
import sys

with open('/tmp/dashboard_updated.json') as f:
    dashboard = json.load(f)

print(f"Dashboard: {dashboard['title']}")
print(f"Total widgets: {len(dashboard['widgets'])}\n")

for i, widget in enumerate(dashboard['widgets'], 1):
    wtype = widget['definition'].get('type', 'unknown')
    title = widget['definition'].get('title', widget['definition'].get('content', '')[:50])
    layout = widget['layout']

    print(f"{i}. [{wtype}] {title}")
    print(f"   Position: x={layout['x']}, y={layout['y']}, w={layout['width']}, h={layout['height']}")

    # Show queries
    if wtype in ['timeseries', 'query_value']:
        requests = widget['definition'].get('requests', [])
        for req in requests:
            queries = req.get('queries', [])
            for q in queries:
                query = q.get('query', '')
                if query:
                    print(f"   Query: {query}")
    print()
EOF
```

#### 2. Compare Two Versions
```bash
# Generate summaries for both versions
python3 summarize_widgets.py /tmp/dashboard_before.json > /tmp/before_summary.txt
python3 summarize_widgets.py /tmp/dashboard_after.json > /tmp/after_summary.txt

# Compare
diff -u /tmp/before_summary.txt /tmp/after_summary.txt
```

### LLM-Assisted Comparison

When using an LLM (like Claude Code) to analyze changes:

**Provide this context:**
1. The exported dashboard JSON (`/tmp/dashboard_updated.json`)
2. The previous version (from jsonnet or previous export)
3. This documentation file
4. The existing jsonnet source (if available)

**Ask the LLM to:**
1. Identify all changes:
   - Widgets added/removed
   - Layout changes (position, size)
   - Query/formula changes
   - Styling changes
   - Title/text changes

2. Categorize changes:
   - Structural (add/remove widgets)
   - Layout (position/size)
   - Configuration (queries, colors, etc.)
   - Cosmetic (titles, descriptions)

3. Prioritize changes:
   - What must be updated in jsonnet
   - What can be ignored (auto-generated IDs, timestamps)
   - What needs manual review

---

## Updating Jsonnet Source

### Process for LLM (or Human)

#### Step 1: Load Context
Read these files:
1. Exported dashboard JSON
2. Current jsonnet source
3. This documentation
4. Any helper libraries (`lib.libsonnet`, etc.)

#### Step 2: Analyze Structure
Understand the jsonnet structure:
- How are widgets created? (factory functions?)
- How is layout managed? (manual coordinates or auto-calculated?)
- Are there any reusable patterns?

#### Step 3: Map Changes
For each change in the exported JSON:
1. Find the corresponding jsonnet code
2. Determine what needs to change
3. Make the update

Example: If a widget was moved from `(x=0, y=6)` to `(x=6, y=6)`:
```jsonnet
// Before
create_timeseries(metric, title, 0, 6, width=6, height=4)

// After
create_timeseries(metric, title, 6, 6, width=6, height=4)
```

Example: If a widget was removed:
```jsonnet
// Before
local widgets = [
  widget1,
  widget2,  // Remove this
  widget3,
];

// After
local widgets = [
  widget1,
  widget3,
];
```

#### Step 4: Handle Layout Changes
If widgets were rearranged, update the y-coordinates:

```jsonnet
// Before
local start_y = 10;
build_section(start_y) = [
  header(start_y),              // y=10
  widget1(start_y + 1),         // y=11
  widget2(start_y + 3),         // y=13
]

// After (widget2 removed, so subsequent sections shift up)
local start_y = 10;
build_section(start_y) = [
  header(start_y),              // y=10
  widget1(start_y + 1),         // y=11
  // widget2 removed
]

// Update next section's start_y
build_next_section(12)  // was 14, now 12
```

#### Step 5: Verify Queries
Ensure all metric queries are correct:
```jsonnet
// Check that metric names match
'avg:wandb.push_to_main.overview.sps{*}'
'avg:wandb.push_to_main.heart.gained{*}'
'avg:wandb.push_to_main.skypilot.queue_latency_s{*}'
```

#### Step 6: Test Generation
```bash
# Generate dashboard from updated jsonnet
jsonnet your_dashboard.jsonnet > /tmp/generated.json

# Compare with exported version (normalize first)
jq -S 'del(.id, .url, .created_at, .modified_at, .author_handle, .author_name, .widgets[].id)' \
  /tmp/dashboard_updated.json > /tmp/exported_norm.json

jq -S 'del(.id, .url, .created_at, .modified_at, .author_handle, .author_name, .widgets[].id)' \
  /tmp/generated.json > /tmp/generated_norm.json

diff /tmp/exported_norm.json /tmp/generated_norm.json
```

### Common Gotchas

1. **Widget IDs**: Always ignored when comparing (auto-generated by Datadog)
2. **Timestamps**: `created_at`, `modified_at` - ignore these
3. **Author info**: `author_handle`, `author_name` - auto-populated by Datadog
4. **Metadata fields**: `url` - generated by Datadog
5. **Field order**: JSON field order doesn't matter, use `jq -S` to sort
6. **Precision**: Floating-point numbers may have minor differences (0.99999 vs 1.0)

---

## Examples

### Example 1: Removing Summary Widgets

**Change in UI:**
- User removed 4 query_value widgets showing PTM metrics
- Kept only the timeseries plots

**Jsonnet Update:**
```jsonnet
// Before
local ptm_section = [
  header,
  create_query_value(metrics.ptm.success_rate, 'Success Rate', 0, y),
  create_query_value(metrics.ptm.completed, 'Completed', 3, y),
  create_query_value(metrics.ptm.failed, 'Failed', 6, y),
  create_query_value(metrics.ptm.sps, 'Latest SPS', 9, y),
  create_timeseries(metrics.ptm.sps, 'SPS Trend', 0, y+2),
  create_timeseries(metrics.ptm.hearts, 'Hearts Trend', 6, y+2),
];

// After
local ptm_section = [
  header,
  // Removed 4 query_value widgets
  create_timeseries(metrics.ptm.sps, 'SPS Trend', 0, y+1),  // y adjusted
  create_timeseries(metrics.ptm.hearts, 'Hearts Trend', 6, y+1),
];
```

### Example 2: Rearranging Widgets

**Change in UI:**
- User moved "Hearts" widget from row 2 to row 1
- Moved "Duration" widget from row 1 to row 2

**Jsonnet Update:**
```jsonnet
// Before
local widgets = [
  create_timeseries(metrics.sps, 'SPS', 0, y, 6, 4),
  create_timeseries(metrics.duration, 'Duration', 6, y, 6, 4),
  create_timeseries(metrics.latency, 'Latency', 0, y+4, 6, 4),
  create_timeseries(metrics.hearts, 'Hearts', 6, y+4, 6, 4),
];

// After
local widgets = [
  create_timeseries(metrics.sps, 'SPS', 0, y, 6, 4),
  create_timeseries(metrics.hearts, 'Hearts', 6, y, 6, 4),  // Moved up
  create_timeseries(metrics.latency, 'Latency', 0, y+4, 6, 4),
  create_timeseries(metrics.duration, 'Duration', 6, y+4, 6, 4),  // Moved down
];
```

### Example 3: Updating Widget Title

**Change in UI:**
- User renamed "Hearts Gained (agent survival)" to "Score (Hearts Gained)"

**Jsonnet Update:**
```jsonnet
// Before
create_timeseries(metrics.hearts, 'Hearts Gained (agent survival)', x, y)

// After
create_timeseries(metrics.hearts, 'Score (Hearts Gained)', x, y)
```

### Example 4: Changing Query Aggregation

**Change in UI:**
- User changed query from `avg:metric{*}` to `max:metric{*}`

**Jsonnet Update:**
```jsonnet
// Before
local create_widget(metric) = {
  definition: {
    requests: [{
      queries: [{
        query: 'avg:%s{*}' % metric,
      }],
    }],
  },
};

// After
local create_widget(metric) = {
  definition: {
    requests: [{
      queries: [{
        query: 'max:%s{*}' % metric,  // Changed aggregation
      }],
    }],
  },
};
```

---

## Troubleshooting

### Issue: Exported Dashboard Missing Widgets

**Symptom:** Exported JSON has fewer widgets than expected.

**Causes:**
- Dashboard wasn't saved in UI
- Wrong dashboard ID
- API permissions issue

**Solution:**
```bash
# Verify dashboard ID
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/fetch_dashboards.py --format=summary | grep "YourDashboard"

# Check widget count
jq '.widgets | length' /tmp/dashboard_exported.json
```

### Issue: Jsonnet Generation Fails

**Symptom:** `jsonnet` command returns errors.

**Causes:**
- Syntax error in jsonnet
- Missing import
- Undefined variable

**Solution:**
```bash
# Check jsonnet syntax
jsonnet your_dashboard.jsonnet 2>&1 | head -20

# Common fixes:
# - Missing comma: [item1 item2] → [item1, item2]
# - Missing semicolon: local x = 1 → local x = 1;
# - Undefined variable: Check all local/import statements
```

### Issue: Dashboard Looks Different After Push

**Symptom:** Dashboard in UI doesn't match expectations after push.

**Causes:**
- Layout coordinates wrong
- Missing widget configuration
- Datadog auto-formatting

**Solution:**
```bash
# Export what's in Datadog now
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/export_dashboard.py <id> > /tmp/actual.json

# Compare with what you pushed
diff /tmp/generated.json /tmp/actual.json

# Check specific widget
jq '.widgets[3]' /tmp/actual.json
```

### Issue: Metrics Not Showing Data

**Symptom:** Dashboard widgets show "No Data" after export/push.

**Causes:**
- Metric names incorrect
- Metrics not being collected
- Time range issue

**Solution:**
```bash
# Verify metrics exist in Datadog
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/list_metrics.py wandb

# Check if collector is running
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/run_collector.py wandb --json

# View live dashboard data
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/view_dashboard.py <dashboard_id>
```

---

## Quick Reference

### Essential Commands

```bash
# Export dashboard
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/export_dashboard.py <dashboard_id> \
  > /tmp/dashboard.json 2>/dev/null

# Push dashboard
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/push_dashboard.py /tmp/dashboard.json

# View dashboard (live data)
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/view_dashboard.py <dashboard_id>

# List all dashboards
PYTHONPATH=/Users/rwalters/GitHub/metta uv run python \
  devops/datadog/scripts/fetch_dashboards.py --format=summary

# Generate from jsonnet
jsonnet devops/datadog/dashboards/sources/dashboard.jsonnet \
  > /tmp/generated.json

# Normalize JSON for comparison
jq -S 'del(.id, .url, .created_at, .modified_at, .author_handle, .author_name, .widgets[].id)' \
  input.json > output_normalized.json
```

### Dashboard IDs

Common dashboard IDs in our system:
- WandB Training Metrics: `dr3-pdj-rrw`
- GitHub CI/CD: `(add yours here)`
- System Health: `(add yours here)`

---

## For LLMs: Quick Start Guide

If you're an LLM helping with this workflow, here's what you need:

### 1. Load These Files
- Exported dashboard JSON: `/tmp/dashboard_updated.json`
- Current jsonnet source: `devops/datadog/dashboards/sources/YOUR_DASHBOARD.jsonnet`
- This documentation: `devops/datadog/docs/DASHBOARD_UI_WORKFLOW.md`

### 2. Analyze Changes
```python
# Pseudo-code for analysis
exported_dashboard = load_json('/tmp/dashboard_updated.json')
jsonnet_source = read_file('devops/datadog/dashboards/sources/YOUR_DASHBOARD.jsonnet')

changes = compare_dashboards(exported_dashboard, previous_version)

for change in changes:
    print(f"- {change.type}: {change.description}")
    print(f"  Action: {change.suggested_jsonnet_update}")
```

### 3. Update Jsonnet
Focus on:
- Widget positions (x, y coordinates)
- Widget sizes (width, height)
- Widget types and queries
- Titles and text content

### 4. Verify
Generate JSON from updated jsonnet and compare:
```bash
jsonnet updated.jsonnet > /tmp/generated.json
diff <(jq -S 'del(.id, .widgets[].id)' /tmp/dashboard_updated.json) \
     <(jq -S 'del(.id, .widgets[].id)' /tmp/generated.json)
```

### 5. Common Patterns
- Removing widgets: Delete from array
- Moving widgets: Update x/y coordinates
- Renaming: Update title field
- Adding widgets: Insert into array with correct position

---

## Additional Resources

- Datadog Dashboard API: https://docs.datadoghq.com/api/latest/dashboards/
- Jsonnet Tutorial: https://jsonnet.org/learning/tutorial.html
- Our Dashboard Architecture: `devops/datadog/docs/DASHBOARD_DESIGN.md`
- Metric Collection: `devops/datadog/docs/COLLECTORS_ARCHITECTURE.md`

---

## Changelog

- 2025-10-27: Initial version created during WandB dashboard cleanup
