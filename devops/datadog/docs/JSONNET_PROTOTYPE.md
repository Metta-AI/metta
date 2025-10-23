# Jsonnet Prototype - Quick Start

## Installation

```bash
# macOS
brew install jsonnet

# Or download from https://github.com/google/jsonnet
```

## Quick Test

Once installed, you can test Jsonnet works:

```bash
# Simple test
echo '{ name: "test", value: 1 + 2 }' | jsonnet -

# Should output:
# {
#    "name": "test",
#    "value": 3
# }
```

## Prototype Files to Create

### 1. Basic Widget Library

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

### 2. Component Library

`components/ci.libsonnet`:
```jsonnet
local widgets = import '../lib/widgets.libsonnet';

{
  testsPassingWidget()::
    widgets.timeseries(
      'Tests are passing on main',
      'avg:ci.tests_passing_on_main{source:softmax-system-health}'
    ),

  revertsCountWidget()::
    widgets.timeseries(
      'Number of reverts in the last 7 days',
      'avg:commits.reverts{source:softmax-system-health}'
    ),
}
```

### 3. Dashboard Definition

`dashboards/test.jsonnet`:
```jsonnet
local ci = import '../components/ci.libsonnet';

{
  title: 'Test Dashboard',
  description: 'Built with Jsonnet!',
  layout_type: 'ordered',
  widgets: [
    ci.testsPassingWidget(),
    ci.revertsCountWidget(),
  ],
}
```

### 4. Build It

```bash
cd devops/datadog
jsonnet dashboards/test.jsonnet > templates/test.json
```

## Grid Layout Example

`lib/layout.libsonnet`:
```jsonnet
{
  // Simple N×M grid
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

Usage:
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

## Next Steps

1. **Install jsonnet**: `brew install jsonnet`
2. **Create directory structure**:
   ```bash
   mkdir -p lib components dashboards
   ```
3. **Extract one widget** from existing dashboard to test
4. **Build and verify** JSON output
5. **Iterate** - add more widgets and features

## Expected Output

When you run `jsonnet dashboards/test.jsonnet`, you get valid Datadog JSON:

```json
{
  "title": "Test Dashboard",
  "description": "Built with Jsonnet!",
  "layout_type": "ordered",
  "widgets": [
    {
      "definition": {
        "type": "timeseries",
        "title": "Tests are passing on main",
        "requests": [
          {
            "formulas": [
              {
                "formula": "query1"
              }
            ],
            "queries": [
              {
                "data_source": "metrics",
                "name": "query1",
                "query": "avg:ci.tests_passing_on_main{source:softmax-system-health}"
              }
            ],
            "display_type": "line"
          }
        ]
      }
    },
    {
      "definition": {
        "type": "timeseries",
        "title": "Number of reverts in the last 7 days",
        "requests": [
          {
            "formulas": [
              {
                "formula": "query1"
              }
            ],
            "queries": [
              {
                "data_source": "metrics",
                "name": "query1",
                "query": "avg:commits.reverts{source:softmax-system-health}"
              }
            ],
            "display_type": "line"
          }
        ]
      }
    }
  ]
}
```

Perfect valid JSON that can be pushed to Datadog!

---

Ready to implement? Install jsonnet and let's build it!
