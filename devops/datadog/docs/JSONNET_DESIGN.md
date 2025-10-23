# Jsonnet-Based Dashboard Templating

## Inspiration: Grafonnet for Grafana

**Grafonnet** is a Jsonnet library for Grafana that lets you build dashboards from reusable components. We can use the same approach for Datadog!

### Why Jsonnet?

✅ **Object-oriented** - Inheritance, mixins, composition
✅ **Functions** - Parameterized widget templates
✅ **Familiar** - Superset of JSON
✅ **Multi-format** - Generates JSON/YAML
✅ **Proven** - Used by Grafana, Kubernetes (libsonnet)
✅ **Simple** - No Python/JS runtime needed

## Proposed Structure

```
devops/datadog/
├── lib/                              # Jsonnet library (like Grafonnet)
│   ├── datadog.libsonnet            # Main entry point
│   ├── dashboard.libsonnet          # Dashboard helpers
│   ├── layout.libsonnet             # Grid layout system
│   └── widgets/                      # Widget library
│       ├── timeseries.libsonnet
│       ├── query_value.libsonnet
│       ├── toplist.libsonnet
│       └── note.libsonnet
│
├── components/                       # Reusable widget definitions
│   ├── ci.libsonnet                 # CI/CD widgets
│   ├── apm.libsonnet                # APM widgets
│   └── custom.libsonnet             # Your custom widgets
│
├── dashboards/                       # Dashboard definitions
│   ├── softmax_health.jsonnet       # Dashboard source
│   ├── policy_eval.jsonnet          # Dashboard source
│   └── custom_view.jsonnet          # Mix and match!
│
└── templates/                        # Generated JSON (output)
    ├── softmax_health.json          # Built from .jsonnet
    ├── policy_eval.json             # Built from .jsonnet
    └── custom_view.json             # Built from .jsonnet
```

## Example: Widget Library

### lib/widgets/timeseries.libsonnet

```jsonnet
{
  // Create a timeseries widget
  timeseries(title, query, options={})::
    {
      definition: {
        type: 'timeseries',
        title: title,
        show_legend: std.get(options, 'show_legend', false),
        requests: [{
          formulas: [{
            formula: 'query1',
          }],
          queries: [{
            data_source: 'metrics',
            name: 'query1',
            query: query,
          }],
          response_format: 'timeseries',
          display_type: 'line',
        }],
      } + if std.objectHas(options, 'markers') then {
        markers: options.markers,
      } else {},
    },
}
```

### components/ci.libsonnet

```jsonnet
local widgets = import '../lib/widgets/timeseries.libsonnet';

{
  // Reusable CI widget definitions
  testsPassingWidget()::
    widgets.timeseries(
      'Tests are passing on main',
      'avg:ci.tests_passing_on_main{source:softmax-system-health}',
      {
        markers: [{
          label: 'Unit-test jobs should be passing on main',
          value: 'y = 1',
          display_type: 'info',
        }],
      }
    ),

  revertsCountWidget()::
    widgets.timeseries(
      'Number of reverts in the last 7 days',
      'avg:commits.reverts{source:softmax-system-health}',
      {
        markers: [{
          label: "We shouldn't have to revert commits on main too often",
          value: 'y = 1',
          display_type: 'info',
        }],
      }
    ),

  hotfixCountWidget()::
    widgets.timeseries(
      'Number of hotfixes in the last 7 days',
      'avg:commits.hotfix{source:softmax-system-health}',
      {
        markers: [{
          label: "We shouldn't have to hotfix commits on main too often",
          value: 'y = 5',
          display_type: 'info',
        }],
      }
    ),
}
```

## Grid Layout System

### lib/layout.libsonnet

```jsonnet
{
  // Create N x M grid layout
  grid(rows, cols, widgets)::
    local widget_width = 12 / cols;
    local widget_height = 3;  // Standard height

    [
      widget {
        layout: {
          x: (i % cols) * widget_width,
          y: std.floor(i / cols) * widget_height,
          width: widget_width,
          height: widget_height,
        },
      }
      for i in std.range(0, std.length(widgets) - 1)
      for widget in [widgets[i]]
    ],

  // CSS-grid-like syntax (future)
  // grid_template: '1fr 2fr / 1fr 1fr 1fr'
  // areas: 'a a b'
  //        'c d b'
}
```

### lib/dashboard.libsonnet

```jsonnet
local layout = import 'layout.libsonnet';

{
  // Create dashboard with grid layout
  dashboard(title, widgets, options={})::
    {
      title: title,
      description: std.get(options, 'description', ''),
      layout_type: std.get(options, 'layout_type', 'ordered'),
      widgets: if std.objectHas(options, 'grid') then
        layout.grid(options.grid.rows, options.grid.cols, widgets)
      else
        widgets,
    },
}
```

## Example: Dashboard Definition

### dashboards/softmax_health.jsonnet

```jsonnet
local dashboard = import '../lib/dashboard.libsonnet';
local ci = import '../components/ci.libsonnet';

dashboard.dashboard(
  'Softmax System Health',
  [
    ci.testsPassingWidget(),
    ci.revertsCountWidget(),
    ci.hotfixCountWidget(),
  ],
  {
    description: 'CI/CD health metrics for Softmax',
    grid: { rows: 2, cols: 2 },  // 2x2 grid
  }
)
```

### dashboards/custom_view.jsonnet

```jsonnet
local dashboard = import '../lib/dashboard.libsonnet';
local ci = import '../components/ci.libsonnet';
local apm = import '../components/apm.libsonnet';

// Mix and match widgets!
dashboard.dashboard(
  'My Custom View',
  [
    ci.testsPassingWidget(),
    apm.orchestratorLatencyWidget(),
    ci.hotfixCountWidget(),
    apm.workerHitsWidget(),
  ],
  {
    description: 'Custom monitoring view',
    grid: { rows: 2, cols: 2 },
  }
)
```

## Workflow

### Build dashboards from Jsonnet

```bash
# Install jsonnet
brew install jsonnet

# Build a single dashboard
jsonnet dashboards/softmax_health.jsonnet > templates/softmax_health.json

# Build all dashboards
make build-dashboards

# Push to Datadog
make push
```

### Updated Makefile

```makefile
# Build dashboards from Jsonnet
build-dashboards:
	@echo "Building dashboards from Jsonnet..."
	@for file in dashboards/*.jsonnet; do \
		base=$$(basename $$file .jsonnet); \
		jsonnet $$file > templates/$$base.json; \
		echo "✓ Built templates/$$base.json"; \
	done

# Complete workflow
workflow: build-dashboards diff
	@echo "Ready to push. Run: make push"
```

## Advanced: CSS Grid-Like Syntax

For more complex layouts, we could support CSS grid syntax:

```jsonnet
local dashboard = import '../lib/dashboard.libsonnet';
local ci = import '../components/ci.libsonnet';

dashboard.dashboard(
  'Complex Layout',
  {
    tests: ci.testsPassingWidget(),
    reverts: ci.revertsCountWidget(),
    hotfix: ci.hotfixCountWidget(),
  },
  {
    grid_template_columns: '1fr 2fr 1fr',  // 3 columns with different widths
    grid_template_rows: 'auto auto',       // 2 rows
    areas: [
      'tests tests reverts',   // Row 1: tests spans 2 cols, reverts 1 col
      'hotfix hotfix hotfix',  // Row 2: hotfix spans all cols
    ],
  }
)
```

This would be implemented in `lib/layout.libsonnet` to calculate positions.

## Benefits

✅ **Reusable widgets** - Define once, use everywhere
✅ **Type safety** - Jsonnet catches errors at build time
✅ **DRY principle** - No duplicated widget definitions
✅ **Grid layouts** - Easy N×M tiling
✅ **Mix and match** - Combine widgets from different sources
✅ **Version control** - Track changes to individual components
✅ **Familiar syntax** - Just enhanced JSON

## Migration Path

### Phase 1: Basic Setup (1-2 hours)
1. Install jsonnet: `brew install jsonnet`
2. Create basic library structure
3. Extract one widget to Jsonnet
4. Build and test

### Phase 2: Widget Library (2-3 hours)
1. Extract all widgets from existing dashboards
2. Create widget library in `components/`
3. Create grid layout helper
4. Rebuild existing dashboards in Jsonnet

### Phase 3: Polish (1 hour)
1. Update Makefile with build commands
2. Document Jsonnet workflow
3. Create examples

### Total time: ~5 hours to full migration

## Comparison: Before vs After

### Before (Monolithic JSON)

```json
{
  "title": "My Dashboard",
  "widgets": [
    {
      "definition": {
        "type": "timeseries",
        "title": "Tests Passing",
        "requests": [/* 20 lines of config */],
        "markers": [/* 5 lines */]
      },
      "layout": {"x": 0, "y": 0, "width": 6, "height": 3}
    },
    {
      /* Another 25 lines for next widget */
    }
  ]
}
```

**Problems:**
- 200+ lines of JSON
- Hard to find specific widgets
- No reuse between dashboards

### After (Jsonnet Components)

```jsonnet
local dashboard = import 'lib/dashboard.libsonnet';
local ci = import 'components/ci.libsonnet';
local apm = import 'components/apm.libsonnet';

dashboard.dashboard(
  'My Dashboard',
  [
    ci.testsPassingWidget(),
    ci.revertsCountWidget(),
    apm.orchestratorLatencyWidget(),
  ],
  { grid: { rows: 1, cols: 3 } }
)
```

**Benefits:**
- 10 lines total
- Clear, readable
- Reusable components
- Easy to modify

## Decision Point

**Should we implement Jsonnet-based system?**

**Pros:**
- Industry standard (Grafonnet, Kubernetes)
- Powerful composition
- Grid layouts built-in
- Better than custom templating

**Cons:**
- New dependency (jsonnet)
- Learning curve
- Build step required

**My recommendation:** YES - Jsonnet is perfect for this. It's exactly what Grafana uses for the same problem.

Want me to build a prototype?

---

Last updated: 2025-10-22
