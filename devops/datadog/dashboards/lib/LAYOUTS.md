# Dashboard Layout Templates

Jsonnet templates for creating Datadog dashboards with consistent, maintainable layouts.

## Layout Types

### Grid Layout (Dashboard)
Widgets snap to a 12-column grid. Best for structured dashboards.

```jsonnet
local layouts = import 'layouts.libsonnet';

layouts.grid(
  'My Dashboard',
  widgets,  // Array of widgets with layout positioning
  {
    description: 'Dashboard description',
    id: 'abc-123-def',  // Optional - for updating existing dashboards
  }
)
```

### Auto Layout (Timeboard)
Automatic layout that fits your browser. Good for simple dashboards.

```jsonnet
layouts.auto(
  'My Timeboard',
  widgets,
  { description: 'Auto-fitting dashboard' }
)
```

### Free Layout (Screenboard)
Pixel-level precision on scrollable canvas. For complex, custom layouts.

```jsonnet
layouts.free(
  'My Screenboard',
  widgets,
  { description: 'Custom positioned dashboard' }
)
```

## Layout Helpers

### Row Layouts

#### Equal-width row
Distribute widgets evenly across 12 columns:

```jsonnet
layouts.row(
  y=0,           // Row position
  widgets=[      // Array of widgets
    widget1,
    widget2,
    widget3,
  ],
  height=3       // Height of all widgets
)
```

#### Custom-width row
Specify exact column widths (must sum to ≤12):

```jsonnet
layouts.rowCustom(
  y=0,
  widgets=[widget1, widget2, widget3],
  widths=[3, 6, 3],  // Columns: 3 + 6 + 3 = 12
  height=3
)
```

### Column Layout

Stack widgets vertically:

```jsonnet
layouts.column(
  x=0,           // Column X position
  startY=0,      // Starting Y position
  widgets=[      // Array of widgets
    widget1,
    widget2,
    widget3,
  ],
  width=6,       // Width of column
  height=3       // Height of each widget
)
```

### Preset Widths

#### Full width (12 columns)
```jsonnet
layouts.fullWidth(y=0, widget, height=3)
```

#### Half width (6 columns)
```jsonnet
layouts.halfWidth(y=0, widget, left=true, height=3)  // Left half
layouts.halfWidth(y=0, widget, left=false, height=3) // Right half
```

#### Third width (4 columns)
```jsonnet
layouts.thirdWidth(y=0, widget, position=0, height=3)  // position: 0, 1, or 2
```

#### Quarter width (3 columns)
```jsonnet
layouts.quarterWidth(y=0, widget, position=0, height=2)  // position: 0, 1, 2, or 3
```

### Grid Layout

Create 2D grids (e.g., 2×2, 3×3):

```jsonnet
layouts.grid2d(
  startY=0,
  widgets=[widget1, widget2, widget3, widget4],
  cols=2,        // Number of columns
  rows=2,        // Number of rows
  {
    height: 3,   // Height of each widget
    spacing: 0,  // Vertical spacing between rows
  }
)
```

### Manual Positioning

Place widget at exact coordinates:

```jsonnet
layouts.at(widget, x=0, y=0, width=3, height=2)
```

## Complete Example

```jsonnet
local layouts = import '../lib/layouts.libsonnet';
local widgets = import '../lib/widgets.libsonnet';
local github = import '../components/github.libsonnet';

layouts.grid(
  'GitHub Metrics',
  std.flattenArrays([
    // Row 1: Four key metrics
    layouts.row(0, [
      github.openPRsWidget(),
      github.mergedPRsWidget(),
      github.activeDevelopersWidget(),
      github.testsPassingWidget(),
    ], height=2),

    // Row 2: Full-width section header
    [layouts.fullWidth(2, widgets.note('## PR Metrics'), height=1)],

    // Row 3: Two timeseries charts
    layouts.row(3, [
      github.prCycleTimeWidget(),
      github.stalePRsWidget(),
    ], height=3),

    // Row 4: Custom widths - emphasis on middle chart
    layouts.rowCustom(
      6,
      [
        github.hotfixesWidget(),
        github.failedWorkflowsWidget(),
        github.revertsWidget(),
      ],
      [3, 6, 3],  // Middle widget is 2x wider
      height=3
    ),

    // Row 5: 2x2 grid
    layouts.grid2d(
      9,
      [
        github.commitsWidget(),
        github.branchesWidget(),
        github.workflowRunsWidget(),
        github.ciDurationWidget(),
      ],
      cols=2,
      rows=2,
      { height: 3 }
    ),
  ]),
  {
    description: 'Development velocity and CI/CD metrics',
  }
)
```

## Layout Grid System

The layout uses a 12-column grid:

```
|<- 12 columns (each 1 unit wide) ->|
|  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  |
|<-quarter->|       |<--half-->|       |<-quarter->|       |       |     |
|<---third---->|    |<---third---->|   |<---third---->|    |             |
|<-----------half---------->|       |<-----------half---------->|       |
|<---------------------------full width---------------------------->|
```

### Common Patterns

- **4 widgets per row**: `quarterWidth()` or `row()` with 4 widgets
- **3 widgets per row**: `thirdWidth()` or `row()` with 3 widgets
- **2 widgets per row**: `halfWidth()` or `row()` with 2 widgets
- **Section headers**: `fullWidth()` with height=1
- **Main content**: `fullWidth()` or `row()` with 2 widgets

## Template Variables

Add dashboard template variables:

```jsonnet
layouts.grid(
  'My Dashboard',
  widgets,
  {
    template_variables: [
      layouts.templateVar('env', 'environment', {
        available_values: ['prod', 'staging', 'dev'],
        default: 'prod',
      }),
      layouts.templateVar('service', 'service', {
        default: '*',
      }),
    ],
  }
)
```

## Tips

1. **Always use `std.flattenArrays()`** when combining multiple row/column/grid helpers
2. **Row height** is typically 2-3 for metrics, 3-4 for charts, 1 for headers
3. **Y coordinates** must be manually calculated based on previous widget heights
4. **Custom widths** must sum to 12 or less
5. **Test locally** with jsonnet before pushing to Datadog

## Related Documentation

- [Dashboard System Overview](../README.md) - Quick start and common patterns
- [Widget Primitives](./WIDGETS.md) - Low-level widget builders
- [Preset Widgets](./PRESETS.md) - Pre-configured common patterns
- [Design Document](../DESIGN.md) - Overall architecture and vision

## Building and Testing

```bash
# Compile to JSON
jsonnet dashboards/sources/my_dashboard.jsonnet > output.json

# Push to Datadog
./devops/datadog/scripts/push_dashboard.py output.json
```
