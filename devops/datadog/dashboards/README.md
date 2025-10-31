# Datadog Dashboard System

Comprehensive Jsonnet-based framework for creating and managing Datadog dashboards with LLM-friendly documentation.

## Quick Start

```bash
# 1. Create a new dashboard using presets
cat > sources/my_dashboard.jsonnet <<'EOF'
local layouts = import '../lib/layouts.libsonnet';
local presets = import '../lib/presets.libsonnet';

layouts.grid(
  'My Service Dashboard',
  std.flattenArrays([
    // Key metrics
    layouts.row(0, [
      presets.activeUsersGauge('Active Users', 'sum:users.active{*}'),
      presets.requestCountGauge('Requests', 'sum:requests{*}'),
      presets.errorRateGauge('Error Rate', 'avg:errors.rate{*}'),
      presets.latencyGauge('Latency', 'avg:latency{*}'),
    ], height=2),

    // Performance trends
    layouts.row(2, [
      presets.cpuTimeseries('CPU', 'avg:system.cpu{*}'),
      presets.memoryTimeseries('Memory', 'avg:system.mem{*}'),
    ], height=3),
  ]),
  { description: 'Service health monitoring' }
)
EOF

# 2. Compile to JSON
jsonnet sources/my_dashboard.jsonnet > /tmp/my_dashboard.json

# 3. Push to Datadog
./scripts/push_dashboard.py /tmp/my_dashboard.json
```

## Architecture

The dashboard system uses a 4-level hierarchy:

```
Level 1: Primitives (lib/widgets.libsonnet)
    ↓ Basic building blocks
Level 2: Presets (lib/presets.libsonnet)
    ↓ Common patterns with smart defaults
Level 3: Domain Components (components/*.libsonnet)
    ↓ Business-specific widgets
Level 4: Dashboards (sources/*.jsonnet)
    ↓ Complete dashboards
```

### Key Features

- **Progressive Disclosure**: Simple one-liners to complex configurations
- **LLM-Friendly**: Extensive inline documentation with @tags
- **Type-Safe**: Documented enums prevent common errors
- **Maintainable**: Automatic positioning, no manual coordinates
- **Composable**: Mix and match components freely

## Directory Structure

```
devops/datadog/dashboards/
├── README.md                    # This file - overview and quick start
├── DESIGN.md                    # Architecture and design philosophy
│
├── lib/                         # Core libraries
│   ├── widgets.libsonnet        # Primitive widget builders
│   ├── presets.libsonnet        # Pre-configured common patterns
│   ├── layouts.libsonnet        # Layout and positioning helpers
│   ├── WIDGETS.md               # Widget primitives documentation
│   ├── PRESETS.md               # Presets documentation
│   └── LAYOUTS.md               # Layout system documentation
│
├── components/                  # Domain-specific components
│   ├── github.libsonnet         # GitHub metrics widgets
│   ├── skypilot.libsonnet       # Skypilot job metrics
│   ├── ec2.libsonnet            # AWS EC2 metrics
│   ├── asana.libsonnet          # Asana project metrics
│   ├── ci.libsonnet             # CI/CD metrics
│   ├── infrastructure.libsonnet # Infrastructure metrics
│   └── apm.libsonnet            # APM/tracing metrics
│
├── sources/                     # Dashboard source files (Jsonnet)
│   ├── github_cicd.jsonnet      # GitHub CI/CD dashboard
│   ├── skypilot_jobs.jsonnet    # Skypilot jobs dashboard
│   ├── ec2.jsonnet              # EC2 infrastructure dashboard
│   ├── asana.jsonnet            # Asana project management
│   ├── kubernetes.jsonnet       # Kubernetes metrics dashboard
│   └── wandb.jsonnet            # Weights & Biases metrics
│
└── templates/                   # Generated JSON (not committed)
    ├── README.md                # Build workflow documentation
    └── *.json                   # Compiled dashboards (gitignored)
```

## Documentation Index

### Getting Started
- **[This README](README.md)** - Overview and quick start
- **[templates/README.md](templates/README.md)** - Build and deployment workflow

### Core Concepts
- **[DESIGN.md](DESIGN.md)** - Architecture, design philosophy, and vision (615 lines)
- **[lib/LAYOUTS.md](lib/LAYOUTS.md)** - Layout system and positioning (260 lines)
- **[lib/WIDGETS.md](lib/WIDGETS.md)** - Widget primitive reference (431 lines)
- **[lib/PRESETS.md](lib/PRESETS.md)** - Pre-configured patterns (524 lines)

### Total Documentation
- **2,041+ lines** of comprehensive documentation
- **LLM-optimized** with searchable @tags and keywords
- **Progressive examples** from simple to advanced
- **Cross-referenced** across all docs

## Common Patterns

### Pattern 1: Simple Dashboard with Presets

```jsonnet
local layouts = import '../lib/layouts.libsonnet';
local presets = import '../lib/presets.libsonnet';

layouts.grid(
  'Service Health',
  std.flattenArrays([
    // Metrics row
    layouts.row(0, [
      presets.requestCountGauge('Requests', 'sum:requests{*}'),
      presets.errorRateGauge('Errors', 'avg:errors.rate{*}'),
      presets.latencyGauge('Latency', 'avg:latency{*}'),
    ], height=2),

    // Charts row
    layouts.row(2, [
      presets.requestRateTimeseries('Request Rate', 'sum:requests{*}'),
      presets.latencyTimeseries('Response Time', 'avg:latency{*}'),
    ], height=3),
  ]),
  { description: 'Service monitoring' }
)
```

### Pattern 2: Custom Widgets with Primitives

```jsonnet
local layouts = import '../lib/layouts.libsonnet';
local widgets = import '../lib/widgets.libsonnet';

layouts.grid(
  'Custom Dashboard',
  std.flattenArrays([
    layouts.row(0, [
      widgets.queryValue('Custom Metric', 'avg:my.metric{*}', {
        precision: 1,
        custom_unit: 'ms',
        aggregator: 'last',
      }),
      widgets.timeseries('Trend', 'avg:my.metric{*}', {
        display_type: 'area',
        palette: 'warm',
        markers: [
          { label: 'SLO', value: 'y = 100', display_type: 'ok dashed' },
        ],
      }),
    ], height=3),
  ]),
  { description: 'Custom monitoring' }
)
```

### Pattern 3: Domain Components

```jsonnet
local layouts = import '../lib/layouts.libsonnet';
local github = import '../components/github.libsonnet';
local presets = import '../lib/presets.libsonnet';

layouts.grid(
  'GitHub Metrics',
  std.flattenArrays([
    // Section header
    [layouts.fullWidth(0, presets.sectionHeader(
      'Pull Requests',
      'PR velocity and cycle time'
    ), height=1)],

    // PR metrics
    layouts.row(1, [
      github.openPRsWidget(),
      github.mergedPRsWidget(),
      github.prCycleTimeWidget(),
    ], height=2),
  ]),
  { description: 'GitHub development metrics' }
)
```

### Pattern 4: Grouped Sections

```jsonnet
local layouts = import '../lib/layouts.libsonnet';
local widgets = import '../lib/widgets.libsonnet';
local ec2 = import '../components/ec2.libsonnet';

layouts.grid(
  'Infrastructure',
  [
    widgets.group(
      'EC2 Instances',
      [
        ec2.totalInstancesWidget(),
        ec2.runningInstancesWidget(),
        ec2.stoppedInstancesWidget(),
      ],
      { background_color: 'vivid_blue' }
    ),

    widgets.group(
      'Cost Tracking',
      [
        ec2.hourlyCostWidget(),
        ec2.monthlyCostWidget(),
        ec2.spotSavingsWidget(),
      ],
      { background_color: 'vivid_purple' }
    ),
  ],
  { description: 'Infrastructure monitoring' }
)
```

## Widget Hierarchy Quick Reference

### Level 1: Primitives (`lib/widgets.libsonnet`)
Basic building blocks with full Datadog API coverage:
- `timeseries()` - Line/bar/area charts
- `queryValue()` - Single number displays
- `toplist()` - Ranked lists
- `note()` - Text/markdown
- `heatmap()` - Density plots
- `change()` - Period-over-period comparison
- `distribution()` - Histograms (APM)
- `table()` - Multi-column data
- `group()` - Widget containers

### Level 2: Presets (`lib/presets.libsonnet`)
Pre-configured patterns with smart defaults:
- Infrastructure: `cpuTimeseries()`, `memoryTimeseries()`, `diskUsageGauge()`
- Application: `errorRateTimeseries()`, `latencyTimeseries()`, `requestRateTimeseries()`
- Business: `activeUsersGauge()`, `conversionRateGauge()`, `revenueTimeseries()`
- Headers: `sectionHeader()`, `subsectionHeader()`, `alertNote()`, `infoNote()`
- Tables: `serviceHealthTable()`, `hostHealthTable()`

### Level 3: Domain Components (`components/*.libsonnet`)
Business-specific metrics:
- `github.*` - PRs, commits, CI/CD, developers
- `skypilot.*` - Jobs, GPUs, clusters, runtime
- `ec2.*` - Instances, volumes, costs
- `asana.*` - Projects, bugs, velocity
- `infrastructure.*` - Containers, pods, nodes
- `ci.*` - Tests, reverts, hotfixes
- `apm.*` - Traces, spans, latency

### Layout Helpers (`lib/layouts.libsonnet`)
Automatic positioning and organization:
- `layouts.grid()` - 12-column grid dashboard
- `layouts.row()` - Equal-width widget rows
- `layouts.rowCustom()` - Custom column widths
- `layouts.fullWidth()` - Full-width widgets (12 cols)
- `layouts.halfWidth()` - Half-width widgets (6 cols)
- `layouts.grid2d()` - 2D grids (e.g., 2×2, 3×3)

## LLM Integration

This system is designed to be LLM-friendly:

### Inline Documentation
Every widget includes:
- `@widget` - Widget type name
- `@purpose` - What it's used for
- `@use_cases` - Common scenarios
- `@options` - Available customizations
- `@enum` - Valid values for options
- `@example_simple` - Basic usage
- `@example_moderate` - Common customization
- `@example_advanced` - Full control
- `@related` - Similar widgets
- `@docs` - Datadog documentation link

### Search Keywords
Natural language queries work well:
- "timeseries chart line graph" → `timeseries()`
- "single number metric gauge" → `queryValue()`
- "section header divider" → `presets.sectionHeader()`
- "error rate with SLO" → `presets.errorRateTimeseries()`

### Example LLM Prompts
```
"Create a dashboard with 4 key metrics and 2 charts below"
"Show me CPU usage with warning thresholds"
"Make a section header for database metrics"
"Add a table comparing service health metrics"
```

## Development Workflow

### 1. Write Dashboard
```bash
# Edit Jsonnet source
$EDITOR sources/my_dashboard.jsonnet
```

### 2. Test Locally
```bash
# Compile and validate
jsonnet sources/my_dashboard.jsonnet > /tmp/my_dashboard.json
jq . /tmp/my_dashboard.json > /dev/null && echo "Valid JSON"
```

### 3. Deploy to Datadog
```bash
# Push to Datadog
./scripts/push_dashboard.py /tmp/my_dashboard.json
```

### 4. Iterate
```bash
# Update source
$EDITOR sources/my_dashboard.jsonnet

# Recompile and push
jsonnet sources/my_dashboard.jsonnet > /tmp/my_dashboard.json
./scripts/push_dashboard.py /tmp/my_dashboard.json
```

## Best Practices

### 1. Use Presets for Common Patterns
```jsonnet
// Good - uses preset with smart defaults
presets.cpuTimeseries('CPU Usage', 'avg:system.cpu{*}')

// Avoid - manually configuring everything
widgets.timeseries('CPU Usage', 'avg:system.cpu{*}', {
  display_type: 'area',
  palette: 'warm',
  markers: [{ label: 'Warning', value: 'y = 80', ... }]
})
```

### 2. Use Layout Helpers
```jsonnet
// Good - automatic positioning
layouts.row(0, [widget1, widget2, widget3], height=2)

// Avoid - manual coordinates
widget1 + { layout: { x: 0, y: 0, width: 4, height: 2 } },
widget2 + { layout: { x: 4, y: 0, width: 4, height: 2 } },
widget3 + { layout: { x: 8, y: 0, width: 4, height: 2 } },
```

### 3. Group Related Widgets
```jsonnet
// Good - logical grouping
widgets.group('Database Metrics', dbWidgets, {
  background_color: 'vivid_blue'
})
```

### 4. Use Section Headers
```jsonnet
// Good - clear organization
[layouts.fullWidth(0, presets.sectionHeader(
  'Performance Metrics',
  'Response times and throughput'
), height=1)]
```

### 5. Flatten Arrays
```jsonnet
// Required when combining layouts
std.flattenArrays([
  layouts.row(0, [...]),
  [layouts.fullWidth(2, ...)],
  layouts.row(3, [...]),
])
```

## Troubleshooting

### Dashboard won't compile
```bash
# Check Jsonnet syntax
jsonnet sources/my_dashboard.jsonnet

# Common issues:
# - Missing imports
# - Incorrect function arguments
# - Forgetting std.flattenArrays()
```

### Dashboard pushes but looks wrong
```bash
# Validate JSON structure
jq . /tmp/my_dashboard.json

# Check widget count
jq '.widgets | length' /tmp/my_dashboard.json

# Verify layout coordinates
jq '.widgets[0].layout' /tmp/my_dashboard.json
```

### Widgets overlap or misaligned
```bash
# Check Y coordinates are sequential
jq '[.widgets[].layout.y] | unique | sort' /tmp/my_dashboard.json

# Verify row heights match spacing
# If row at y=0 has height=2, next row should be y=2
```

## Examples

See working production dashboards in `sources/`:
- **github_cicd.jsonnet** - GitHub CI/CD metrics
- **skypilot_jobs.jsonnet** - Skypilot job monitoring
- **ec2.jsonnet** - AWS EC2 infrastructure with groups
- **asana.jsonnet** - Asana project management with groups
- **kubernetes.jsonnet** - Kubernetes cluster metrics
- **wandb.jsonnet** - Weights & Biases training metrics

## Contributing

### Adding New Widgets
1. Add primitive to `lib/widgets.libsonnet` with full documentation
2. Create preset in `lib/presets.libsonnet` if it's a common pattern
3. Update `lib/WIDGETS.md` or `lib/PRESETS.md`
4. Add example to demo dashboard

### Adding New Components
1. Create `components/my_service.libsonnet`
2. Import primitives or presets
3. Export specific widget functions
4. Create example dashboard in `sources/`

### Documentation Standards
- Use `@tags` for LLM discoverability
- Provide simple → moderate → advanced examples
- Document all enums with valid values
- Link to Datadog docs where applicable
- Keep examples copy-paste ready

## Resources

- **[Datadog Dashboard API](https://docs.datadoghq.com/api/latest/dashboards/)**
- **[Datadog Widget Types](https://docs.datadoghq.com/dashboards/widgets/)**
- **[Jsonnet Language](https://jsonnet.org/)**
- **[Jsonnet Tutorial](https://jsonnet.org/learning/tutorial.html)**

## License

MIT License - See repository root for details
