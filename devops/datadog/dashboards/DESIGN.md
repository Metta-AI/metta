# Datadog Jsonnet Template System - Design Document

## Vision

Create a comprehensive, LLM-friendly Jsonnet library that mirrors all Datadog widgets as composable, well-documented components. Eventually evolve into a CLI tool for generating and managing dashboards programmatically.

## Core Principles

### 1. Progressive Disclosure
**Simple things simple, complex things possible**

```jsonnet
// Simple - most common use case
widgets.timeseries('CPU Usage', 'avg:system.cpu{*}')

// Moderate - common customization
widgets.timeseries('CPU Usage', 'avg:system.cpu{*}', {
  displayType: 'bars',
  palette: 'warm',
})

// Complex - full control
widgets.timeseries('CPU Usage', {
  queries: [
    { query: 'avg:system.cpu{*}', name: 'cpu' },
    { query: 'avg:system.load{*}', name: 'load' },
  ],
  formulas: [
    { formula: 'cpu / 100', alias: 'CPU %' },
  ],
  yaxis: { scale: 'log', min: 0 },
  markers: [{ value: 80, display_type: 'error dashed' }],
})
```

### 2. Inline Documentation
**Self-documenting code for LLM discoverability**

Every widget function includes:
- Purpose and common use cases
- Parameter descriptions
- Example usage
- Related widgets
- Links to Datadog docs

```jsonnet
{
  /**
   * Timeseries Widget - Display metrics over time
   *
   * Most common widget for showing metric trends. Supports multiple queries,
   * formulas, and various display types (line, area, bars).
   *
   * Common use cases:
   * - CPU/memory utilization over time
   * - Request rate trends
   * - Error rate monitoring
   * - Latency percentiles
   *
   * @param title Display title for the widget
   * @param query Datadog metric query (e.g., 'avg:system.cpu{*}')
   * @param options Optional customization:
   *   - displayType: 'line' | 'area' | 'bars' (default: 'line')
   *   - palette: 'dog_classic' | 'cool' | 'warm' | 'purple' | 'orange' | 'gray'
   *   - lineType: 'solid' | 'dashed' | 'dotted'
   *   - lineWidth: 'thin' | 'normal' | 'thick'
   *   - showLegend: boolean (default: false)
   *   - yaxis: { min, max, scale: 'linear' | 'log' | 'sqrt' }
   *   - markers: Array of threshold markers
   *
   * @example Basic usage
   *   widgets.timeseries('CPU Usage', 'avg:system.cpu{*}')
   *
   * @example With styling
   *   widgets.timeseries('Memory', 'avg:system.mem.used{*}', {
   *     displayType: 'area',
   *     palette: 'purple',
   *     showLegend: true,
   *   })
   *
   * @example Multiple queries with formula
   *   widgets.timeseries('Error Rate %', {
   *     queries: [
   *       { query: 'sum:requests.error{*}', name: 'errors' },
   *       { query: 'sum:requests.total{*}', name: 'total' },
   *     ],
   *     formulas: [
   *       { formula: '(errors / total) * 100', alias: 'Error %' },
   *     ],
   *   })
   *
   * @see https://docs.datadoghq.com/dashboards/widgets/timeseries/
   * @related queryValue, toplist, change
   */
  timeseries(title, query, options={}):: { /* implementation */ },
}
```

### 3. Composability
**Widgets as building blocks**

```jsonnet
// Presets compose with customization
local cpuWidget = widgets.timeseries('CPU', 'avg:system.cpu{*}', {
  palette: 'warm',
  yaxis: { max: 100 },
});

// Domain components compose from primitives
local githubWidgets = {
  openPRs():: widgets.queryValue(
    'Open PRs',
    'avg:github.prs.open{*}',
    { customUnit: 'PRs', precision: 0 }
  ),

  prCycleTime():: widgets.timeseries(
    'PR Cycle Time',
    'avg:github.prs.cycle_time_hours{*}',
    { customUnit: 'hours', markers: [{ value: 48, display_type: 'warning dashed' }] }
  ),
};

// Layouts compose widgets
layouts.grid('My Dashboard', [
  layouts.row(0, [githubWidgets.openPRs(), cpuWidget]),
])
```

### 4. Type-Safe Enums
**Prevent common errors with constrained values**

```jsonnet
{
  // Define valid options as constants
  displayTypes:: {
    line: 'line',
    area: 'area',
    bars: 'bars',
  },

  palettes:: {
    classic: 'dog_classic',
    cool: 'cool',
    warm: 'warm',
    purple: 'purple',
    orange: 'orange',
    gray: 'gray',
    green: 'green',
    blue: 'blue',
  },

  // Use in functions with validation
  timeseries(title, query, options={})::
    local displayType = if std.objectHas(options, 'displayType')
      then options.displayType
      else self.displayTypes.line;

    assert std.member(std.objectValues(self.displayTypes), displayType)
      : 'displayType must be one of: ' + std.join(', ', std.objectValues(self.displayTypes));

    { /* widget implementation */ },
}
```

## Widget Hierarchy

### Level 1: Primitives (`lib/widgets.libsonnet`)
**Low-level, 1:1 mapping to Datadog API**

- timeseries
- queryValue
- toplist
- heatmap
- distribution
- table
- note
- group
- change
- iframe
- image
- logStream
- alertGraph
- alertValue
- checkStatus
- hostMap
- serviceMap
- slo
- monitor

### Level 2: Presets (`lib/presets.libsonnet`)
**Commonly-used configurations**

```jsonnet
{
  // Metrics presets
  cpuTimeseries(query):: widgets.timeseries('CPU Usage', query, {
    displayType: 'area',
    palette: 'warm',
    yaxis: { min: 0, max: 100 },
    customUnit: '%',
  }),

  memoryTimeseries(query):: widgets.timeseries('Memory Usage', query, {
    displayType: 'area',
    palette: 'purple',
    yaxis: { min: 0 },
    customUnit: 'bytes',
  }),

  errorRateTimeseries(errorQuery, totalQuery):: widgets.timeseries('Error Rate', {
    queries: [
      { query: errorQuery, name: 'errors' },
      { query: totalQuery, name: 'total' },
    ],
    formulas: [
      { formula: '(errors / total) * 100', alias: 'Error %' },
    ],
    markers: [
      { value: 1, display_type: 'warning dashed' },
      { value: 5, display_type: 'error dashed' },
    ],
  }),

  // Layout presets
  metricRow(y, metrics):: layouts.row(y, [
    widgets.queryValue(m.title, m.query, { precision: 0 })
    for m in metrics
  ], height=2),

  sectionHeader(y, title, description='')::
    layouts.fullWidth(y, widgets.note(
      '## ' + title + if description != '' then '\n\n' + description else '',
      { background_color: 'vivid_blue', font_size: '18' }
    ), height=1),
}
```

### Level 3: Domain Components (`components/*.libsonnet`)
**Service-specific widget factories**

```jsonnet
// components/github.libsonnet
local widgets = import '../lib/widgets.libsonnet';
local presets = import '../lib/presets.libsonnet';

{
  // Simple metric widgets
  openPRsWidget():: widgets.queryValue(
    'Open PRs',
    'avg:github.prs.open{*}',
    { customUnit: 'PRs', precision: 0 }
  ),

  mergedPRsWidget():: widgets.queryValue(
    'Merged (7d)',
    'sum:github.prs.merged_7d{*}',
    { customUnit: 'PRs', precision: 0 }
  ),

  // Timeseries with domain-specific styling
  prCycleTimeWidget():: widgets.timeseries(
    'PR Cycle Time',
    'avg:github.prs.cycle_time_hours{*}',
    {
      customUnit: 'hours',
      displayType: 'line',
      markers: [
        { value: 24, display_type: 'ok dashed', label: 'Target (24h)' },
        { value: 48, display_type: 'warning dashed', label: 'Warning (48h)' },
        { value: 72, display_type: 'error dashed', label: 'Critical (72h)' },
      ],
    }
  ),

  // Complex widgets with multiple queries
  ciHealthWidget():: widgets.table(
    'CI Health Summary',
    [
      { query: 'avg:github.ci.tests_passing{*}', alias: 'Tests Passing' },
      { query: 'avg:github.ci.duration_p90_minutes{*}', alias: 'Duration P90' },
      { query: 'sum:github.ci.failed_workflows_7d{*}', alias: 'Failed (7d)' },
    ],
    { hasSearchBar: false }
  ),

  // Preset-based widgets
  ciDurationWidget():: presets.cpuTimeseries('avg:github.ci.duration_p90_minutes{*}') + {
    definition+: {
      title: 'CI Duration P90',
      customUnit: 'minutes',
      yaxis+: { max: null },  // Override CPU preset's max
    },
  },
}
```

### Level 4: Dashboards (`sources/*.jsonnet`)
**Complete dashboard definitions**

```jsonnet
local layouts = import '../lib/layouts.libsonnet';
local github = import '../components/github.libsonnet';
local presets = import '../lib/presets.libsonnet';

layouts.grid(
  'GitHub CI/CD Dashboard',
  std.flattenArrays([
    // Key metrics row
    layouts.row(0, [
      github.openPRsWidget(),
      github.mergedPRsWidget(),
      github.activeDevelopersWidget(),
      github.testsPassingWidget(),
    ], height=2),

    // Section header
    [presets.sectionHeader(2, 'PR Metrics', 'Pull request velocity and health')],

    // PR charts
    layouts.row(3, [
      github.prCycleTimeWidget(),
      github.stalePRsWidget(),
    ], height=3),
  ]),
  {
    description: 'Development velocity and CI/CD health metrics',
  }
)
```

## File Structure

```
devops/datadog/dashboards/
├── lib/
│   ├── widgets.libsonnet      # Level 1: Primitive widgets
│   ├── presets.libsonnet      # Level 2: Common presets
│   ├── layouts.libsonnet      # Layout helpers
│   ├── constants.libsonnet    # Enums, palettes, etc.
│   └── utils.libsonnet        # Helper functions
│
├── components/
│   ├── github.libsonnet       # Level 3: GitHub widgets
│   ├── skypilot.libsonnet     # Level 3: Skypilot widgets
│   ├── ec2.libsonnet          # Level 3: EC2 widgets
│   ├── kubernetes.libsonnet   # Level 3: K8s widgets
│   ├── wandb.libsonnet        # Level 3: WandB widgets
│   ├── asana.libsonnet        # Level 3: Asana widgets
│   └── health_fom.libsonnet   # Level 3: Health FoM widgets
│
├── sources/
│   ├── github_cicd.jsonnet    # Level 4: Complete dashboards
│   ├── skypilot_jobs.jsonnet
│   ├── ec2.jsonnet
│   ├── asana.jsonnet
│   ├── kubernetes.jsonnet
│   └── wandb.jsonnet
│
├── templates/                  # Compiled JSON outputs
│   ├── github_cicd.json
│   ├── skypilot_jobs.json
│   └── ...
│
└── docs/
    ├── DESIGN.md              # This file
    ├── LAYOUTS.md             # Layout system docs
    ├── WIDGETS.md             # Widget reference
    └── COMPONENTS.md          # Component catalog
```

## Widget API Design Patterns

### Pattern 1: Simple String Query
For 90% of use cases:

```jsonnet
widgets.timeseries('CPU', 'avg:system.cpu{*}')
```

### Pattern 2: Options Object
For common customization:

```jsonnet
widgets.timeseries('CPU', 'avg:system.cpu{*}', {
  displayType: 'area',
  palette: 'warm',
  yaxis: { max: 100 },
})
```

### Pattern 3: Advanced Object
For complex multi-query scenarios:

```jsonnet
widgets.timeseries('Error Rate', {
  queries: [
    { query: 'sum:errors{*}', name: 'errors' },
    { query: 'sum:total{*}', name: 'total' },
  ],
  formulas: [
    { formula: '(errors / total) * 100' },
  ],
})
```

### Pattern 4: Builder Pattern (Future)
For even more complex scenarios:

```jsonnet
local builder = import '../lib/builder.libsonnet';

builder.timeseries('My Widget')
  .query('avg:system.cpu{*}', 'cpu')
  .query('avg:system.load{*}', 'load')
  .formula('cpu / 100')
  .yaxis(min=0, max=100)
  .marker(80, 'warning')
  .marker(90, 'error')
  .build()
```

## LLM-Friendly Features

### 1. Searchable Documentation
Every widget includes keywords for LLM discovery:

```jsonnet
/**
 * @keywords time-series, metrics, graph, chart, trend, monitoring
 * @category visualization
 * @difficulty easy
 * @commonUses cpu, memory, requests, latency, throughput
 */
```

### 2. Example Library
Extensive examples for common patterns:

```jsonnet
// examples/common_patterns.libsonnet
{
  errorRate: {
    description: 'Calculate error rate from error and total counts',
    code: |||
      widgets.timeseries('Error Rate', {
        queries: [
          { query: 'sum:requests.error{*}', name: 'errors' },
          { query: 'sum:requests.total{*}', name: 'total' },
        ],
        formulas: [{ formula: '(errors / total) * 100' }],
      })
    |||,
  },

  percentiles: {
    description: 'Show P50, P90, P99 latency on same chart',
    code: |||
      widgets.timeseries('Latency Percentiles', {
        queries: [
          { query: 'p50:latency{*}', name: 'p50' },
          { query: 'p90:latency{*}', name: 'p90' },
          { query: 'p99:latency{*}', name: 'p99' },
        ],
      })
    |||,
  },
}
```

### 3. Validation and Helpful Errors

```jsonnet
timeseries(title, query, options={})::
  // Validate required params
  assert std.isString(title) : 'title must be a string, got: ' + std.type(title);
  assert std.isString(query) || std.isObject(query)
    : 'query must be string or object, got: ' + std.type(query);

  // Validate options
  local validDisplayTypes = ['line', 'area', 'bars'];
  if std.objectHas(options, 'displayType') then
    assert std.member(validDisplayTypes, options.displayType)
      : 'displayType must be one of: ' + std.join(', ', validDisplayTypes) +
        '. Got: ' + options.displayType +
        '\nDid you mean: ' + (
          if options.displayType == 'bar' then 'bars'
          else if options.displayType == 'lines' then 'line'
          else validDisplayTypes[0]
        );
```

### 4. Smart Defaults with Override

```jsonnet
local defaults = {
  timeseries: {
    displayType: 'line',
    palette: 'dog_classic',
    showLegend: false,
    yaxis: { include_zero: true, scale: 'linear' },
  },
};

timeseries(title, query, options={})::
  local merged = defaults.timeseries + options;
  // Use merged config
```

## CLI Tool Vision (Future)

### Generate new widget
```bash
# Interactive
metta datadog widget create

# Direct
metta datadog widget create timeseries \
  --title "CPU Usage" \
  --query "avg:system.cpu{*}" \
  --output components/infrastructure.libsonnet

# From template
metta datadog widget create from-template error-rate \
  --errors-query "sum:requests.error{*}" \
  --total-query "sum:requests.total{*}"
```

### Generate dashboard
```bash
# From scratch
metta datadog dashboard create "My Dashboard"

# From template
metta datadog dashboard create from-template cicd \
  --service github

# From existing
metta datadog dashboard import abc-123-def \
  --output sources/imported.jsonnet
```

### Compile and deploy
```bash
# Compile to JSON
metta datadog compile sources/github_cicd.jsonnet

# Push to Datadog
metta datadog push sources/github_cicd.jsonnet

# Watch and auto-deploy
metta datadog watch sources/
```

## Migration Path

### Phase 1: Core Widgets (Current)
- ✅ Basic widget primitives (timeseries, queryValue, etc.)
- ✅ Layout system
- ⏳ Comprehensive inline docs
- ⏳ All widget types

### Phase 2: Presets & Components
- Domain-specific components (github, skypilot, etc.)
- Common presets library
- Example catalog
- Validation and error messages

### Phase 3: Advanced Features
- Builder pattern
- Widget inheritance/composition
- Template variables helpers
- SLO widgets
- Advanced formulas

### Phase 4: Tooling
- CLI for generation
- LSP for autocomplete
- Validator
- Dashboard differ
- Import from existing dashboards

### Phase 5: LLM Integration
- Natural language to Jsonnet
- Dashboard suggestions
- Optimization recommendations
- A/B testing dashboards

## Success Metrics

1. **Discoverability**: Can find the right widget in <30 seconds
2. **Productivity**: Create new dashboard in <10 minutes
3. **Maintainability**: Update existing dashboard without breaking
4. **Consistency**: All dashboards follow same patterns
5. **LLM-Friendliness**: LLM can generate valid dashboards 90%+ of the time

## Open Questions

1. **Versioning**: How to handle Datadog API changes?
2. **Testing**: How to test generated dashboards?
3. **Validation**: Schema validation vs runtime validation?
4. **Imports**: Best way to handle dashboard imports from Datadog?
5. **Templates**: How opinionated should presets be?

## Implementation Status

✅ **Completed**:
1. ✅ Documented all 10+ core Datadog widget types with progressive examples
2. ✅ Created `presets.libsonnet` with 20+ common patterns (infrastructure, app, business)
3. ✅ Implemented `layouts.libsonnet` with automatic positioning (grid, row, column helpers)
4. ✅ Built comprehensive documentation (2,041+ lines across 5 files)
5. ✅ Migrated all 4 production dashboards to new framework
6. ✅ Created 3 demo dashboards showcasing system capabilities
7. ✅ Added LLM-friendly inline documentation with @tags
8. ✅ Type-safe enums and progressive disclosure patterns
9. ✅ Cross-referenced documentation across all files

**Production Dashboards Using Framework**:
- `github_cicd.jsonnet` - GitHub CI/CD metrics
- `skypilot_jobs.jsonnet` - Skypilot job tracking
- `ec2.jsonnet` - AWS EC2 infrastructure
- `asana.jsonnet` - Project management


**Future Enhancements** (Nice to Have):
1. CLI tool for generating widget boilerplate
2. Schema validation helpers
3. Dashboard import/export utilities
4. Additional widget types as needed (scatter plots, funnel charts)
5. More domain-specific components (database, cache, queue metrics)

## Related Documentation

- **[README.md](README.md)** - Quick start guide and common patterns
- **[lib/WIDGETS.md](lib/WIDGETS.md)** - Widget primitives reference (431 lines)
- **[lib/PRESETS.md](lib/PRESETS.md)** - Preset patterns documentation (524 lines)
- **[lib/LAYOUTS.md](lib/LAYOUTS.md)** - Layout system guide (260 lines)
- **[templates/README.md](templates/README.md)** - Build workflow (211 lines)

---

**Key Insight**: The sweet spot is making simple things one-liners while preserving full Datadog API power for complex cases. Documentation and examples are more important than clever abstractions.
