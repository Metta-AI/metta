# Datadog Widget Library

Comprehensive, LLM-friendly Jsonnet library for creating Datadog dashboard widgets.

## Philosophy

**Progressive Disclosure**: Simple things are one-liners, complex things are possible.

```jsonnet
// Simple - most common use case
widgets.timeseries('CPU Usage', 'avg:system.cpu{*}')

// Moderate - common customization
widgets.timeseries('CPU Usage', 'avg:system.cpu{*}', {
  display_type: 'area',
  palette: 'warm',
})

// Advanced - full control
widgets.timeseries('CPU Usage', 'avg:system.cpu{*}', {
  markers: [
    { label: 'Warning', value: 'y = 80', display_type: 'warning dashed' },
  ],
})
```

## Quick Reference

### Visualization Widgets

| Widget | Purpose | Common Use Cases |
|--------|---------|------------------|
| **timeseries** | Line/bar/area charts over time | CPU usage, request rates, latency trends |
| **queryValue** | Single number display | Current count, gauge, percentage |
| **toplist** | Ranked list of values | Top hosts, busiest services, highest errors |
| **heatmap** | Density plot by color | Latency distributions, host metrics |
| **distribution** | Histogram (APM focused) | Request duration distributions |
| **change** | Period-over-period comparison | Day-over-day growth, trends |
| **table** | Multi-column tabular data | Service health matrix, resource inventory |

### Organizational Widgets

| Widget | Purpose | Common Use Cases |
|--------|---------|------------------|
| **note** | Text/markdown display | Section headers, documentation, alerts |
| **group** | Container for widgets | Logical grouping, collapsible sections |

## Widget Details

### timeseries

Display metric trends over time as lines, bars, or areas.

**Simplest usage:**
```jsonnet
widgets.timeseries('CPU Usage', 'avg:system.cpu{*}')
```

**Common options:**
```jsonnet
widgets.timeseries('CPU Usage', 'avg:system.cpu{*}', {
  display_type: 'area',           // 'line' | 'bars' | 'area'
  palette: 'warm',                 // 'dog_classic' | 'warm' | 'cool' | 'purple' | 'orange'
  line_width: 'thick',             // 'thin' | 'normal' | 'thick'
  show_legend: true,               // Show legend
})
```

**Advanced - with SLO markers:**
```jsonnet
widgets.timeseries('Response Time', 'avg:response.time{*}', {
  markers: [
    { label: 'Target', value: 'y = 200', display_type: 'ok dashed' },
    { label: 'Warning', value: 'y = 500', display_type: 'warning dashed' },
    { label: 'Critical', value: 'y = 1000', display_type: 'error dashed' },
  ],
})
```

**LLM search keywords:** timeseries, line chart, bar chart, area chart, graph, trend, time series

---

### queryValue

Display a single metric value as a large number.

**Simplest usage:**
```jsonnet
widgets.queryValue('Active Users', 'sum:app.users.active{*}')
```

**Common options:**
```jsonnet
widgets.queryValue('Error Rate', 'avg:app.errors.rate{*}', {
  precision: 1,                    // Decimal places
  custom_unit: '%',                // Custom unit display
  autoscale: false,                // Disable auto-scaling (1000 → 1K)
  aggregator: 'last',              // 'avg' | 'sum' | 'min' | 'max' | 'last'
})
```

**LLM search keywords:** query value, single number, gauge, counter, metric, current value

---

### toplist

Display ranked list of metric values (top N).

**Simplest usage:**
```jsonnet
widgets.toplist('Top Hosts by CPU', 'avg:system.cpu{*} by {host}')
```

**Common options:**
```jsonnet
widgets.toplist('Busiest Services', 'sum:requests{*} by {service}', {
  title_size: '18',
})
```

**LLM search keywords:** toplist, top list, ranked list, top n, leaderboard

---

### note

Display text, markdown, or section headers.

**Simplest usage:**
```jsonnet
widgets.note('## Performance Metrics')
```

**Common options:**
```jsonnet
widgets.note('## System Health\n\nAll metrics updated every 60s', {
  background_color: 'blue',        // 'white' | 'blue' | 'purple' | 'gray' | 'yellow' | 'red'
  font_size: '18',                 // '14' | '16' | '18' | '20'
  text_align: 'center',            // 'left' | 'center' | 'right'
  vertical_align: 'center',        // 'top' | 'center' | 'bottom'
})
```

**LLM search keywords:** note, text, markdown, header, section, documentation

---

### heatmap

Display metric distribution as a density plot (color intensity).

**Simplest usage:**
```jsonnet
widgets.heatmap('Request Latency', 'avg:request.duration{*} by {host}')
```

**Common options:**
```jsonnet
widgets.heatmap('Response Time Distribution', 'avg:response.time{*}', {
  palette: 'warm',                 // 'dog_classic' | 'warm' | 'cool' | 'purple'
  show_legend: true,
  scale: 'log',                    // 'linear' | 'log' | 'sqrt'
})
```

**LLM search keywords:** heatmap, heat map, density plot, distribution, color intensity

---

### change

Display metric change compared to a previous time period.

**Simplest usage:**
```jsonnet
widgets.change('Daily Active Users Change', 'avg:users.active{*}')
```

**Common options:**
```jsonnet
widgets.change('Weekly Revenue Growth', 'sum:revenue{*}', {
  compare_to: 'week_before',       // 'hour_before' | 'day_before' | 'week_before' | 'month_before'
  increase_good: true,             // Green for increase (vs decrease)
  show_present: true,              // Show current value
})
```

**Advanced - errors (increase is bad):**
```jsonnet
widgets.change('Error Rate Change', 'avg:errors.rate{*}', {
  compare_to: 'day_before',
  increase_good: false,            // Increase in errors is bad
})
```

**LLM search keywords:** change, comparison, period over period, growth, trend, delta

---

### distribution

Display histogram of metric distribution (APM/tracing focused).

**Simplest usage:**
```jsonnet
widgets.distribution('Latency Distribution', 'avg:trace.duration{*}')
```

**Common options:**
```jsonnet
widgets.distribution('Request Duration', 'avg:trace.duration{*}', {
  service: 'web-api',              // APM service name
  env: 'production',               // Environment
  stat: 'p95',                     // 'avg' | 'p50' | 'p75' | 'p90' | 'p95' | 'p99'
  palette: 'warm',
})
```

**Note:** Primarily for APM/tracing data.

**LLM search keywords:** distribution, histogram, APM, tracing, latency distribution

---

### table

Display metrics in tabular format with multiple columns.

**Simplest usage (single query):**
```jsonnet
widgets.table('Service Health', 'avg:service.requests{*} by {service}')
```

**Advanced (multiple queries with aliases):**
```jsonnet
widgets.table('Service Metrics', [
  { query: 'avg:requests{*} by {service}', alias: 'Requests', aggregator: 'sum' },
  { query: 'avg:errors{*} by {service}', alias: 'Errors', aggregator: 'sum' },
  { query: 'avg:latency{*} by {service}', alias: 'Latency', aggregator: 'avg' },
])
```

**Common options:**
```jsonnet
widgets.table('Top Services', 'avg:requests{*} by {service}', {
  has_search_bar: 'always',        // 'auto' | 'always' | 'never'
})
```

**LLM search keywords:** table, tabular, multi-column, matrix, comparison

---

### group

Container for organizing related widgets into sections.

**Simplest usage:**
```jsonnet
widgets.group('Database Metrics', [
  cpuWidget,
  memoryWidget,
  diskWidget,
])
```

**Common options:**
```jsonnet
widgets.group('API Metrics', apiWidgets, {
  background_color: 'vivid_purple', // 'vivid_blue' | 'vivid_purple' | 'vivid_pink' | 'gray'
  layout_type: 'ordered',           // 'ordered' | 'free'
  show_title: true,
})
```

**LLM search keywords:** group, container, section, organize, collapsible

---

## Complete Example

```jsonnet
local layouts = import '../lib/layouts.libsonnet';
local widgets = import '../lib/widgets.libsonnet';

layouts.grid(
  'Application Dashboard',
  std.flattenArrays([
    // Row 1: Key metrics (4 equal-width widgets)
    layouts.row(0, [
      widgets.queryValue('Active Users', 'sum:users.active{*}', {
        precision: 0,
        aggregator: 'last',
      }),
      widgets.queryValue('Requests/min', 'sum:requests{*}', {
        precision: 0,
        autoscale: true,
      }),
      widgets.queryValue('Error Rate', 'avg:errors.rate{*}', {
        precision: 2,
        custom_unit: '%',
      }),
      widgets.queryValue('Avg Latency', 'avg:latency{*}', {
        precision: 0,
        custom_unit: 'ms',
      }),
    ], height=2),

    // Row 2: Section header
    [layouts.fullWidth(2, widgets.note('## Performance Trends'), height=1)],

    // Row 3: Two timeseries charts
    layouts.row(3, [
      widgets.timeseries('Request Rate', 'sum:requests{*}', {
        display_type: 'bars',
        palette: 'purple',
      }),
      widgets.timeseries('Response Time', 'avg:latency{*}', {
        display_type: 'line',
        markers: [
          { label: 'SLO', value: 'y = 200', display_type: 'ok dashed' },
        ],
      }),
    ], height=3),

    // Row 4: Top lists
    layouts.row(6, [
      widgets.toplist('Top Endpoints', 'sum:requests{*} by {endpoint}'),
      widgets.toplist('Slowest Endpoints', 'avg:latency{*} by {endpoint}'),
    ], height=3),
  ]),
  {
    description: 'Application performance and health metrics',
  }
)
```

## Best Practices

### 1. Use Semantic Names

```jsonnet
// Good
widgets.queryValue('Active Sessions', 'sum:sessions.active{*}')

// Avoid
widgets.queryValue('Metric 1', 'sum:sessions.active{*}')
```

### 2. Group Related Widgets

```jsonnet
local apiMetrics = [
  widgets.queryValue('API Requests', 'sum:api.requests{*}'),
  widgets.queryValue('API Latency', 'avg:api.latency{*}'),
  widgets.queryValue('API Errors', 'sum:api.errors{*}'),
];

widgets.group('API Health', apiMetrics, { background_color: 'vivid_blue' })
```

### 3. Use Markers for SLOs

```jsonnet
widgets.timeseries('Error Rate', 'avg:errors.rate{*}', {
  markers: [
    { label: 'Target: < 0.1%', value: 'y = 0.1', display_type: 'ok dashed' },
    { label: 'Critical: > 1%', value: 'y = 1.0', display_type: 'error dashed' },
  ],
})
```

### 4. Consistent Color Schemes

```jsonnet
// Use consistent palettes for related metrics
local errorPalette = { palette: 'warm' };
local successPalette = { palette: 'cool' };

widgets.timeseries('Errors', 'sum:errors{*}', errorPalette)
widgets.timeseries('Success', 'sum:success{*}', successPalette)
```

### 5. Clear Section Headers

```jsonnet
layouts.row(y, [
  layouts.fullWidth(y, widgets.note('## Database Performance', {
    background_color: 'gray',
    text_align: 'center',
    font_size: '18',
  }), height=1),
])
```

## LLM Integration

This library is designed to be LLM-friendly:

- **Inline documentation** with `@` tags for easy parsing
- **Search keywords** for natural language queries
- **Progressive examples** from simple to advanced
- **Type-safe enums** with all valid values documented
- **Consistent patterns** across all widgets

Example LLM prompts that work well:

- "Create a timeseries widget for CPU usage with a warning threshold at 80%"
- "Show me how to make a query value widget for error rate with 1 decimal place"
- "Create a dashboard with 4 key metrics in a row and 2 charts below"

## Related Documentation

- [Dashboard System Overview](../README.md) - Quick start and common patterns
- [Preset Widgets](./PRESETS.md) - Pre-configured common patterns
- [Layout Templates](./LAYOUTS.md) - Grid, row, and positioning helpers
- [Design Document](../DESIGN.md) - Overall architecture and vision
- [Datadog Widget Docs](https://docs.datadoghq.com/dashboards/widgets/)

## Building and Testing

```bash
# Compile Jsonnet to JSON
jsonnet dashboards/sources/my_dashboard.jsonnet > output.json

# Validate JSON structure
jq . output.json > /dev/null && echo "Valid JSON"

# Push to Datadog
./devops/datadog/scripts/push_dashboard.py output.json
```
