# Datadog Widget Presets

Pre-configured widgets for common monitoring patterns with smart defaults.

## Purpose

Presets eliminate repetitive configuration by providing battle-tested widget configurations for common use cases. Each preset includes:

- **Smart defaults**: Appropriate display types, colors, and scales
- **SLO markers**: Industry-standard thresholds where applicable
- **Consistent styling**: Coordinated color palettes and formatting

## Quick Start

```jsonnet
local layouts = import '../lib/layouts.libsonnet';
local presets = import '../lib/presets.libsonnet';

layouts.grid(
  'My Dashboard',
  std.flattenArrays([
    layouts.row(0, [
      presets.cpuTimeseries(),              // CPU with 80%/95% thresholds
      presets.memoryTimeseries(),           // Memory with 80%/90% thresholds
      presets.errorRateTimeseries(),        // Errors with 0.1%/1% SLO
      presets.latencyTimeseries(),          // Latency with 200ms/500ms SLO
    ], height=3),
  ])
)
```

## Infrastructure Presets

### CPU Monitoring

**cpuTimeseries(title, query)**

Displays CPU usage with warning (80%) and critical (95%) thresholds.

```jsonnet
// Default
presets.cpuTimeseries()

// Custom query
presets.cpuTimeseries('API Server CPU', 'avg:system.cpu.user{service:api}')
```

**Features:**
- Area chart with warm palette
- 80% warning threshold (orange dashed line)
- 95% critical threshold (red dashed line)

---

### Memory Monitoring

**memoryTimeseries(title, query)**

Displays memory usage with warning (80%) and critical (90%) thresholds.

```jsonnet
// Default
presets.memoryTimeseries()

// Custom query
presets.memoryTimeseries('Database Memory', 'avg:system.mem.used{service:db} / avg:system.mem.total{service:db} * 100')
```

**Features:**
- Area chart with purple palette
- 80% warning threshold
- 90% critical threshold

---

### Disk Usage

**diskUsageGauge(title, query)**

Single-value display for current disk usage percentage.

```jsonnet
presets.diskUsageGauge('Disk Usage', 'avg:system.disk.in_use{*}')
```

**Features:**
- No decimal places
- Percentage unit
- Last value aggregation

---

### Network Throughput

**networkThroughput(title, query)**

Line chart for network bytes in/out.

```jsonnet
presets.networkThroughput('Network In', 'avg:system.net.bytes_rcvd{*}')
```

**Features:**
- Line chart with cool palette
- Legend enabled for multiple series

---

## Application Presets

### Error Rate Monitoring

**errorRateTimeseries(title, query)**

Bar chart for error rate with SLO thresholds.

```jsonnet
presets.errorRateTimeseries('API Error Rate', 'avg:api.errors.rate{*}')
```

**Features:**
- Bar chart with warm palette
- 0.1% target (green dashed line)
- 1% SLO breach (red dashed line)

**errorRateGauge(title, query)**

Single-value display for current error rate.

```jsonnet
presets.errorRateGauge('Current Errors', 'avg:errors.rate{*}')
```

**Features:**
- 2 decimal places
- Percentage unit
- Average aggregation

---

### Latency Monitoring

**latencyTimeseries(title, query)**

Line chart for response times with SLO thresholds.

```jsonnet
presets.latencyTimeseries('API Latency', 'avg:api.latency{*}')
```

**Features:**
- Line chart with purple palette, thick lines
- 200ms SLO (green dashed line)
- 500ms warning (orange dashed line)

**latencyGauge(title, query)**

Single-value display for average latency.

```jsonnet
presets.latencyGauge('Avg Response Time', 'avg:api.latency{*}')
```

**Features:**
- No decimal places
- Millisecond unit
- Average aggregation

---

### Request Rate Monitoring

**requestRateTimeseries(title, query)**

Bar chart for throughput over time.

```jsonnet
presets.requestRateTimeseries('Requests/sec', 'sum:requests{*}.as_rate()')
```

**Features:**
- Bar chart with cool palette

**requestCountGauge(title, query)**

Single-value display for total request count.

```jsonnet
presets.requestCountGauge('Total Requests', 'sum:requests{*}')
```

**Features:**
- No decimal places
- Auto-scaling (1000 → 1K)
- Sum aggregation

---

### Success Rate Monitoring

**successRateTimeseries(title, query)**

Line chart for uptime/availability with SLO.

```jsonnet
presets.successRateTimeseries('Availability', '(sum:requests{*} - sum:errors{*}) / sum:requests{*} * 100')
```

**Features:**
- Line chart with cool palette, thick lines
- 99% SLO (green dashed line)
- 95% critical (red dashed line)

---

## Business Presets

### User Metrics

**activeUsersGauge(title, query)**

Current active user count.

```jsonnet
presets.activeUsersGauge('Active Sessions', 'sum:sessions.active{*}')
```

**userGrowthChange(title, query)**

Day-over-day user growth.

```jsonnet
presets.userGrowthChange('Daily Growth', 'sum:users.active{*}')
```

**Features:**
- Day-over-day comparison
- Increase shown as positive (green)
- Shows current value

---

### Conversion Tracking

**conversionRateGauge(title, query)**

Conversion funnel percentage.

```jsonnet
presets.conversionRateGauge('Cart → Purchase', 'sum:purchases{*} / sum:cart_adds{*} * 100')
```

**Features:**
- 1 decimal place
- Percentage unit
- Average aggregation

---

### Revenue Tracking

**revenueTimeseries(title, query)**

Bar chart for revenue over time.

```jsonnet
presets.revenueTimeseries('Daily Revenue', 'sum:revenue{*}')
```

**Features:**
- Bar chart with cool palette

---

## Top Lists

### Service Rankings

**topServicesByRequests(title, query)**

Services ranked by request count.

```jsonnet
presets.topServicesByRequests('Busiest Services', 'sum:requests{*} by {service}')
```

**topEndpointsByLatency(title, query)**

Endpoints ranked by latency (slowest first).

```jsonnet
presets.topEndpointsByLatency('Slowest Endpoints', 'avg:latency{*} by {endpoint}')
```

---

### Host Rankings

**topHostsByCPU(title, query)**

Hosts ranked by CPU usage.

```jsonnet
presets.topHostsByCPU('Top CPU Hosts', 'avg:system.cpu.user{*} by {host}')
```

**topHostsByMemory(title, query)**

Hosts ranked by memory usage.

```jsonnet
presets.topHostsByMemory('Top Memory Hosts', 'avg:system.mem.used{*} by {host}')
```

---

## Section Headers

### sectionHeader(title, description)

Large blue header for major sections.

```jsonnet
presets.sectionHeader('Performance Metrics', 'Real-time application performance')
```

**Features:**
- Blue background
- 18pt font
- Center-aligned
- Optional description

---

### subsectionHeader(title)

Smaller gray header for subsections.

```jsonnet
presets.subsectionHeader('Database Queries')
```

**Features:**
- Gray background
- 16pt font
- Left-aligned

---

### alertNote(message)

Red alert/warning message.

```jsonnet
presets.alertNote('High error rate detected - investigate immediately')
```

**Features:**
- Red background
- Bold "Alert:" prefix
- Center-aligned

---

### infoNote(message)

Plain informational note.

```jsonnet
presets.infoNote('Metrics updated every 60 seconds')
```

**Features:**
- White background
- Small font
- Left-aligned

---

## Health Tables

### serviceHealthTable(title)

Multi-metric service health table.

```jsonnet
presets.serviceHealthTable('Service Health')
```

**Columns:**
- Requests (sum)
- Latency (ms, avg)
- Errors (sum)

**Features:**
- Automatic search bar
- Grouped by service

---

### hostHealthTable(title)

Multi-metric host health table.

```jsonnet
presets.hostHealthTable('Host Health')
```

**Columns:**
- CPU % (avg)
- Memory % (avg)
- Disk % (avg)

**Features:**
- Automatic search bar
- Grouped by host

---

## Complete Example

```jsonnet
local layouts = import '../lib/layouts.libsonnet';
local presets = import '../lib/presets.libsonnet';

layouts.grid(
  'Service Dashboard',
  std.flattenArrays([
    // Header
    [layouts.fullWidth(0, presets.sectionHeader(
      'Service Health Overview',
      'Real-time monitoring of all microservices'
    ), height=1)],

    // Key metrics row
    layouts.row(1, [
      presets.activeUsersGauge('Active Users', 'sum:users.active{*}'),
      presets.requestCountGauge('Total Requests', 'sum:requests{*}'),
      presets.errorRateGauge('Error Rate', 'avg:errors.rate{*}'),
      presets.latencyGauge('Avg Latency', 'avg:latency{*}'),
    ], height=2),

    // Performance trends
    [layouts.fullWidth(3, presets.subsectionHeader('Performance Trends'), height=1)],

    layouts.row(4, [
      presets.requestRateTimeseries('Request Rate', 'sum:requests{*}.as_rate()'),
      presets.latencyTimeseries('P95 Latency', 'p95:latency{*}'),
    ], height=3),

    // Error monitoring
    [layouts.fullWidth(7, presets.subsectionHeader('Error Monitoring'), height=1)],

    layouts.row(8, [
      presets.errorRateTimeseries('Error Rate', 'avg:errors.rate{*}'),
      presets.successRateTimeseries('Success Rate', '(sum:requests{*} - sum:errors{*}) / sum:requests{*} * 100'),
    ], height=3),

    // Infrastructure
    [layouts.fullWidth(11, presets.subsectionHeader('Infrastructure'), height=1)],

    layouts.row(12, [
      presets.cpuTimeseries('CPU Usage', 'avg:system.cpu.user{*}'),
      presets.memoryTimeseries('Memory Usage', 'avg:system.mem.pct_usable{*}'),
    ], height=3),

    // Rankings
    [layouts.fullWidth(15, presets.subsectionHeader('Service Rankings'), height=1)],

    layouts.row(16, [
      presets.topServicesByRequests('Busiest Services', 'sum:requests{*} by {service}'),
      presets.topEndpointsByLatency('Slowest Endpoints', 'avg:latency{*} by {endpoint}'),
    ], height=3),

    // Health table
    [layouts.fullWidth(19, presets.subsectionHeader('Detailed Health'), height=1)],
    [layouts.fullWidth(20, presets.serviceHealthTable('All Services'), height=4)],
  ]),
  {
    description: 'Comprehensive service monitoring dashboard',
  }
)
```

## Customization

All presets accept custom queries while maintaining their smart defaults:

```jsonnet
// Use preset defaults
presets.cpuTimeseries()

// Override query only
presets.cpuTimeseries('Custom CPU', 'avg:system.cpu.user{env:prod}')

// Mix preset with custom query
presets.latencyTimeseries('API Latency', 'p99:api.latency{endpoint:/users}')
```

## Design Principles

1. **Smart defaults**: Each preset has been tuned for its use case
2. **Industry standards**: Thresholds based on common SLOs
3. **Visual consistency**: Coordinated colors and styles
4. **Easy customization**: Override query while keeping styling

## Related Documentation

- [Dashboard System Overview](../README.md) - Quick start and common patterns
- [Widget Primitives](./WIDGETS.md) - Low-level widget builders
- [Layout Templates](./LAYOUTS.md) - Positioning and organization
- [Design Document](../DESIGN.md) - Overall architecture

## Building

```bash
# Compile dashboard with presets
jsonnet dashboards/sources/my_dashboard.jsonnet > output.json

# Validate
jq . output.json > /dev/null && echo "Valid JSON"

# Push to Datadog
./devops/datadog/scripts/push_dashboard.py output.json
```
